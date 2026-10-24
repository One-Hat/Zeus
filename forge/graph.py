"""LangGraph orchestration pipeline for Forge."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict
from langgraph.graph import END, START, StateGraph

from forge.agents import (
    BackendWorker,
    DeployerAgent,
    FixerAgent,
    FrontendWorker,
    IntegratorAgent,
    MLWorker,
    PlannerAgent,
    ReviewerAgent,
)
from forge.llm import LLMClient
from forge.router import BanditRouter
from forge.sandbox.runner import DockerSandboxRunner
from forge.schemas import DeployResult, Report, Score, Spec, WorkerOutput


class ForgeState(TypedDict, total=False):
    run_id: str
    idea: str
    spec: Spec
    backend_output: WorkerOutput
    frontend_output: WorkerOutput
    ml_output: WorkerOutput
    report: Report
    fix_iterations: int
    max_fix_loops: int
    score: Score
    deploy_result: DeployResult
    start_time: float


class ForgePipeline:
    """End-to-end multi-LLM application generator orchestrated with LangGraph."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        sandbox: Optional[DockerSandboxRunner] = None,
        router: Optional[BanditRouter] = None,
        runs_dir: Optional[str | Path] = None,
        single_model_override: Optional[str] = None,
    ):
        self.llm = llm_client or LLMClient()
        self.sandbox = sandbox or DockerSandboxRunner()
        self.router = router or BanditRouter()
        self.runs_dir = Path(runs_dir) if runs_dir else Path(__file__).parent / "runs"
        self.single_model_override = single_model_override

        self.planner = PlannerAgent(self.llm)
        self.backend_worker = BackendWorker(self.llm, self.sandbox)
        self.frontend_worker = FrontendWorker(self.llm, self.sandbox)
        self.ml_worker = MLWorker(self.llm, self.sandbox)
        self.integrator = IntegratorAgent(self.sandbox)
        self.fixer = FixerAgent(self.llm, self.sandbox)
        self.reviewer = ReviewerAgent(self.llm)
        self.deployer = DeployerAgent(self.sandbox)

        self.graph = self._build_graph()

    def _model_for_role(self, role: str) -> Optional[str]:
        if self.single_model_override:
            return self.single_model_override
        arm = self.router.select_arm(role)
        return arm.model

    def _plan_node(self, state: ForgeState) -> Dict[str, Any]:
        spec = self.planner.run(state["idea"], override_model=self._model_for_role("planner"))
        return {"spec": spec}

    def _backend_node(self, state: ForgeState) -> Dict[str, Any]:
        out = self.backend_worker.run(state["spec"], override_model=self._model_for_role("backend"))
        return {"backend_output": out}

    def _frontend_node(self, state: ForgeState) -> Dict[str, Any]:
        out = self.frontend_worker.run(state["spec"], override_model=self._model_for_role("frontend"))
        return {"frontend_output": out}

    def _ml_node(self, state: ForgeState) -> Dict[str, Any]:
        out = self.ml_worker.run(state["spec"], override_model=self._model_for_role("ml"))
        return {"ml_output": out}

    def _integrator_node(self, state: ForgeState) -> Dict[str, Any]:
        fix_iters = state.get("fix_iterations", 0)
        report = self.integrator.run(
            run_id=state["run_id"],
            spec=state["spec"],
            fix_iterations=fix_iters,
            total_cost_usd=self.llm.total_cost_usd,
        )
        elapsed = time.time() - state.get("start_time", time.time())
        report.wall_time_seconds = round(elapsed, 3)
        return {"report": report}

    def _fixer_node(self, state: ForgeState) -> Dict[str, Any]:
        fix_iters = state.get("fix_iterations", 0) + 1
        self.fixer.run(
            spec=state["spec"],
            report=state["report"],
            override_model=self._model_for_role("fixer"),
        )
        return {"fix_iterations": fix_iters}

    def _reviewer_node(self, state: ForgeState) -> Dict[str, Any]:
        report = state["report"]
        report.total_cost_usd = round(self.llm.total_cost_usd, 6)
        score = self.reviewer.run(
            spec=state["spec"],
            report=report,
            override_model=self._model_for_role("reviewer"),
        )

        # Update bandit arms and register any Reviewer-proposed prompt versions
        for call in self.llm.call_history:
            role_cfg = self.llm.get_role_config(call.role)
            p_ver = role_cfg.get("prompt_version", "v1")
            self.router.update_arm(
                role=call.role,
                model=call.model,
                prompt_version=p_ver,
                pass_rate=score.pass_rate,
                cost_usd=call.cost_usd,
                latency_seconds=call.latency_seconds,
            )

        for role, suggestion in score.proposed_prompt_versions.items():
            new_ver = f"v_auto_{state['run_id'][:6]}"
            self.router.add_prompt_version(role=role, prompt_version=new_ver, prompt_template=suggestion)

        return {"score": score, "report": report}

    def _deployer_node(self, state: ForgeState) -> Dict[str, Any]:
        deploy_res = self.deployer.run(
            spec=state["spec"],
            report=state["report"],
            target_env="staging",
        )
        return {"deploy_result": deploy_res}

    def _should_fix_or_review(self, state: ForgeState) -> str:
        report = state["report"]
        fix_iters = state.get("fix_iterations", 0)
        max_loops = state.get("max_fix_loops", 3)
        if report.success:
            return "reviewer"
        if fix_iters < max_loops:
            return "fixer"
        return "reviewer"

    def _build_graph(self):
        builder = StateGraph(ForgeState)

        builder.add_node("planner", self._plan_node)
        builder.add_node("backend_worker", self._backend_node)
        builder.add_node("frontend_worker", self._frontend_node)
        builder.add_node("ml_worker", self._ml_node)
        builder.add_node("integrator", self._integrator_node)
        builder.add_node("fixer", self._fixer_node)
        builder.add_node("reviewer", self._reviewer_node)
        builder.add_node("deployer", self._deployer_node)

        builder.add_edge(START, "planner")
        # Parallel fan-out from planner to workers
        builder.add_edge("planner", "backend_worker")
        builder.add_edge("planner", "frontend_worker")
        builder.add_edge("planner", "ml_worker")

        # Fan-in to integrator
        builder.add_edge("backend_worker", "integrator")
        builder.add_edge("frontend_worker", "integrator")
        builder.add_edge("ml_worker", "integrator")

        # Fixer loop with hard cap
        builder.add_conditional_edges(
            "integrator",
            self._should_fix_or_review,
            {
                "fixer": "fixer",
                "reviewer": "reviewer",
            },
        )
        builder.add_edge("fixer", "integrator")
        builder.add_edge("reviewer", "deployer")
        builder.add_edge("deployer", END)

        return builder.compile()

    def run(self, idea: str, run_id: Optional[str] = None) -> ForgeState:
        """Execute the full Forge pipeline and persist artifacts to runs/<id>/."""
        rid = run_id or uuid.uuid4().hex[:10]
        max_loops = int(self.llm.config.get("max_fix_loops", 3))
        initial_state: ForgeState = {
            "run_id": rid,
            "idea": idea,
            "fix_iterations": 0,
            "max_fix_loops": max_loops,
            "start_time": time.time(),
        }

        final_state: ForgeState = self.graph.invoke(initial_state)  # type: ignore
        self._persist_run(rid, final_state)
        return final_state

    def _persist_run(self, run_id: str, state: ForgeState) -> Path:
        """Save spec, report, score, deploy result, and call logs to runs/<id>/."""
        run_folder = self.runs_dir / run_id
        run_folder.mkdir(parents=True, exist_ok=True)

        if "spec" in state:
            (run_folder / "spec.json").write_text(
                state["spec"].model_dump_json(indent=2), encoding="utf-8"
            )
        if "report" in state:
            (run_folder / "report.json").write_text(
                state["report"].model_dump_json(indent=2), encoding="utf-8"
            )
            (run_folder / "integrator.log").write_text(
                state["report"].logs, encoding="utf-8"
            )
        if "score" in state:
            (run_folder / "score.json").write_text(
                state["score"].model_dump_json(indent=2), encoding="utf-8"
            )
        if "deploy_result" in state:
            (run_folder / "deploy.json").write_text(
                state["deploy_result"].model_dump_json(indent=2), encoding="utf-8"
            )

        calls_data = [c.model_dump() for c in self.llm.call_history]
        (run_folder / "llm_calls.json").write_text(
            json.dumps(calls_data, indent=2), encoding="utf-8"
        )
        return run_folder
