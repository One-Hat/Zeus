"""End-to-end tests for Sandbox Runner, Agents, Bandit Router, and LangGraph Pipeline."""

import os
from pathlib import Path
import pytest

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
from forge.graph import ForgePipeline
from forge.llm import LLMClient
from forge.router import BanditRouter
from forge.sandbox.runner import DockerSandboxRunner


def test_sandbox_hello_world_and_timeout(tmp_path: Path):
    runner = DockerSandboxRunner(workspace_dir=tmp_path, use_docker=False, default_timeout=5)
    written = runner.write_files({"hello.py": "print('hello from forge sandbox')"})
    assert "hello.py" in written
    assert runner.read_file("hello.py") == "print('hello from forge sandbox')"

    res = runner.exec_command('python hello.py')
    assert res.exit_code == 0
    assert "hello from forge sandbox" in res.stdout
    assert res.timed_out is False

    # Enforce path traversal protection
    with pytest.raises(ValueError):
        runner.write_files({"../outside.py": "bad"})


def test_planner_on_sample_ideas(monkeypatch):
    monkeypatch.setenv("FORGE_MOCK_LLM", "1")
    llm = LLMClient()
    planner = PlannerAgent(llm)

    sample_ideas = [
        "A todo app with auth",
        "A personal expense tracker",
        "A customer support ticket system with ML sentiment scoring",
        "A recipe organizer app",
        "A markdown note-taking app",
    ]
    for idea in sample_ideas:
        llm.mock_responses.pop("planner", None)
        spec = planner.run(idea)
        assert spec.idea == idea
        assert "openapi" in spec.api_contract
        assert len(spec.tasks) >= 2
        assert len(spec.acceptance_tests) >= 1


def test_bandit_router_and_prompt_evolution(tmp_path: Path):
    score_file = tmp_path / "score_table.json"
    router = BanditRouter(score_table_path=score_file, epsilon=0.0, seed=42)

    arm = router.select_arm("planner", exploit_only=True)
    assert arm.role == "planner"

    updated = router.update_arm(
        role="planner",
        model=arm.model,
        prompt_version=arm.prompt_version,
        pass_rate=1.0,
        cost_usd=0.002,
        latency_seconds=0.8,
    )
    assert updated.pulls == 1
    assert score_file.exists()

    new_arms = router.add_prompt_version("backend", "v2_refined", "Check all status codes")
    assert len(new_arms) >= 1


def test_deployer_staging_and_production_gate():
    deployer = DeployerAgent()
    spec = PlannerAgent.build_default_spec("todo app")
    llm = LLMClient()
    sandbox = DockerSandboxRunner(use_docker=False)
    try:
        BackendWorker(llm, sandbox).run(spec)
        report = IntegratorAgent(sandbox).run("run-deploy-test", spec)
        assert report.success is True

        # Staging succeeds automatically
        staging_res = deployer.run(spec, report, target_env="staging")
        assert staging_res.deployed is True
        assert staging_res.staging_url is not None

        # Production is blocked without human approval
        prod_blocked = deployer.run(spec, report, target_env="production", human_approved=False)
        assert prod_blocked.deployed is False
        assert prod_blocked.requires_human_approval is True

        # Production succeeds with explicit human approval
        prod_approved = deployer.run(spec, report, target_env="production", human_approved=True)
        assert prod_approved.deployed is True
        assert prod_approved.approval_granted is True
    finally:
        sandbox.cleanup()


def test_full_langgraph_pipeline_end_to_end(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FORGE_MOCK_LLM", "1")
    llm = LLMClient()
    sandbox = DockerSandboxRunner(workspace_dir=tmp_path / "ws", use_docker=False)
    router = BanditRouter(score_table_path=tmp_path / "scores.json")
    runs_dir = tmp_path / "runs"

    pipeline = ForgePipeline(
        llm_client=llm,
        sandbox=sandbox,
        router=router,
        runs_dir=runs_dir,
    )
    state = pipeline.run("a todo app with auth", run_id="test_e2e_01")

    assert state["report"].success is True
    assert state["deploy_result"].deployed is True
    assert state["score"].pass_rate == 1.0
    assert (runs_dir / "test_e2e_01" / "spec.json").exists()
    assert (runs_dir / "test_e2e_01" / "report.json").exists()
    assert (runs_dir / "test_e2e_01" / "score.json").exists()
    assert (runs_dir / "test_e2e_01" / "deploy.json").exists()
