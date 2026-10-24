"""Reviewer Agent: evaluates the run report, scores quality, and proposes new prompt arms."""

from __future__ import annotations

import json
from typing import Optional
from forge.llm import LLMClient
from forge.schemas import Report, Score, Spec

REVIEWER_SYSTEM_PROMPT = """You are the Forge Reviewer Agent.
Inspect the `Spec`, the final `Report`, and LLM call statistics.
Return a structured `Score` with:
- `score` (0.0 to 1.0)
- `pass_rate` (fraction of acceptance tests passed)
- `critique` (concise assessment of code quality, test coverage, and failure causes)
- `failed_decisions` (specific prompt/model/plan issues if any failed)
- `proposed_prompt_versions` (TextGrad/DSPy-style prompt refinements per role)
"""


class ReviewerAgent:
    """Evaluates a completed Forge run and proposes prompt improvements for the bandit router."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    @staticmethod
    def build_default_score(report: Report) -> Score:
        total_tests = len(report.acceptance_results)
        passed_tests = sum(1 for r in report.acceptance_results if r.passed)
        pass_rate = (passed_tests / total_tests) if total_tests > 0 else (1.0 if report.success else 0.0)
        base_score = pass_rate if report.success else max(0.0, pass_rate * 0.5)
        penalty = min(0.3, report.fix_iterations * 0.1)
        final_score = round(max(0.0, min(1.0, base_score - penalty)), 3)

        failed_decisions = []
        proposed = {}
        if not report.success:
            failed_decisions.append(f"Failed verification after {report.fix_iterations} fix loops: {report.error_summary}")
            proposed["backend"] = "Ensure all OpenAPI status codes (e.g. 201 on POST) strictly match Spec acceptance_tests."

        return Score(
            run_id=report.run_id,
            score=final_score,
            pass_rate=round(pass_rate, 3),
            critique=(
                f"Run {'succeeded' if report.success else 'failed'} with {passed_tests}/{total_tests} "
                f"acceptance tests passing and {report.fix_iterations} fix iterations."
            ),
            failed_decisions=failed_decisions,
            proposed_prompt_versions=proposed,
        )

    def run(self, spec: Spec, report: Report, override_model: Optional[str] = None) -> Score:
        self.llm.mock_responses["reviewer"] = self.build_default_score(report)

        payload = {
            "run_id": report.run_id,
            "idea": spec.idea,
            "success": report.success,
            "fix_iterations": report.fix_iterations,
            "total_cost_usd": report.total_cost_usd,
            "acceptance_results": [r.model_dump() for r in report.acceptance_results],
            "error_summary": report.error_summary,
        }
        messages = [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]
        return self.llm.call_structured(
            role="reviewer",
            messages=messages,
            response_model=Score,
            override_model=override_model,
        )
