"""CLI entrypoint for Forge (`forge run "idea"`)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from forge.graph import ForgePipeline
from forge.llm import LLMClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Forge: a multi-LLM pipeline that turns one sentence into a tested, staged web app.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Generate, test, and stage a web app from a one-sentence idea")
    run_parser.add_argument("idea", type=str, help="One-sentence application idea")
    run_parser.add_argument("--budget", type=float, default=None, help="Override max budget in USD")
    run_parser.add_argument("--mock", action="store_true", help="Run using deterministic mock LLM responses")

    args = parser.parse_args(argv)

    if args.command == "run":
        if args.mock:
            os.environ["FORGE_MOCK_LLM"] = "1"

        llm = LLMClient(max_budget_usd=args.budget)
        pipeline = ForgePipeline(llm_client=llm)
        state = pipeline.run(args.idea)

        report = state["report"]
        deploy = state["deploy_result"]
        score = state["score"]

        summary = {
            "run_id": state["run_id"],
            "idea": args.idea,
            "success": report.success,
            "staging_url": deploy.staging_url,
            "cost_usd": report.total_cost_usd,
            "wall_time_seconds": report.wall_time_seconds,
            "fix_iterations": report.fix_iterations,
            "score": score.score,
            "pass_rate": score.pass_rate,
            "artifacts_dir": str(pipeline.runs_dir / state["run_id"]),
        }
        print(json.dumps(summary, indent=2))
        return 0 if report.success else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
