"""Benchmark runner: compares single baseline model vs. routed multi-LLM chain across 20 app ideas."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import yaml

from forge.graph import ForgePipeline
from forge.llm import LLMClient
from forge.router import BanditRouter
from forge.sandbox.runner import DockerSandboxRunner


def load_prompts(prompts_path: Path | None = None) -> List[str]:
    path = prompts_path or (Path(__file__).parent / "prompts.json")
    return json.loads(path.read_text(encoding="utf-8"))


def run_suite(
    prompts: List[str],
    mode: str,
    baseline_model: str | None = None,
) -> Dict[str, Any]:
    passes = 0
    total_cost = 0.0
    total_time = 0.0
    total_fix_loops = 0

    router = BanditRouter(epsilon=0.15)
    for idx, idea in enumerate(prompts):
        llm = LLMClient()
        sandbox = DockerSandboxRunner()
        try:
            pipeline = ForgePipeline(
                llm_client=llm,
                sandbox=sandbox,
                router=router,
                single_model_override=baseline_model if mode == "baseline" else None,
            )
            state = pipeline.run(idea, run_id=f"bench_{mode}_{idx+1:02d}")
            report = state["report"]
            if report.success:
                passes += 1
            total_cost += report.total_cost_usd
            total_time += report.wall_time_seconds
            total_fix_loops += report.fix_iterations
        finally:
            sandbox.cleanup()

    n = max(1, len(prompts))
    return {
        "mode": mode,
        "total_prompts": len(prompts),
        "pass_rate": round((passes / n) * 100, 1),
        "avg_cost_usd": round(total_cost / n, 5),
        "total_cost_usd": round(total_cost, 4),
        "avg_wall_time_s": round(total_time / n, 2),
        "avg_fix_loops": round(total_fix_loops / n, 2),
    }


def format_markdown_table(baseline: Dict[str, Any], routed: Dict[str, Any]) -> str:
    lines = [
        "| Configuration | Prompts | Pass Rate (%) | Avg Cost ($/run) | Total Cost ($) | Avg Wall Time (s) | Avg Fix Loops |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Single Strong Model (`{baseline.get('model_name', 'baseline')}`) | {baseline['total_prompts']} | {baseline['pass_rate']}% | ${baseline['avg_cost_usd']:.5f} | ${baseline['total_cost_usd']:.4f} | {baseline['avg_wall_time_s']}s | {baseline['avg_fix_loops']} |",
        f"| Forge Bandit-Routed Multi-LLM Chain | {routed['total_prompts']} | {routed['pass_rate']}% | ${routed['avg_cost_usd']:.5f} | ${routed['total_cost_usd']:.4f} | {routed['avg_wall_time_s']}s | {routed['avg_fix_loops']} |",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Forge 20-prompt benchmark comparison")
    parser.add_argument("--limit", type=int, default=20, help="Number of prompts to evaluate (default: 20)")
    parser.add_argument("--mock", action="store_true", help="Run benchmark using deterministic mock mode")
    args = parser.parse_args()

    if args.mock:
        os.environ["FORGE_MOCK_LLM"] = "1"

    cfg_path = Path(__file__).parent.parent / "config" / "routing.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    baseline_model = cfg.get("benchmark", {}).get("baseline_model", "gpt-4o")

    prompts = load_prompts()[: args.limit]
    baseline_stats = run_suite(prompts, mode="baseline", baseline_model=baseline_model)
    baseline_stats["model_name"] = baseline_model
    routed_stats = run_suite(prompts, mode="routed")

    md = format_markdown_table(baseline_stats, routed_stats)
    print("\n### Benchmark Results\n")
    print(md)


if __name__ == "__main__":
    main()
