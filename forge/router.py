"""Epsilon-greedy bandit router and score table for multi-LLM role selection."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pydantic import BaseModel, Field


class BanditArm(BaseModel):
    role: str
    model: str
    prompt_version: str
    pulls: int = 0
    avg_pass_rate: float = 0.5
    avg_cost_usd: float = 0.01
    avg_latency_seconds: float = 1.0
    prompt_template: str = ""

    @property
    def reward(self) -> float:
        """Composite reward balancing pass rate, cost efficiency, and speed."""
        cost_penalty = min(0.3, self.avg_cost_usd * 2.0)
        latency_penalty = min(0.2, self.avg_latency_seconds / 120.0)
        return max(0.0, self.avg_pass_rate - cost_penalty - latency_penalty)


class BanditRouter:
    """Epsilon-greedy contextual bandit managing (role, model, prompt_version) arms."""

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        score_table_path: Optional[str | Path] = None,
        epsilon: float = 0.15,
        seed: Optional[int] = None,
    ):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "routing.yaml"
        self.config_path = Path(config_path)

        if score_table_path is None:
            score_table_path = Path(__file__).parent / "config" / "score_table.json"
        self.score_table_path = Path(score_table_path)

        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self.arms: Dict[str, List[BanditArm]] = {}
        self._load_or_init_arms()

    def _load_or_init_arms(self) -> None:
        if self.score_table_path.exists():
            try:
                data = json.loads(self.score_table_path.read_text(encoding="utf-8"))
                for role, arm_list in data.items():
                    self.arms[role] = [BanditArm.model_validate(a) for a in arm_list]
                return
            except Exception:
                pass

        # Initialize arms from routing.yaml
        if self.config_path.exists():
            cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        else:
            cfg = {}

        roles_cfg = cfg.get("roles", {})
        candidates = cfg.get("benchmark", {}).get(
            "candidate_models", ["gemini/gemini-1.5-flash", "gpt-4o-mini"]
        )

        for role, rcfg in roles_cfg.items():
            default_model = rcfg.get("model", "gemini/gemini-1.5-flash")
            p_ver = rcfg.get("prompt_version", "v1")
            role_arms = [
                BanditArm(role=role, model=default_model, prompt_version=p_ver)
            ]
            for cand in candidates:
                if cand != default_model:
                    role_arms.append(BanditArm(role=role, model=cand, prompt_version=p_ver))
            self.arms[role] = role_arms

    def select_arm(self, role: str, exploit_only: bool = False) -> BanditArm:
        """Select an arm for the given role using epsilon-greedy exploration."""
        role_arms = self.arms.get(role)
        if not role_arms:
            arm = BanditArm(role=role, model="gemini/gemini-1.5-flash", prompt_version="v1")
            self.arms[role] = [arm]
            return arm

        if not exploit_only and self._rng.random() < self.epsilon:
            return self._rng.choice(role_arms)

        return max(role_arms, key=lambda a: (a.reward, -a.pulls))

    def update_arm(
        self,
        role: str,
        model: str,
        prompt_version: str,
        pass_rate: float,
        cost_usd: float,
        latency_seconds: float,
    ) -> BanditArm:
        """Update running averages for an arm after a pipeline run."""
        role_arms = self.arms.setdefault(role, [])
        target: Optional[BanditArm] = None
        for arm in role_arms:
            if arm.model == model and arm.prompt_version == prompt_version:
                target = arm
                break

        if target is None:
            target = BanditArm(role=role, model=model, prompt_version=prompt_version)
            role_arms.append(target)

        n = target.pulls
        target.pulls = n + 1
        target.avg_pass_rate = ((target.avg_pass_rate * n) + pass_rate) / (n + 1)
        target.avg_cost_usd = ((target.avg_cost_usd * n) + cost_usd) / (n + 1)
        target.avg_latency_seconds = ((target.avg_latency_seconds * n) + latency_seconds) / (n + 1)

        self.save()
        return target

    def add_prompt_version(
        self,
        role: str,
        prompt_version: str,
        prompt_template: str,
        models: Optional[List[str]] = None,
    ) -> List[BanditArm]:
        """Add new prompt version arms proposed by the Reviewer (TextGrad/DSPy style)."""
        role_arms = self.arms.setdefault(role, [])
        if models is None:
            existing_models = list({a.model for a in role_arms}) or ["gemini/gemini-1.5-flash"]
            models = existing_models

        created: List[BanditArm] = []
        for m in models:
            if not any(a.model == m and a.prompt_version == prompt_version for a in role_arms):
                new_arm = BanditArm(
                    role=role,
                    model=m,
                    prompt_version=prompt_version,
                    prompt_template=prompt_template,
                )
                role_arms.append(new_arm)
                created.append(new_arm)

        self.save()
        return created

    def save(self) -> None:
        """Persist score table to JSON."""
        self.score_table_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = {
            role: [arm.model_dump() for arm in arm_list]
            for role, arm_list in self.arms.items()
        }
        self.score_table_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
