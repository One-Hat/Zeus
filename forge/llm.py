"""LiteLLM wrapper with cost tracking, budget enforcement, and structured output parsing."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar
import yaml
from pydantic import BaseModel

# Prevent litellm from hanging on network calls for remote cost maps in offline/isolated environments
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
from litellm import completion, completion_cost

litellm.suppress_debug_info = True

from forge.schemas import LLMCallRecord

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BudgetExceededError(RuntimeError):
    """Raised when total run cost exceeds the configured max budget."""
    pass


class StepTimeoutError(TimeoutError):
    """Raised when an LLM call exceeds the allowed per-step timeout."""
    pass


class LLMClient:
    """Centralized LLM client supporting model routing, cost tracking, and Pydantic outputs."""

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        max_budget_usd: Optional[float] = None,
        step_timeout_seconds: Optional[int] = None,
    ):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "routing.yaml"
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.max_budget_usd = (
            max_budget_usd
            if max_budget_usd is not None
            else float(self.config.get("default_budget_usd", 2.0))
        )
        self.step_timeout_seconds = (
            step_timeout_seconds
            if step_timeout_seconds is not None
            else int(self.config.get("step_timeout_seconds", 120))
        )

        self.call_history: List[LLMCallRecord] = []
        self.total_cost_usd: float = 0.0
        self.mock_responses: Dict[str, Any] = {}

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {
                "default_budget_usd": 2.0,
                "step_timeout_seconds": 120,
                "roles": {},
            }
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_role_config(self, role: str) -> Dict[str, Any]:
        roles = self.config.get("roles", {})
        return roles.get(
            role,
            {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "prompt_version": "v1",
            },
        )

    def register_mock_response(self, role: str, response: BaseModel | Dict[str, Any] | str) -> None:
        """Register a mock response for deterministic testing."""
        self.mock_responses[role] = response

    def _clean_json_text(self, text: str) -> str:
        """Clean markdown formatting from JSON strings."""
        text = text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1).strip()
        return text

    def call_structured(
        self,
        role: str,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        override_model: Optional[str] = None,
        override_temperature: Optional[float] = None,
    ) -> T:
        """Execute a model call with strict structured output parsing and cost tracking."""
        if self.total_cost_usd >= self.max_budget_usd:
            raise BudgetExceededError(
                f"Budget exceeded: current cost ${self.total_cost_usd:.4f} >= limit ${self.max_budget_usd:.4f}"
            )

        role_cfg = self.get_role_config(role)
        model = override_model or role_cfg.get("model", "gpt-4o-mini")
        temperature = (
            override_temperature
            if override_temperature is not None
            else role_cfg.get("temperature", 0.2)
        )

        start_time = time.time()

        if role in self.mock_responses or os.environ.get("FORGE_MOCK_LLM") == "1":
            raw_mock = self.mock_responses.get(role)
            if raw_mock is None:
                try:
                    mock_obj = response_model()  # type: ignore
                except Exception:
                    raise ValueError(f"No mock response configured for role '{role}'")
            elif isinstance(raw_mock, response_model):
                mock_obj = raw_mock
            elif isinstance(raw_mock, dict):
                mock_obj = response_model.model_validate(raw_mock)
            elif isinstance(raw_mock, str):
                cleaned = self._clean_json_text(raw_mock)
                mock_obj = response_model.model_validate_json(cleaned)
            else:
                mock_obj = response_model.model_validate(raw_mock)

            latency = time.time() - start_time
            record = LLMCallRecord(
                role=role,
                model=f"mock-{model}",
                prompt_tokens=100,
                completion_tokens=150,
                cost_usd=0.0001,
                latency_seconds=latency,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self.call_history.append(record)
            self.total_cost_usd += record.cost_usd
            return mock_obj

        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        system_instruction = (
            f"\n\nYou MUST respond ONLY with a valid JSON object matching this schema:\n{schema_json}\n"
            "Do not include any text outside the JSON object."
        )

        formatted_messages = list(messages)
        if formatted_messages and formatted_messages[0].get("role") == "system":
            formatted_messages[0] = {
                "role": "system",
                "content": formatted_messages[0]["content"] + system_instruction,
            }
        else:
            formatted_messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You are a specialized autonomous software engineering agent." + system_instruction,
                },
            )

        try:
            response = completion(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                timeout=self.step_timeout_seconds,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            fallback = role_cfg.get("fallback_model")
            if fallback and fallback != model:
                logger.warning("Primary model %s failed (%s). Retrying with fallback %s", model, e, fallback)
                response = completion(
                    model=fallback,
                    messages=formatted_messages,
                    temperature=temperature,
                    timeout=self.step_timeout_seconds,
                    response_format={"type": "json_object"},
                )
                model = fallback
            else:
                raise e

        latency = time.time() - start_time

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        try:
            call_cost = float(completion_cost(completion_response=response))
        except Exception:
            call_cost = (prompt_tokens * 0.000001) + (completion_tokens * 0.000002)

        record = LLMCallRecord(
            role=role,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=call_cost,
            latency_seconds=latency,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.call_history.append(record)
        self.total_cost_usd += call_cost

        content = response.choices[0].message.content or ""
        cleaned_json = self._clean_json_text(content)

        try:
            return response_model.model_validate_json(cleaned_json)
        except Exception as parse_err:
            logger.error("Failed to parse structured output for %s: %s\nRaw content: %s", role, parse_err, content)
            raise ValueError(f"Agent '{role}' returned invalid structured output: {parse_err}") from parse_err

    def reset_tracking(self) -> None:
        """Reset cost tracking and call history for a new run."""
        self.call_history.clear()
        self.total_cost_usd = 0.0
