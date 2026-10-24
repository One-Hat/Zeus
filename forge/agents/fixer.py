"""Fixer Agent: diagnoses failing Integrator logs and patches sandbox files."""

from __future__ import annotations

import json
from typing import Dict, Optional
from forge.agents.backend import DEFAULT_BACKEND_APP_PY, DEFAULT_BACKEND_TEST_PY
from forge.llm import LLMClient
from forge.sandbox.runner import DockerSandboxRunner
from forge.schemas import FixerOutput, Report, Spec

FIXER_SYSTEM_PROMPT = """You are the Forge Fixer Agent.
You receive the `Spec`, the failing `Report` logs from the Integrator, and the current contents of the failing files.
Analyze the root cause and return a `FixerOutput` containing `explanation` and `patched_files` mapping relative paths to the full repaired file contents.
"""


class FixerAgent:
    """Patches broken files in the sandbox based on Integrator error logs."""

    def __init__(self, llm_client: LLMClient, sandbox: DockerSandboxRunner):
        self.llm = llm_client
        self.sandbox = sandbox

    @staticmethod
    def build_default_output() -> FixerOutput:
        return FixerOutput(
            explanation="Restored clean FastAPI implementation and unit tests to resolve failing checks.",
            patched_files={
                "backend/app.py": DEFAULT_BACKEND_APP_PY,
                "backend/test_app.py": DEFAULT_BACKEND_TEST_PY,
            },
        )

    def run(self, spec: Spec, report: Report, override_model: Optional[str] = None) -> FixerOutput:
        self.llm.mock_responses.setdefault("fixer", self.build_default_output())

        file_contents: Dict[str, str] = {}
        targets = report.failing_files or ["backend/app.py", "backend/test_app.py"]
        for rel_path in targets:
            try:
                file_contents[rel_path] = self.sandbox.read_file(rel_path)
            except Exception:
                file_contents[rel_path] = "<file missing>"

        payload = {
            "app_name": spec.name,
            "error_summary": report.error_summary,
            "logs": report.logs[-4000:],
            "failing_files": file_contents,
            "acceptance_tests": [t.model_dump() for t in spec.acceptance_tests],
        }

        messages = [
            {"role": "system", "content": FIXER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]
        output = self.llm.call_structured(
            role="fixer",
            messages=messages,
            response_model=FixerOutput,
            override_model=override_model,
        )

        if output.patched_files:
            self.sandbox.write_files(output.patched_files)

        return output
