"""Integrator Agent: builds, lints, runs unit tests, and executes OpenAPI contract smoke tests in the sandbox."""

from __future__ import annotations

import json
import sys
import time
from typing import List, Optional
from forge.sandbox.runner import DockerSandboxRunner
from forge.schemas import Report, Spec, TestResult


SMOKE_TEST_SCRIPT_TEMPLATE = '''"""Auto-generated Forge contract smoke test runner."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)
tests = __ACCEPTANCE_TESTS_JSON__
results = []

for t in tests:
    method = t.get("method", "GET").upper()
    endpoint = t.get("endpoint", "/health")
    payload = t.get("payload")
    headers = t.get("headers") or {}
    expected_status = int(t.get("expected_status", 200))
    expected_contains = t.get("expected_body_contains")

    try:
        if method in ("POST", "PUT", "PATCH"):
            resp = client.request(method, endpoint, json=payload, headers=headers)
        else:
            resp = client.request(method, endpoint, headers=headers)

        status_ok = resp.status_code == expected_status
        body_text = resp.text
        body_ok = True
        if expected_contains:
            body_ok = expected_contains in body_text

        passed = status_ok and body_ok
        details = f"HTTP {resp.status_code} (expected {expected_status})"
        if not body_ok:
            details += f" | Missing expected substring '{expected_contains}' in '{body_text[:120]}'"
        results.append({
            "test_id": t["id"],
            "passed": passed,
            "status_code": resp.status_code,
            "details": details,
        })
    except Exception as exc:
        results.append({
            "test_id": t["id"],
            "passed": False,
            "status_code": None,
            "details": f"Exception: {exc}",
        })

print("FORGE_SMOKE_RESULTS_START")
print(json.dumps(results))
print("FORGE_SMOKE_RESULTS_END")
sys.exit(0 if all(r["passed"] for r in results) else 1)
'''


class IntegratorAgent:
    """Validates workspace files via syntax/lint check, unit tests, and OpenAPI contract smoke tests."""

    def __init__(self, sandbox: DockerSandboxRunner, python_executable: Optional[str] = None):
        self.sandbox = sandbox
        self.python_exe = python_executable or sys.executable

    def run(
        self,
        run_id: str,
        spec: Spec,
        fix_iterations: int = 0,
        total_cost_usd: float = 0.0,
    ) -> Report:
        start_time = time.time()
        logs: List[str] = []
        failing_files: List[str] = []

        # 1. Verify expected files exist (Build step)
        workspace_files = self.sandbox.list_files()
        py_files = [f for f in workspace_files if f.endswith(".py") and not f.startswith("_forge_")]
        build_passed = len(py_files) > 0
        logs.append(f"[BUILD] Workspace files ({len(workspace_files)}): {', '.join(workspace_files)}")
        if not build_passed:
            logs.append("[BUILD] ERROR: No Python files found in workspace.")

        # 2. Syntax & Compile Lint Check
        lint_passed = True
        if py_files:
            quoted_files = " ".join(f'"{f}"' for f in py_files)
            lint_cmd = f'"{self.python_exe}" -m py_compile {quoted_files}'
            lint_res = self.sandbox.exec_command(lint_cmd)
            lint_passed = lint_res.exit_code == 0
            logs.append(f"[LINT] Exit {lint_res.exit_code}\n{lint_res.stdout}\n{lint_res.stderr}".strip())
            if not lint_passed:
                for f in py_files:
                    if f in lint_res.stderr or f in lint_res.stdout:
                        failing_files.append(f)
        else:
            lint_passed = False

        # 3. Unit Tests (pytest)
        unit_tests_passed = False
        if build_passed and lint_passed:
            pytest_cmd = (
                f'"{self.python_exe}" -c "import sys; '
                "sys.path = [p for p in sys.path if not p.endswith('Python311')]; "
                'import pytest; sys.exit(pytest.main([\'-q\', \'-p\', \'no:cacheprovider\']))"'
            )
            unit_res = self.sandbox.exec_command(pytest_cmd)
            unit_tests_passed = unit_res.exit_code == 0
            logs.append(f"[UNIT TESTS] Exit {unit_res.exit_code}\n{unit_res.stdout}\n{unit_res.stderr}".strip())
            if not unit_tests_passed:
                for f in py_files:
                    if f in unit_res.stdout or f in unit_res.stderr:
                        if f not in failing_files:
                            failing_files.append(f)

        # 4. Contract & Acceptance Smoke Tests
        smoke_tests_passed = False
        acceptance_results: List[TestResult] = []
        if build_passed and lint_passed:
            tests_json = json.dumps([t.model_dump() for t in spec.acceptance_tests])
            smoke_script = SMOKE_TEST_SCRIPT_TEMPLATE.replace("__ACCEPTANCE_TESTS_JSON__", tests_json)
            self.sandbox.write_files({"_forge_smoke_runner.py": smoke_script})

            smoke_cmd = f'"{self.python_exe}" _forge_smoke_runner.py'
            smoke_res = self.sandbox.exec_command(smoke_cmd)
            logs.append(f"[SMOKE TESTS] Exit {smoke_res.exit_code}\n{smoke_res.stdout}\n{smoke_res.stderr}".strip())

            if "FORGE_SMOKE_RESULTS_START" in smoke_res.stdout:
                try:
                    raw_part = smoke_res.stdout.split("FORGE_SMOKE_RESULTS_START")[1].split("FORGE_SMOKE_RESULTS_END")[0].strip()
                    parsed_list = json.loads(raw_part)
                    acceptance_results = [TestResult.model_validate(item) for item in parsed_list]
                    smoke_tests_passed = all(r.passed for r in acceptance_results) and smoke_res.exit_code == 0
                except Exception as exc:
                    logs.append(f"[SMOKE TESTS] Failed to parse smoke output: {exc}")
                    smoke_tests_passed = False
            else:
                smoke_tests_passed = False

            if not smoke_tests_passed and "backend/app.py" not in failing_files:
                failing_files.append("backend/app.py")

        overall_success = build_passed and lint_passed and unit_tests_passed and smoke_tests_passed
        error_summary = None
        if not overall_success:
            error_summary = (
                f"Build={build_passed}, Lint={lint_passed}, "
                f"UnitTests={unit_tests_passed}, SmokeTests={smoke_tests_passed}"
            )

        wall_time = time.time() - start_time
        return Report(
            run_id=run_id,
            success=overall_success,
            build_passed=build_passed,
            lint_passed=lint_passed,
            unit_tests_passed=unit_tests_passed,
            smoke_tests_passed=smoke_tests_passed,
            acceptance_results=acceptance_results,
            logs="\n\n".join(logs),
            failing_files=failing_files,
            error_summary=error_summary,
            wall_time_seconds=round(wall_time, 3),
            total_cost_usd=round(total_cost_usd, 6),
            fix_iterations=fix_iterations,
        )
