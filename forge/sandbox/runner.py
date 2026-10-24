"""Docker sandbox runner for isolated code execution with timeout enforcement."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CommandResult(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_seconds: float = 0.0


class DockerSandboxRunner:
    """Executes generated code inside an isolated Docker container (or isolated temp workspace fallback)."""

    def __init__(
        self,
        workspace_dir: Optional[str | Path] = None,
        image: str = "python:3.11-slim",
        default_timeout: int = 60,
        use_docker: Optional[bool] = None,
    ):
        if workspace_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="forge_sandbox_")
            self.workspace_dir = Path(self._temp_dir)
            self._owns_workspace = True
        else:
            self.workspace_dir = Path(workspace_dir).resolve()
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            self._owns_workspace = False

        self.image = image
        self.default_timeout = default_timeout
        self.use_docker = use_docker if use_docker is not None else self._detect_docker()

    def _detect_docker(self) -> bool:
        """Check if Docker daemon is reachable."""
        if os.environ.get("FORGE_DISABLE_DOCKER") == "1":
            return False
        docker_bin = shutil.which("docker")
        if not docker_bin:
            return False
        try:
            proc = subprocess.run(
                [docker_bin, "info"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return proc.returncode == 0
        except Exception:
            return False

    def write_files(self, files: Dict[str, str]) -> List[str]:
        """Write a dictionary of relative_path -> content into the sandbox workspace."""
        written: List[str] = []
        for rel_path, content in files.items():
            # Prevent directory traversal outside workspace
            clean_rel = Path(rel_path)
            if clean_rel.is_absolute() or ".." in clean_rel.parts:
                raise ValueError(f"Unsafe file path rejected by sandbox: {rel_path}")
            dest = self.workspace_dir / clean_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            written.append(str(clean_rel).replace("\\", "/"))
        return written

    def read_file(self, rel_path: str) -> str:
        """Read a file from the sandbox workspace."""
        clean_rel = Path(rel_path)
        if clean_rel.is_absolute() or ".." in clean_rel.parts:
            raise ValueError(f"Unsafe file path rejected by sandbox: {rel_path}")
        target = self.workspace_dir / clean_rel
        return target.read_text(encoding="utf-8")

    def list_files(self) -> List[str]:
        """List all relative file paths inside the sandbox workspace."""
        result = []
        for p in self.workspace_dir.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                result.append(str(p.relative_to(self.workspace_dir)).replace("\\", "/"))
        return sorted(result)

    def exec_command(
        self,
        command: str,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        """Run a shell command inside the sandbox, capturing stdout/stderr/exit_code and enforcing timeout."""
        effective_timeout = timeout if timeout is not None else self.default_timeout
        start_time = time.time()

        if self.use_docker:
            cmd_args = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{self.workspace_dir}:/workspace",
                "-w",
                "/workspace",
            ]
            if env:
                for k, v in env.items():
                    cmd_args.extend(["-e", f"{k}={v}"])
            cmd_args.extend([self.image, "sh", "-c", command])
            exec_kwargs = {"shell": False}
        else:
            cmd_args = command
            run_env = os.environ.copy()
            if env:
                run_env.update(env)
            exec_kwargs = {
                "shell": True,
                "cwd": str(self.workspace_dir),
                "env": run_env,
            }

        try:
            proc = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                **exec_kwargs,
            )
            duration = time.time() - start_time
            return CommandResult(
                command=command,
                exit_code=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                timed_out=False,
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - start_time
            stdout_str = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr_str = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return CommandResult(
                command=command,
                exit_code=124,
                stdout=stdout_str,
                stderr=stderr_str + f"\nCommand timed out after {effective_timeout} seconds.",
                timed_out=True,
                duration_seconds=duration,
            )

    def cleanup(self) -> None:
        """Remove temporary sandbox directory if owned by this runner."""
        if self._owns_workspace and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
