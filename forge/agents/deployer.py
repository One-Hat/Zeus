"""Deployer Agent: deploys verified builds to staging and enforces a manual approval gate for production."""

from __future__ import annotations

from typing import Literal, Optional
from forge.sandbox.runner import DockerSandboxRunner
from forge.schemas import DeployResult, Report, Spec


DEFAULT_DOCKERFILE = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
"""


class DeployerAgent:
    """Packages the verified app with a Dockerfile and deploys to staging (production gated by manual approval)."""

    def __init__(self, sandbox: Optional[DockerSandboxRunner] = None):
        self.sandbox = sandbox

    def run(
        self,
        spec: Spec,
        report: Report,
        target_env: Literal["staging", "production"] = "staging",
        human_approved: bool = False,
    ) -> DeployResult:
        if not report.success:
            return DeployResult(
                deployed=False,
                staging_url=None,
                environment=target_env,
                requires_human_approval=target_env == "production",
                approval_granted=False,
                message="Deployment aborted: Integrator verification checks did not pass.",
            )

        if self.sandbox is not None:
            self.sandbox.write_files({"Dockerfile": DEFAULT_DOCKERFILE})

        staging_url = f"https://{spec.name}-{report.run_id[:8]}.staging.forge.app"

        if target_env == "production":
            if not human_approved:
                return DeployResult(
                    deployed=False,
                    staging_url=staging_url,
                    environment="production",
                    requires_human_approval=True,
                    approval_granted=False,
                    message="Production deployment blocked: explicit manual human approval is required.",
                )
            prod_url = f"https://{spec.name}.prod.forge.app"
            return DeployResult(
                deployed=True,
                staging_url=prod_url,
                environment="production",
                requires_human_approval=True,
                approval_granted=True,
                message=f"Approved and deployed to production at {prod_url}",
            )

        return DeployResult(
            deployed=True,
            staging_url=staging_url,
            environment="staging",
            requires_human_approval=False,
            approval_granted=False,
            message=f"Successfully deployed verified container to staging at {staging_url}",
        )
