"""Pydantic schemas for the Forge multi-LLM pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class RoleType(str, Enum):
    PLANNER = "planner"
    FRONTEND = "frontend"
    BACKEND = "backend"
    ML = "ml"
    INTEGRATOR = "integrator"
    FIXER = "fixer"
    REVIEWER = "reviewer"
    DEPLOYER = "deployer"


class Task(BaseModel):
    id: str = Field(..., description="Unique task identifier, e.g. 'be-1', 'fe-1'")
    role: Literal["frontend", "backend", "ml"] = Field(..., description="Worker role responsible")
    title: str = Field(..., description="Short summary of the task")
    description: str = Field(..., description="Detailed instructions for the worker")
    target_files: List[str] = Field(default_factory=list, description="Target filepaths to produce or edit")
    dependencies: List[str] = Field(default_factory=list, description="IDs of tasks that must precede this")


class AcceptanceTest(BaseModel):
    id: str = Field(..., description="Test identifier, e.g. 'test-create-todo'")
    endpoint: str = Field(..., description="Path, e.g. '/api/todos'")
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET"
    headers: Dict[str, str] = Field(default_factory=dict)
    payload: Optional[Dict[str, Any]] = None
    expected_status: int = 200
    expected_body_contains: Optional[str] = None
    description: str = Field(default="", description="What this test verifies")


class Spec(BaseModel):
    idea: str = Field(..., description="Original one-sentence input prompt")
    name: str = Field(..., description="App name in kebab-case or snake_case")
    description: str = Field(..., description="Architecture and functional overview")
    api_contract: Dict[str, Any] = Field(
        ...,
        description="OpenAPI 3.x document describing all endpoints, models, parameters"
    )
    db_schema: str = Field(..., description="SQL DDL statements or schema definition")
    file_tree: List[str] = Field(..., description="List of all relative file paths to be created")
    tasks: List[Task] = Field(..., description="List of tasks split into frontend / backend / ml")
    acceptance_tests: List[AcceptanceTest] = Field(..., description="End-to-end acceptance tests")


class WorkerOutput(BaseModel):
    role: Literal["frontend", "backend", "ml"]
    files: Dict[str, str] = Field(
        ...,
        description="Mapping of relative file path to full file content"
    )
    summary: str = Field(..., description="Summary of work executed")


class TestResult(BaseModel):
    test_id: str
    passed: bool
    status_code: Optional[int] = None
    details: str = ""


class Report(BaseModel):
    run_id: str
    success: bool
    build_passed: bool
    lint_passed: bool
    unit_tests_passed: bool
    smoke_tests_passed: bool
    acceptance_results: List[TestResult] = Field(default_factory=list)
    logs: str = ""
    failing_files: List[str] = Field(default_factory=list)
    error_summary: Optional[str] = None
    wall_time_seconds: float = 0.0
    total_cost_usd: float = 0.0
    fix_iterations: int = 0


class FixerOutput(BaseModel):
    explanation: str = Field(..., description="Root cause diagnosis and fix explanation")
    patched_files: Dict[str, str] = Field(
        ...,
        description="Mapping of relative file path to full patched file content"
    )


class Score(BaseModel):
    run_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score 0.0 to 1.0")
    pass_rate: float = Field(..., ge=0.0, le=1.0, description="Acceptance test pass rate")
    critique: str = Field(..., description="Detailed reviewer evaluation")
    failed_decisions: List[str] = Field(default_factory=list, description="Decisions or errors that degraded the run")
    proposed_prompt_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Suggested prompt improvements per role (role -> prompt suggestion)"
    )


class DeployResult(BaseModel):
    deployed: bool
    staging_url: Optional[str] = None
    environment: Literal["staging", "production"] = "staging"
    requires_human_approval: bool = False
    approval_granted: bool = False
    message: str = ""


class LLMCallRecord(BaseModel):
    role: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    timestamp: str = ""
