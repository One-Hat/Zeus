"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError
from forge.schemas import (
    AcceptanceTest,
    DeployResult,
    FixerOutput,
    Report,
    Score,
    Spec,
    Task,
    TestResult,
    WorkerOutput,
)


def test_task_schema():
    task = Task(
        id="be-1",
        role="backend",
        title="Setup database models",
        description="Create SQLite models using SQLAlchemy",
        target_files=["backend/models.py"],
    )
    assert task.id == "be-1"
    assert task.role == "backend"
    assert task.target_files == ["backend/models.py"]
    assert task.dependencies == []


def test_acceptance_test_schema():
    test = AcceptanceTest(
        id="test-1",
        endpoint="/api/todos",
        method="POST",
        payload={"title": "Buy milk"},
        expected_status=201,
        expected_body_contains="Buy milk",
    )
    assert test.endpoint == "/api/todos"
    assert test.expected_status == 201


def test_spec_schema():
    spec = Spec(
        idea="A todo app with auth",
        name="todo-app",
        description="A full-stack todo application with JWT auth",
        api_contract={
            "openapi": "3.0.0",
            "info": {"title": "Todo API", "version": "1.0.0"},
            "paths": {},
        },
        db_schema="CREATE TABLE todos (id INTEGER PRIMARY KEY, title TEXT);",
        file_tree=["backend/main.py", "frontend/index.html"],
        tasks=[
            Task(
                id="be-1",
                role="backend",
                title="Create API",
                description="FastAPI CRUD",
                target_files=["backend/main.py"],
            )
        ],
        acceptance_tests=[
            AcceptanceTest(
                id="t1",
                endpoint="/api/todos",
                method="GET",
                expected_status=200,
            )
        ],
    )
    assert spec.name == "todo-app"
    assert len(spec.tasks) == 1
    assert len(spec.acceptance_tests) == 1


def test_score_validation():
    # Valid score
    score = Score(
        run_id="run-123",
        score=0.85,
        pass_rate=1.0,
        critique="Good code quality and passing tests",
    )
    assert score.score == 0.85

    # Invalid score > 1.0
    with pytest.raises(ValidationError):
        Score(
            run_id="run-123",
            score=1.5,
            pass_rate=1.0,
            critique="Invalid score",
        )


def test_report_and_deploy_result():
    report = Report(
        run_id="run-1",
        success=True,
        build_passed=True,
        lint_passed=True,
        unit_tests_passed=True,
        smoke_tests_passed=True,
        acceptance_results=[TestResult(test_id="t1", passed=True, status_code=200)],
        logs="All checks passed.",
        wall_time_seconds=12.5,
        total_cost_usd=0.015,
        fix_iterations=0,
    )
    assert report.success is True
    assert report.acceptance_results[0].passed is True

    deploy = DeployResult(
        deployed=True,
        staging_url="https://forge-staging-run-1.internal",
        environment="staging",
        requires_human_approval=False,
    )
    assert deploy.staging_url is not None
    assert deploy.environment == "staging"
