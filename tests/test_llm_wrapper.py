"""Tests for LiteLLM wrapper with cost tracking and budget caps."""

import pytest
from forge.llm import BudgetExceededError, LLMClient
from forge.schemas import Spec, Task, AcceptanceTest


def test_llm_client_initialization():
    client = LLMClient()
    assert client.total_cost_usd == 0.0
    assert client.max_budget_usd > 0
    assert client.step_timeout_seconds > 0
    role_cfg = client.get_role_config("planner")
    assert "model" in role_cfg


def test_llm_client_mock_structured_call():
    client = LLMClient(max_budget_usd=1.0)
    mock_spec = Spec(
        idea="A simple notes app",
        name="notes-app",
        description="Simple notes app",
        api_contract={"openapi": "3.0.0"},
        db_schema="CREATE TABLE notes (id INT, text TEXT);",
        file_tree=["main.py"],
        tasks=[Task(id="1", role="backend", title="Write API", description="API", target_files=["main.py"])],
        acceptance_tests=[AcceptanceTest(id="t1", endpoint="/notes", method="GET")],
    )

    client.register_mock_response("planner", mock_spec)
    result = client.call_structured(
        role="planner",
        messages=[{"role": "user", "content": "Create notes app"}],
        response_model=Spec,
    )

    assert result.name == "notes-app"
    assert len(client.call_history) == 1
    assert client.total_cost_usd > 0
    assert client.call_history[0].role == "planner"


def test_llm_budget_cap_enforcement():
    # Set an extremely low budget to test hard cap
    client = LLMClient(max_budget_usd=0.00001)
    client.total_cost_usd = 0.00002  # Artificially simulate spent budget

    with pytest.raises(BudgetExceededError) as exc_info:
        client.call_structured(
            role="planner",
            messages=[{"role": "user", "content": "test"}],
            response_model=Spec,
        )
    assert "Budget exceeded" in str(exc_info.value)
