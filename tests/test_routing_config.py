"""Tests for config/routing.yaml."""

from pathlib import Path
import yaml


def test_routing_config_file_exists():
    config_path = Path(__file__).parent.parent / "forge" / "config" / "routing.yaml"
    assert config_path.exists(), "config/routing.yaml must exist"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "default_budget_usd" in config
    assert "max_fix_loops" in config
    assert config["max_fix_loops"] == 3
    assert "step_timeout_seconds" in config
    assert "roles" in config

    required_roles = [
        "planner",
        "frontend",
        "backend",
        "ml",
        "integrator",
        "fixer",
        "reviewer",
        "deployer",
    ]
    for role in required_roles:
        assert role in config["roles"], f"Role '{role}' must be defined in routing.yaml"
        role_data = config["roles"][role]
        assert "model" in role_data, f"Role '{role}' must specify a model"
        assert "prompt_version" in role_data, f"Role '{role}' must specify a prompt_version"
