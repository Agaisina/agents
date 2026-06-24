import pytest
from unittest.mock import patch, mock_open
from src.agents.application.utils import load_agent_prompt


def test_load_planner_yaml():
    config = load_agent_prompt("planner")
    assert "agent" in config
    assert config["agent"]["name"] == "Corporate Planner"


def test_load_operator_yaml():
    config = load_agent_prompt("operator")
    assert "agent" in config
    assert "temperature" in config["agent"]


def test_load_nonexistent_key_raises_file_not_found():
    with pytest.raises(FileNotFoundError, match="nonexistent_agent"):
        load_agent_prompt("nonexistent_agent")


def test_load_agent_prompt_result_is_cached():
    """lru_cache ensures the YAML is only parsed once per key."""
    first = load_agent_prompt("writer")
    second = load_agent_prompt("writer")
    assert first is second  # Same dict object from cache


def test_load_agent_prompt_returns_dict():
    result = load_agent_prompt("corporate_researcher")
    assert isinstance(result, dict)