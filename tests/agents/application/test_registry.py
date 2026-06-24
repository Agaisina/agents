import pytest
from unittest.mock import MagicMock, patch
from src.agents.application.registry import AgentRegistry
from src.agents.application.agent_definitions import AgentDefinition, AGENT_DEFINITIONS, TOOL_REGISTRY


def test_registry_seeded_from_agent_definitions(mock_credentials):
    registry = AgentRegistry(**mock_credentials)
    for role_name in AGENT_DEFINITIONS:
        assert role_name in registry._definitions


def test_all_default_roles_present(mock_credentials):
    registry = AgentRegistry(**mock_credentials)
    for role in ("corporate_researcher", "planner", "operator", "writer"):
        assert role in registry._definitions


def test_get_agent_raises_error_if_not_found(mock_credentials):
    registry = AgentRegistry(**mock_credentials)
    with pytest.raises(ValueError, match="is not registered"):
        registry.get_agent("non_existent_role")


def test_get_agent_error_lists_available_roles(mock_credentials):
    registry = AgentRegistry(**mock_credentials)
    with pytest.raises(ValueError) as exc_info:
        registry.get_agent("ghost")
    assert "corporate_researcher" in str(exc_info.value)


def test_missing_api_key_raises(mock_credentials):
    with pytest.raises(ValueError, match="API key"):
        AgentRegistry(api_key="", model_id="some-model")


def test_missing_model_id_raises(mock_credentials):
    with pytest.raises(ValueError, match="model_id"):
        AgentRegistry(api_key="some-key", model_id="")


def test_register_custom_definition(mock_credentials, mock_agent):
    registry = AgentRegistry(**mock_credentials)
    custom_def = AgentDefinition(prompt_key="planner", default_tool_keys=[])

    with patch.object(registry, "_build_from_definition", return_value=mock_agent):
        registry.register("custom_role", custom_def)
        result = registry.get_agent("custom_role")

    assert result is mock_agent
    assert "custom_role" in registry._definitions


def test_get_agent_caches_instance(mock_credentials, mock_agent):
    registry = AgentRegistry(**mock_credentials)
    with patch.object(registry, "_build_from_definition", return_value=mock_agent) as mock_build:
        first = registry.get_agent("writer")
        second = registry.get_agent("writer")

    assert first is second
    assert mock_build.call_count == 1


def test_get_agent_with_extra_tools_bypasses_cache(mock_credentials, mock_agent, mock_tools):
    registry = AgentRegistry(**mock_credentials)
    with patch.object(registry, "_build_from_definition", return_value=mock_agent) as mock_build:
        registry.get_agent("planner", tools=mock_tools)
        registry.get_agent("planner", tools=mock_tools)

    assert mock_build.call_count == 2


def test_register_overrides_invalidate_cache(mock_credentials, mock_agent):
    registry = AgentRegistry(**mock_credentials)
    def_v1 = AgentDefinition(prompt_key="planner", default_tool_keys=[])
    def_v2 = AgentDefinition(prompt_key="writer", default_tool_keys=[])

    mock_v1 = MagicMock()
    mock_v2 = MagicMock()

    with patch.object(registry, "_build_from_definition", side_effect=[mock_v1, mock_v2]):
        registry.register("my_role", def_v1)
        registry.get_agent("my_role")         # populates cache with mock_v1

        registry.register("my_role", def_v2)  # should clear cache
        result = registry.get_agent("my_role")

    assert result is mock_v2


def test_role_lookup_is_case_insensitive(mock_credentials, mock_agent):
    registry = AgentRegistry(**mock_credentials)
    with patch.object(registry, "_build_from_definition", return_value=mock_agent):
        result = registry.get_agent("WRITER")
    assert result is mock_agent


def test_agent_definitions_planner_has_geo_tools():
    planner_def = AGENT_DEFINITIONS["planner"]
    assert "get_travel_distance" in planner_def.default_tool_keys
    assert "get_transit_vs_taxi_time" in planner_def.default_tool_keys
    assert "find_optimal_transport" in planner_def.default_tool_keys


def test_agent_definitions_operator_has_calculator_tools():
    operator_def = AGENT_DEFINITIONS["operator"]
    assert "safe_calculator" in operator_def.default_tool_keys
    assert "convert_currency" in operator_def.default_tool_keys


def test_tool_registry_contains_all_declared_tools():
    all_tool_keys = {
        key
        for defn in AGENT_DEFINITIONS.values()
        for key in defn.default_tool_keys
    }
    for key in all_tool_keys:
        assert key in TOOL_REGISTRY, f"Tool '{key}' declared in AGENT_DEFINITIONS but missing from TOOL_REGISTRY"