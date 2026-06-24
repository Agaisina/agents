"""
Adding a new agent to the system requires only:
  1. Adding its tools to TOOL_REGISTRY (if new tools are needed).
  2. Adding an AgentDefinition entry to AGENT_DEFINITIONS.
  3. Placing the corresponding YAML prompt in src/agents/application/prompts/.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentDefinition:
    """Declarative specification for a single agent role.

    Attributes:
        prompt_key: Stem of the YAML file in ``prompts/`` that holds the
                    agent's name, role description, and temperature.
        default_tool_keys: Keys into :data:`TOOL_REGISTRY` for tools that
                           are always attached to this agent.
    """
    prompt_key: str
    default_tool_keys: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------
def _build_tool_registry() -> Dict[str, Any]:
    """Lazily import and return the canonical tool-name → callable mapping.

    Deferred imports avoid circular-import issues at module load time.
    """
    from src.agents.infrastructure.tools.operator_tools import safe_calculator, convert_currency
    from src.agents.infrastructure.tools.geo_tools import (
        get_travel_distance,
        get_transit_vs_taxi_time,
        find_optimal_transport,
    )

    return {
        "safe_calculator": safe_calculator,
        "convert_currency": convert_currency,
        "get_travel_distance": get_travel_distance,
        "get_transit_vs_taxi_time": get_transit_vs_taxi_time,
        "find_optimal_transport": find_optimal_transport,
    }


TOOL_REGISTRY: Dict[str, Any] = _build_tool_registry()


# ---------------------------------------------------------------------------
# Agent Definitions
# ---------------------------------------------------------------------------
AGENT_DEFINITIONS: Dict[str, AgentDefinition] = {
    "corporate_researcher": AgentDefinition(
        prompt_key="corporate_researcher",
        default_tool_keys=[],
    ),
    "planner": AgentDefinition(
        prompt_key="planner",
        default_tool_keys=["get_travel_distance", "get_transit_vs_taxi_time", "find_optimal_transport"],
    ),
    "operator": AgentDefinition(
        prompt_key="operator",
        default_tool_keys=["safe_calculator", "convert_currency"],
    ),
    "writer": AgentDefinition(
        prompt_key="writer",
        default_tool_keys=[],
    ),
}
