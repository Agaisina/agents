import logging
from typing import Dict, List, Optional, Any

from src.agents.domain.interfaces import IAgent
from src.agents.application.factory import create_agent
from src.agents.application.utils import load_agent_prompt
from src.agents.application.agent_definitions import (
    AgentDefinition,
    AGENT_DEFINITIONS,
    TOOL_REGISTRY,
)

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Central registry for the multi-agent system.

    Agents are declared via AgentDefinition objects in agent_definitions.py.
    Agents built without custom extra-tools are cached after first build.
    """

    def __init__(self, api_key: str, model_id: str, base_url: str = None, provider: str = None):
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = base_url
        self.provider = provider  # resolved by factory; None → env-var fallback

        if not self.api_key:
            raise ValueError("AgentRegistry requires a valid API key.")
        if not self.model_id:
            raise ValueError("AgentRegistry requires a model_id.")

        self._definitions: Dict[str, AgentDefinition] = dict(AGENT_DEFINITIONS)
        self._cache: Dict[str, IAgent] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, role_name: str, definition: AgentDefinition) -> None:
        """Register or replace an agent definition at runtime.

        Clears the cache for ``role_name`` so the next call to
        :meth:`get_agent` will build a fresh instance.
        """
        role_name = role_name.lower()
        self._definitions[role_name] = definition
        self._cache.pop(role_name, None)

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    def _build_from_definition(
        self,
        role_name: str,
        definition: AgentDefinition,
        extra_tools: Optional[List[Any]] = None,
    ) -> IAgent:
        """Construct a concrete agent from a definition.

        Loads the YAML prompt, resolves tool instances, and delegates
        instantiation to :func:`create_agent`.

        Args:
            role_name: Human-readable role key used for logging.
            definition: The ``AgentDefinition`` record with prompt key and tool keys.
            extra_tools: Additional runtime tools to prepend before the defaults.

        Returns:
            A fully configured :class:`~src.agents.domain.interfaces.IAgent` instance.
        """
        try:
            config = load_agent_prompt(definition.prompt_key)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Prompt file for agent '{role_name}' not found "
                f"(prompt_key='{definition.prompt_key}')."
            ) from exc

        agent_data = config.get("agent")
        if agent_data is None:
            raise KeyError(
                f"YAML for prompt_key='{definition.prompt_key}' is missing the required 'agent' key."
            )

        default_tools = [TOOL_REGISTRY[key] for key in definition.default_tool_keys]
        all_tools = (extra_tools or []) + default_tools

        logger.info("Building agent '%s' (role: %s).", agent_data["name"], role_name)
        return create_agent(
            name=agent_data["name"],
            role_description=agent_data["prompt"],
            token=self.api_key,
            model_id=self.model_id,
            base_url=self.base_url,
            temperature=agent_data["temperature"],
            tools=all_tools if all_tools else None,
            provider=self.provider,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_agent(self, role_name: str, tools: Optional[List[Any]] = None) -> IAgent:
        """Return an agent for ``role_name``, building it on first call.

        Agents requested without ``tools`` are cached. Agents requested with
        extra ``tools`` are always freshly constructed and never cached.

        Args:
            role_name: A registered role key, e.g. ``"planner"``.
            tools: Optional extra tools to attach on top of the role's defaults.

        Returns:
            A :class:`~src.agents.domain.interfaces.IAgent` instance.

        Raises:
            ValueError: When ``role_name`` is not found in the registry.
        """
        role_name = role_name.lower()
        if role_name not in self._definitions:
            available = list(self._definitions)
            raise ValueError(
                f"Agent role '{role_name}' is not registered. Available: {available}"
            )

        definition = self._definitions[role_name]

        if tools is None:
            if role_name not in self._cache:
                logger.info("Building and caching agent '%s'.", role_name)
                self._cache[role_name] = self._build_from_definition(role_name, definition)
            else:
                logger.debug("Returning cached agent '%s'.", role_name)
            return self._cache[role_name]

        logger.info("Building agent '%s' with extra tools (not cached).", role_name)
        return self._build_from_definition(role_name, definition, extra_tools=tools)