import os
import importlib
import logging
from typing import List, Optional, Any
from src.agents.domain.interfaces import IAgent
from src.agents.domain.models import AgentConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
_PROVIDER_REGISTRY: dict[str, tuple[str, str, str]] = {
    "gemini": (
        "src.agents.infrastructure.adapters.gemini_agent",
        "GeminiAgent",
        "GEMINI",
    ),
    "openai": (
        "src.agents.infrastructure.adapters.openai_agent",
        "OpenAIAgent",
        "OPENAI",
    ),
}


def create_agent(
    name: str,
    role_description: str,
    token: str,
    model_id: str,
    temperature: float = 0.1,
    base_url: str = None,
    tools: Optional[List[Any]] = None,
    provider: Optional[str] = None,
) -> IAgent:
    """
    Factory method to create an agent instance based on the LLM provider.

    The provider is resolved in order:
      1. ``provider`` argument (set by AgentRegistry from settings.yaml).
      2. ``LLM_PROVIDER`` environment variable (legacy / override).
      3. Falls back to ``openai``.

    To add a new provider, register it in _PROVIDER_REGISTRY above.
    """
    resolved = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()

    if resolved not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown LLM provider: '{resolved}'. "
            f"Valid options: {list(_PROVIDER_REGISTRY)}"
        )

    module_path, class_name, display_label = _PROVIDER_REGISTRY[resolved]
    logger.info("⚙️ Building [%s] using %s", name, display_label)

    agent_class = getattr(importlib.import_module(module_path), class_name)

    return agent_class(
        name=name,
        role_description=role_description,
        token=token,
        config=AgentConfig(model_id=model_id, temperature=temperature, base_url=base_url),
        tools=tools,
    )