# Agent Interfaces

The abstract base that all concrete agent implementations must satisfy.
The workflow layer depends **only** on `IAgent` — it never imports provider-specific classes.

## IAgent

::: src.agents.domain.interfaces.IAgent

## BaseAgent

::: src.agents.infrastructure.adapters.base_agent.BaseAgent

## Provider Implementations

### GeminiAgent

::: src.agents.infrastructure.adapters.gemini_agent.GeminiAgent

### OpenAIAgent

::: src.agents.infrastructure.adapters.openai_agent.OpenAIAgent
