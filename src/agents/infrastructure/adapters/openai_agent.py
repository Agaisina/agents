import logging
from typing import Any, List, Optional

from langchain_openai import ChatOpenAI
from openai import APIError

from src.agents.infrastructure.adapters.base_agent import BaseAgent, AgentConfig, AgentExecutionError, generic_token_counter

logger = logging.getLogger(__name__)


class OpenAIAgent(BaseAgent):
    """Agent implementation backed by OpenAI."""

    def _create_llm(self, token: str) -> ChatOpenAI:
        """Instantiate ``ChatOpenAI``, with support for local LLMs via ``base_url``."""
        try:
            llm_kwargs = {
                "model": self.config.model_id,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                **self.config.extra_params,
            }
            if self.config.base_url:
                llm_kwargs["base_url"] = self.config.base_url
                # Local LLMs (e.g. Ollama) do not require a real API key;
                # use the provided token or an explicit placeholder so the
                # intent is clear and credential validation is not silently bypassed.
                if not token:
                    logger.warning(
                        "No token supplied for local LLM at %s; using placeholder key.",
                        self.config.base_url,
                    )
                llm_kwargs["api_key"] = token if token else "not-required"
            else:
                llm_kwargs["api_key"] = token
            return ChatOpenAI(**llm_kwargs)
        except Exception as error:
            logger.critical(
                "Failed to connect to OpenAI/Local LLM for model %s: %s",
                self.config.model_id,
                error,
            )
            raise AgentExecutionError("LLM Initialization Error") from error

    @property
    def _retry_exceptions(self):
        """Exceptions that trigger automatic retry for OpenAI API calls."""
        return (ConnectionError, TimeoutError, APIError)

    @property
    def _token_counter(self):
        return generic_token_counter
