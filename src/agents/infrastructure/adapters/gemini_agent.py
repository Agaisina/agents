import logging
from typing import Any, List, Optional

from google.api_core.exceptions import GoogleAPIError
from google.genai.errors import ServerError as GenAIServerError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from src.agents.infrastructure.adapters.base_agent import BaseAgent, AgentExecutionError, generic_token_counter

logger = logging.getLogger(__name__)


class GeminiAgent(BaseAgent):
    """Agent implementation backed by Google Gemini."""

    def _create_llm(self, token: str) -> ChatGoogleGenerativeAI:
        """Instantiate ``ChatGoogleGenerativeAI`` from the provided API key."""
        try:
            return ChatGoogleGenerativeAI(
                model=self.config.model_id,
                google_api_key=token,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.extra_params,
            )
        except Exception as error:
            logger.critical(
                "Failed to connect to Google Gemini for model %s: %s",
                self.config.model_id,
                error,
            )
            raise AgentExecutionError("LLM Initialization Error") from error

    @property
    def _retry_exceptions(self):
        """Exceptions that trigger automatic retry for Gemini API calls.

        ``ChatGoogleGenerativeAIError`` is the LangChain wrapper that captures
        429 RESOURCE_EXHAUSTED (rate-limit) and 5xx errors returned by the
        Gemini API, so it must be included here to trigger the retry loop.
        """
        return (ConnectionError, TimeoutError, GoogleAPIError, ChatGoogleGenerativeAIError, GenAIServerError)

    @property
    def _token_counter(self):
        return generic_token_counter
