import os
import pytest
from unittest.mock import patch
from src.agents.application.factory import create_agent


def test_create_agent_openai_logic(mock_credentials):
    with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
        with patch("src.agents.infrastructure.adapters.openai_agent.OpenAIAgent") as MockOpenAI:
            create_agent(
                name="Test",
                role_description="Desc",
                token="key",
                model_id="gpt-4",
            )
            MockOpenAI.assert_called_once()


def test_create_agent_gemini_logic(mock_credentials):
    with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}):
        with patch("src.agents.infrastructure.adapters.gemini_agent.GeminiAgent") as MockGemini:
            create_agent(
                name="Test",
                role_description="Desc",
                token="key",
                model_id="gemini-1",
            )
            MockGemini.assert_called_once()


def test_create_agent_unsupported_provider_raises():
    with patch.dict(os.environ, {"LLM_PROVIDER": "unsupported"}):
        with pytest.raises(ValueError, match="Unknown LLM provider: 'unsupported'"):
            create_agent(name="T", role_description="D", token="k", model_id="m")


def test_create_agent_unknown_provider_lists_valid_options():
    with patch.dict(os.environ, {"LLM_PROVIDER": "azure"}):
        with pytest.raises(ValueError) as exc_info:
            create_agent(name="T", role_description="D", token="k", model_id="m")
        assert "gemini" in str(exc_info.value)
        assert "openai" in str(exc_info.value)


def test_create_agent_passes_tools_to_adapter(mock_tools):
    with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
        with patch("src.agents.infrastructure.adapters.openai_agent.OpenAIAgent") as MockOpenAI:
            create_agent(
                name="T",
                role_description="D",
                token="k",
                model_id="m",
                tools=mock_tools,
            )
            _, kwargs = MockOpenAI.call_args
            assert kwargs["tools"] == mock_tools


def test_create_agent_provider_case_insensitive():
    """LLM_PROVIDER value should be normalised to lowercase."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "OPENAI"}):
        with patch("src.agents.infrastructure.adapters.openai_agent.OpenAIAgent") as MockOpenAI:
            create_agent(name="T", role_description="D", token="k", model_id="m")
            MockOpenAI.assert_called_once()
