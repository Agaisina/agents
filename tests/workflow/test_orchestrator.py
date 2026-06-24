import pytest
from src.workflow.service import AgentOrchestrator


# ─────────────────────────────────────────
#   _normalize_base_url
# ─────────────────────────────────────────

def test_normalize_base_url_returns_none_for_empty_string():
    assert AgentOrchestrator._normalize_base_url("") is None


def test_normalize_base_url_returns_none_for_whitespace():
    result = AgentOrchestrator._normalize_base_url("   ")
    assert result is None or result.strip() == ""


def test_normalize_base_url_returns_none_for_none_value():
    assert AgentOrchestrator._normalize_base_url(None) is None


def test_normalize_base_url_returns_none_for_string_none():
    assert AgentOrchestrator._normalize_base_url("none") is None


def test_normalize_base_url_returns_none_for_string_none_mixed_case():
    assert AgentOrchestrator._normalize_base_url("NONE") is None
    assert AgentOrchestrator._normalize_base_url("None") is None


def test_normalize_base_url_returns_url_unchanged():
    url = "http://localhost:11434"
    assert AgentOrchestrator._normalize_base_url(url) == url


def test_normalize_base_url_preserves_trailing_slash():
    url = "http://localhost:11434/"
    assert AgentOrchestrator._normalize_base_url(url) == url


# ─────────────────────────────────────────
#   _build_config
# ─────────────────────────────────────────

def test_build_config_sets_thread_id():
    config = AgentOrchestrator._build_config("thread-42")
    assert config["configurable"]["thread_id"] == "thread-42"


def test_build_config_no_callback_by_default():
    config = AgentOrchestrator._build_config("thread-1")
    assert "callbacks" not in config


def test_build_config_attaches_callback_when_provided():
    from unittest.mock import MagicMock
    handler = MagicMock()
    config = AgentOrchestrator._build_config("thread-2", langfuse_handler=handler)
    assert config["callbacks"] == [handler]


def test_build_config_none_handler_omits_callbacks():
    config = AgentOrchestrator._build_config("thread-3", langfuse_handler=None)
    assert "callbacks" not in config
