import pytest
from langchain_core.messages import SystemMessage, HumanMessage

from src.agents.infrastructure.adapters.base_agent import BaseAgent


class DummyAgent(BaseAgent):
    def _create_llm(self, token: str):
        return object()

    @property
    def _retry_exceptions(self):
        return (Exception,)


def test_build_messages_formats_context_placeholder_inline():
    """When {context} appears in role_description it is substituted inline
    and the 'USE THIS CONTEXT TO ANSWER' section is NOT appended."""
    agent = DummyAgent(
        name="Writer",
        role_description="INTERNAL DATA:\n{context}\n\nEmployee query: {user_input}",
        token="fake-token",
    )

    messages = agent._build_messages(
        user_input="Book a trip to London",
        context="Policies apply",
        chat_history=None,
    )

    system_content = messages[0].content
    assert isinstance(messages[0], SystemMessage)
    # Placeholder was substituted inline
    assert "Policies apply" in system_content
    assert "Book a trip to London" in system_content
    # Literal placeholder is gone
    assert "{context}" not in system_content
    assert "{user_input}" not in system_content
    # Standalone section NOT added (already embedded)
    assert "USE THIS CONTEXT TO ANSWER" not in system_content
    # user_input still present as HumanMessage
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == "Book a trip to London"


def test_build_messages_appends_context_when_no_placeholder():
    """When role_description has no {context} placeholder, context is appended
    as a standalone 'USE THIS CONTEXT TO ANSWER' section."""
    agent = DummyAgent(
        name="Operator",
        role_description="You are a strict financial auditor.",
        token="fake-token",
    )

    messages = agent._build_messages(
        user_input="Audit this plan",
        context="Max hotel: 150 EUR/night",
        chat_history=None,
    )

    system_content = messages[0].content
    assert "USE THIS CONTEXT TO ANSWER" in system_content
    assert "Max hotel: 150 EUR/night" in system_content
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == "Audit this plan"


def test_build_messages_unknown_placeholders_kept_literal():
    """Placeholders not in (context, user_input) are kept as-is."""
    agent = DummyAgent(
        name="Dummy",
        role_description="Keep this literal {value} intact.",
        token="fake-token",
    )

    messages = agent._build_messages(
        user_input="Hello",
        context="",
        chat_history=None,
    )

    assert "{value}" in messages[0].content


# ─────────────────────────────────────────
#   _strip_markdown_fences
# ─────────────────────────────────────────

from src.agents.infrastructure.adapters.base_agent import _strip_markdown_fences, generic_token_counter
from langchain_core.messages import AIMessage


def test_strip_markdown_fences_removes_json_block():
    raw = "```json\n{\"key\": \"value\"}\n```"
    assert _strip_markdown_fences(raw) == '{"key": "value"}'


def test_strip_markdown_fences_removes_plain_block():
    raw = "```\nsome text\n```"
    assert _strip_markdown_fences(raw) == "some text"


def test_strip_markdown_fences_leaves_plain_text_unchanged():
    raw = '{"key": "value"}'
    assert _strip_markdown_fences(raw) == raw


def test_strip_markdown_fences_strips_surrounding_whitespace():
    raw = "  ```json\n{}\n```  "
    assert _strip_markdown_fences(raw) == "{}"


# ─────────────────────────────────────────
#   generic_token_counter
# ─────────────────────────────────────────

def test_generic_token_counter_estimates_by_chars():
    msgs = [HumanMessage(content="abcd")]  # 4 chars → 1 token
    assert generic_token_counter(msgs) == 1


def test_generic_token_counter_sums_multiple_messages():
    msgs = [
        HumanMessage(content="aaaabbbb"),   # 8 chars
        AIMessage(content="cccc"),          # 4 chars
    ]
    assert generic_token_counter(msgs) == 3  # 12 // 4


def test_generic_token_counter_handles_empty_messages():
    assert generic_token_counter([]) == 0


def test_generic_token_counter_handles_list_content():
    """Messages with list-of-blocks content (multimodal) are handled."""
    msg = HumanMessage(content=[{"type": "text", "text": "helloworld"}])  # 10 chars
    assert generic_token_counter([msg]) == 2


# ─────────────────────────────────────────
#   BaseAgent.__init__ validation
# ─────────────────────────────────────────

def test_base_agent_raises_without_token_and_no_base_url():
    with pytest.raises(ValueError, match="token is missing"):
        DummyAgent(name="X", role_description="Desc", token="")


def test_base_agent_allows_empty_token_when_base_url_set():
    from src.agents.domain.models import AgentConfig
    cfg = AgentConfig(base_url="http://localhost:11434")
    # Should NOT raise — local LLM doesn't require a real API key
    agent = DummyAgent(name="Local", role_description="Desc", token="", config=cfg)
    assert agent.name == "Local"


# ─────────────────────────────────────────
#   _build_messages with chat_history
# ─────────────────────────────────────────

def test_build_messages_includes_chat_history():
    from langchain_core.messages import AIMessage as AI
    agent = DummyAgent(
        name="Writer",
        role_description="You answer questions.",
        token="fake-token",
    )
    history = [HumanMessage(content="previous q"), AI(content="previous a")]

    messages = agent._build_messages(
        user_input="New question",
        context="",
        chat_history=history,
    )

    # [SystemMessage, HumanMessage(prev), AIMessage(prev), HumanMessage(new)]
    assert len(messages) == 4
    assert messages[1].content == "previous q"
    assert messages[2].content == "previous a"
    assert messages[-1].content == "New question"


def test_build_messages_no_history_has_two_messages():
    agent = DummyAgent(
        name="Writer",
        role_description="You answer questions.",
        token="fake-token",
    )
    messages = agent._build_messages(user_input="Hello", context="", chat_history=None)
    assert len(messages) == 2

