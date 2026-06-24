import pytest
from unittest.mock import MagicMock
from langchain_core.tools import tool
from src.agents.domain.interfaces import IAgent


class MockAgent(IAgent):
    def __init__(self, name="Mock", **kwargs):
        self._name = name

    @property
    def name(self): return self._name

    def run(self, user_input, context="", chat_history=None):
        return f"Respuesta mock para: {user_input}"


@pytest.fixture
def mock_credentials():
    return {"api_key": "fake_key", "model_id": "fake_model"}


@pytest.fixture
def mock_agent():
    """A lightweight IAgent mock suitable for registry/factory tests."""
    agent = MagicMock(spec=IAgent)
    agent.name = "MockAgent"
    return agent


@pytest.fixture
def mock_tools():
    """Two minimal LangChain tools for testing tool composition."""
    @tool
    def tool_a(x: str) -> str:
        """Tool A."""
        return x

    @tool
    def tool_b(x: str) -> str:
        """Tool B."""
        return x

    return [tool_a, tool_b]


@pytest.fixture
def mock_vector_db():
    """Mock VectorStoreManager that returns a stub retriever."""
    db = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = []
    db.get_retriever.return_value = retriever
    return db


@pytest.fixture
def mock_graph_state():
    """A minimal AgentState dict for graph routing tests."""
    from src.workflow.graph import AgentState
    from langchain_core.messages import HumanMessage
    return AgentState(
        messages=[HumanMessage(content="Test query")],
        rag_facts=None,
        planner_facts=None,
        operator_facts=None,
        manager_override=None,
        final_response=None,
        planner_tool_iterations=0,
    )