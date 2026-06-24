import logging
import sqlite3
from typing import TypedDict, Annotated, Optional, Any, Union
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.application.registry import AgentRegistry
from src.agents.domain.models import RAGFacts, PlannerFacts, OperatorFacts
from src.agents.infrastructure.tools.geo_tools import create_optimal_hotel_tool

logger = logging.getLogger(__name__)

RESEARCHER_NODE = "researcher"
PLANNER_NODE = "planner"
PLANNER_TOOLS_NODE = "planner_tools"
PLANNER_EXTRACTOR_NODE = "planner_extractor"
OPERATOR_NODE = "operator"
OPERATOR_TOOLS_NODE = "operator_tools"
OPERATOR_EXTRACTOR_NODE = "operator_extractor"
WRITER_NODE = "writer"
HUMAN_REVIEW_NODE = "human_review"

AUTO_APPROVED = "auto_approved"
NEEDS_REVIEW = "needs_review"
APPROVED = "approved"
REJECTED = "rejected"

DEFAULT_RAG_CONTEXT = "No database connected."
DEFAULT_RAG_CONTEXT_FALLBACK = "No database context available."
MAX_PLANNER_TOOL_ITERATIONS = 5 

PLANNER_TOOL_RULE = (
    "ANTI-LOOP RULE: If a tool returns 'not_found' or empty, DO NOT call it again for the same parameter. "
    "Accept the lack of data and proceed to formulate the plan."
)
OPERATOR_TOOL_RULE = "Use your tools (safe_calculator, convert_currency) to verify the Total Cost of Ownership (TCO)."
PLANNER_EXTRACTION_PROMPT = "Based on the conversation and tool results, extract the EXACT logistics and prices."
OPERATOR_EXTRACTION_PROMPT = "Based on your calculations, generate the final OperatorFacts audit report."
RAG_QUERY_TEMPLATE = (
    "Find corporate travel policies, MAXIMUM accommodation limits (per diem), "
    "allowed flight cabin classes, and high-risk destination rules for this user request: {user_query}"
)
PLANNER_POLICY_PREFIX = "CRITICAL POLICIES:"
OPERATOR_POLICY_PREFIX = "POLICIES:"
OPERATOR_PLAN_PREFIX = "PROPOSED PLAN TO AUDIT:"

# ==========================================
# 1. GRAPH STATE
# ==========================================
class AgentState(TypedDict):
    """Shared mutable state passed between every node in the LangGraph workflow.

    All fields are optional at startup and are progressively populated as the
    graph executes. The ``messages`` list uses the ``add_messages`` reducer so
    that each node's new messages are appended rather than replaced.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    rag_facts: Optional[RAGFacts]
    planner_facts: Optional[PlannerFacts]
    operator_facts: Optional[OperatorFacts]
    manager_override: Optional[bool]
    final_response: Optional[str]
    planner_tool_iterations: int  # Tracks tool-call rounds to prevent infinite loops
    planner_turn_start: int       # Index into messages where the current planner cycle starts
    operator_turn_start: int      # Index into messages where the current operator cycle starts

# ==========================================
# 2. WORKFLOW BUILDER
# ==========================================
class WorkflowBuilder:
    """Assembles and compiles the Veritas Corporate multi-agent LangGraph workflow.

    The graph follows a **Researcher → Planner → Operator → Writer** pipeline
    with optional HITL (Human-in-the-Loop) interruption between the Operator
    and Writer nodes. The Planner and Operator each run a tool-calling loop
    with a hard iteration cap to prevent infinite cycles.
    """
    def __init__(self, registry: AgentRegistry, db_manager: Any = None):
        self.registry = registry
        self.db_manager = db_manager
        self._setup_agents()

    def _setup_agents(self) -> None:
        """Build all agent instances and their associated tool nodes."""
        self.researcher_agent = self.registry.get_agent("corporate_researcher")

        dynamic_planner_tools = []
        if self.db_manager:
            hotel_retriever = self.db_manager.get_retriever(collection_name="preferred_vendors")
            dynamic_planner_tools.append(
                create_optimal_hotel_tool(hotel_retriever, self.researcher_agent.llm)
            )

        self.planner_agent = self.registry.get_agent("planner", tools=dynamic_planner_tools)
        self.operator_agent = self.registry.get_agent("operator")
        self.writer_agent = self.registry.get_agent("writer")

        self.planner_tool_node = ToolNode(self.planner_agent.tools)
        self.operator_tool_node = ToolNode(self.operator_agent.tools)


    # ==========================================
    # 3. NODE DEFINITIONS
    # ==========================================

    def _get_rag_context(self, user_query: str) -> str:
        """Retrieve relevant policy fragments from the vector store via RAG.

        Returns a fallback string when the DB is unavailable or the query fails.
        """
        if not self.db_manager:
            return DEFAULT_RAG_CONTEXT

        try:
            retriever = self.db_manager.get_retriever(collection_name="te_policies_compliance")
            smart_query = RAG_QUERY_TEMPLATE.format(user_query=user_query)

            docs = retriever.invoke(smart_query)
            logger.info("📚 RAG retrieved %d policy fragments.", len(docs))
            return "\n\n".join([d.page_content for d in docs])
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.error("RAG Researcher error: %s", exc)
            return DEFAULT_RAG_CONTEXT_FALLBACK

    def _build_policy_prompt(self, state: AgentState, user_query: str, context: str) -> str:
        """Build the structured prompt payload sent to the Researcher agent."""
        return (
            f"User role: {state.get('user_role', 'Employee')}\n"
            f"User request: {user_query}\n\n"
            f"CORPORATE POLICY CONTEXT:\n{context}"
        )

    def _build_policy_payload(self, rag_json: str, plan_json: Optional[str] = None, *, for_operator: bool = False) -> str:
        """Format the RAG facts (and optionally the plan) into a compact policy block."""
        if for_operator:
            sections = [f"{OPERATOR_POLICY_PREFIX}\n{rag_json}"]
            if plan_json:
                sections.append(f"{OPERATOR_PLAN_PREFIX}\n{plan_json}")
            return "\n\n".join(sections)
        return f"{PLANNER_POLICY_PREFIX}\n{rag_json}"

    def _format_state_context(self, state: AgentState) -> str:
        """Serialize all accumulated facts into a single context string for the Writer."""
        parts = [
            f"POLICIES: {state.get('rag_facts')}",
            f"ITINERARY: {state.get('planner_facts')}",
            f"FINANCIAL_VERDICT: {state.get('operator_facts')}",
        ]
        if state.get("manager_override") is True:
            parts.append("MANAGER_OVERRIDE: Approved by manager.")
        elif state.get("manager_override") is False:
            parts.append("MANAGER_OVERRIDE: Rejected by manager.")
        return "\n".join(parts)

    @staticmethod
    def _serialize_facts(facts: Optional[Any]) -> str:
        """Return a JSON string for ``facts``, or ``"{}"`` when ``facts`` is ``None``."""
        return facts.model_dump_json() if facts else "{}"

    def _get_latest_user_input(self, state: AgentState) -> str:
        """Extract the text content from the most recent message in the state."""
        messages = state.get("messages") or []
        if not messages:
            return ""
        last_message = messages[-1]
        return getattr(last_message, "content", "") or ""

    def _build_system_message(self, role_description: str, policy_payload: str, footer: Optional[str] = None) -> SystemMessage:
        """Combine a role description, policy payload, and optional footer into a SystemMessage."""
        sections = [role_description]
        if policy_payload:
            sections.append(policy_payload)
        if footer:
            sections.append(footer)
        return SystemMessage(content="\n\n".join(section for section in sections if section))

    def _researcher_node(self, state: AgentState):
        """Run the Compliance Researcher to extract RAGFacts from the policy knowledge base."""
        logger.info("🔍 Running Researcher Node...")
        
        user_query = self._get_latest_user_input(state)
        context = self._get_rag_context(user_query)
        prompt = self._build_policy_prompt(state, user_query, context)

        rag_data = self.researcher_agent.run_structured(
            schema=RAGFacts,
            user_input=prompt,
            chat_history=[],
        )
        
        return {"rag_facts": rag_data}


    def _planner_node(self, state: AgentState):
        """Run the Logistics Planner to build or revise the travel itinerary (PlannerFacts).

        Tracks the current cycle's start index in ``planner_turn_start`` so the
        extractor node can scope its context correctly. A tool loop is entered
        when the last message is a ``ToolMessage``.
        """
        logger.info("📅 Running Planner Node...")

        rag_json = self._serialize_facts(state.get("rag_facts"))
        policy_payload = self._build_policy_payload(rag_json)
        footer = PLANNER_TOOL_RULE
        sys_msg = self._build_system_message(self.planner_agent.role_description, policy_payload, footer)

        messages = state.get("messages") or []
        last_is_tool = bool(messages) and isinstance(messages[-1], ToolMessage)

        updates: dict = {}

        if last_is_tool:
            start_idx = state.get("planner_turn_start") or 0
        else:
            start_idx = len(messages)
            updates["planner_turn_start"] = start_idx

        # Pass: original employee request + any manager feedback + current planner cycle
        user_msgs    = messages[:1]
        manager_msgs = [m for m in messages[1:start_idx] if isinstance(m, HumanMessage)]
        cycle_msgs   = messages[start_idx:]
        planner_context = user_msgs + manager_msgs + cycle_msgs

        response = self.planner_agent.invoke_with_tools([sys_msg] + planner_context)
        iterations = (state.get("planner_tool_iterations") or 0) + 1
        updates.update({"messages": [response], "planner_tool_iterations": iterations})
        return updates


    def _planner_extractor_node(self, state: AgentState):
        """Extract structured PlannerFacts from the Planner's tool-augmented conversation."""
        logger.info("⚙️ Extracting PlannerFacts...")
        start_idx = state.get("planner_turn_start") or 0
        planner_messages = (state.get("messages") or [])[start_idx:]

        planner_data = self.planner_agent.run_structured(
            schema=PlannerFacts,
            user_input=PLANNER_EXTRACTION_PROMPT,
            chat_history=planner_messages,
        )
        return {"planner_facts": planner_data}


    def _operator_node(self, state: AgentState):
        """Run the Financial Operator to audit the plan and compute Total Cost of Ownership.

        On a fresh call the operator receives only the system message (policy +
        plan already embedded). On tool-loop continuation it receives the
        messages since the last ``operator_turn_start``.
        """
        logger.info("⚖️ Running Financial Operator Node...")
        rag_json = self._serialize_facts(state.get("rag_facts"))
        plan_json = self._serialize_facts(state.get("planner_facts"))

        policy_payload = self._build_policy_payload(rag_json, plan_json, for_operator=True)
        footer = OPERATOR_TOOL_RULE
        sys_msg = self._build_system_message(self.operator_agent.role_description, policy_payload, footer)

        messages = state.get("messages") or []
        last_is_tool = bool(messages) and isinstance(messages[-1], ToolMessage)

        updates: dict = {}

        if last_is_tool:
            # Continuing tool loop — pass only current operator cycle messages
            start_idx = state.get("operator_turn_start") or 0
            op_context = messages[start_idx:]
        else:
            # Fresh operator start — sys_msg already has rag_json + plan_json
            start_idx = len(messages)
            updates["operator_turn_start"] = start_idx
            op_context = []

        response = self.operator_agent.invoke_with_tools([sys_msg] + op_context)
        updates["messages"] = [response]
        return updates


    def _operator_extractor_node(self, state: AgentState):
        """Extract structured OperatorFacts with the compliance verdict and approval routing."""
        logger.info("📑 Extracting final OperatorFacts...")
        start_idx = state.get("operator_turn_start") or 0
        operator_messages = (state.get("messages") or [])[start_idx:]

        final_audit = self.operator_agent.run_structured(
            schema=OperatorFacts,
            user_input=OPERATOR_EXTRACTION_PROMPT,
            chat_history=operator_messages,
        )
        return {"operator_facts": final_audit}

    def _writer_node(self, state: AgentState):
        """Produce the final employee-facing travel report from all accumulated facts."""
        logger.info("✍️ Running Writer Node...")
        messages = state.get("messages") or []
        employee_query = messages[0].content if messages else ""
        combined_context = self._format_state_context(state)

        final_text = self.writer_agent.run(
            user_input=employee_query,
            context=combined_context,
            chat_history=None,
        )
        return {"final_response": final_text}


    def _planner_tools_condition(self, state: AgentState) -> str:
        """Custom routing for the planner loop: enforces a hard iteration limit."""
        iterations = state.get("planner_tool_iterations") or 0
        if iterations >= MAX_PLANNER_TOOL_ITERATIONS:
            logger.warning(
                "⚠️ Planner reached max tool iterations (%d). Forcing extraction.",
                MAX_PLANNER_TOOL_ITERATIONS,
            )
            return "__end__"
        return tools_condition(state)

    def _route_from_operator(self, state: AgentState):
        """Route after the Operator: auto-approve, send to human review, or force re-plan."""
        facts = state.get("operator_facts")

        if not facts:
            logger.warning("⚠️ operator_facts is empty. Sending to human review.")
            return NEEDS_REVIEW

        routing_decision = facts.approval_routing

        if routing_decision == "AUTO_APPROVED":
            logger.info("✅ Trip fully compliant. Fast-tracking to Writer.")
            return AUTO_APPROVED

        if routing_decision == "REJECTED_MUST_REPLAN":
            logger.warning("❌ Critical policy violation (e.g. 5-star hotel without approval). Returning to Planner.")
            return REJECTED

        logger.warning("⚠️ Requires human review for policy: %s", routing_decision)
        return NEEDS_REVIEW

    def _human_review_node(self, state: AgentState):
        """Placeholder node that suspends execution for manager approval (HITL)."""
        logger.info("👨‍💼 Manager has reviewed the trip.")
        return {}

    def _route_from_human(self, state: AgentState):
        """Route after the manager decision: proceed to Writer (approved) or back to Planner (rejected)."""
        logger.info("🔀 Evaluating Manager decision...")
        override = state.get("manager_override")

        if override is True:
            logger.info("✅ Manager approved the exception.")
            return APPROVED

        if override is None:
            logger.warning("⚠️ manager_override is None — treating as rejection (safety default).")

        logger.info("❌ Manager rejected. Returning to Planner.")
        return REJECTED


    # ==========================================
    # 4. GRAPH COMPILATION
    # ==========================================

    def compile(self, checkpointer: Optional[Union[Any, None]] = None) -> Any:
        """Wire all nodes and edges and return a compiled ``CompiledStateGraph``.

        If no ``checkpointer`` is provided, an in-memory SQLite saver is used
        (suitable for testing and local development only).

        Args:
            checkpointer: A LangGraph checkpoint backend
                          (``SqliteSaver`` or ``PostgresSaver``).

        Returns:
            A ``CompiledStateGraph`` ready to call ``.invoke()`` or ``.stream()`` on.
        """
        logger.info("🏗️ Building the Symmetric Multi-Agent Graph...")
        workflow = StateGraph(AgentState)
        
        workflow.add_node(RESEARCHER_NODE, self._researcher_node)
        workflow.add_node(PLANNER_NODE, self._planner_node)
        workflow.add_node(PLANNER_TOOLS_NODE, self.planner_tool_node)
        workflow.add_node(PLANNER_EXTRACTOR_NODE, self._planner_extractor_node)
        workflow.add_node(OPERATOR_NODE, self._operator_node)
        workflow.add_node(OPERATOR_TOOLS_NODE, self.operator_tool_node)
        workflow.add_node(OPERATOR_EXTRACTOR_NODE, self._operator_extractor_node)
        workflow.add_node(WRITER_NODE, self._writer_node)
        workflow.add_node(HUMAN_REVIEW_NODE, self._human_review_node)

        workflow.add_edge(START, RESEARCHER_NODE)
        workflow.add_edge(RESEARCHER_NODE, PLANNER_NODE)
        
        workflow.add_conditional_edges(
            PLANNER_NODE, self._planner_tools_condition,
            {"tools": PLANNER_TOOLS_NODE, "__end__": PLANNER_EXTRACTOR_NODE}
        )
        workflow.add_edge(PLANNER_TOOLS_NODE, PLANNER_NODE)

        workflow.add_edge(PLANNER_EXTRACTOR_NODE, OPERATOR_NODE)
        
        workflow.add_conditional_edges(
            OPERATOR_NODE, tools_condition,
            {"tools": OPERATOR_TOOLS_NODE, "__end__": OPERATOR_EXTRACTOR_NODE}
        )
        workflow.add_edge(OPERATOR_TOOLS_NODE, OPERATOR_NODE)
        
        workflow.add_conditional_edges(
            OPERATOR_EXTRACTOR_NODE,
            self._route_from_operator,
            {
                AUTO_APPROVED: WRITER_NODE,
                NEEDS_REVIEW: HUMAN_REVIEW_NODE,
                REJECTED: PLANNER_NODE,
            }
        )

        workflow.add_conditional_edges(
            HUMAN_REVIEW_NODE,
            self._route_from_human,
            {
                APPROVED: WRITER_NODE,
                REJECTED: PLANNER_NODE,
            }
        )

        workflow.add_edge(WRITER_NODE, END)
        
        if checkpointer is None:
            checkpointer = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
        return workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=[HUMAN_REVIEW_NODE]
        )