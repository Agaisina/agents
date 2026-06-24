import logging
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Type

from langchain.agents import create_agent as create_llm_agent
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, trim_messages
from langchain_core.runnables import RunnableConfig

from src.agents.domain.interfaces import IAgent
from src.agents.domain.models import AgentConfig

logger = logging.getLogger(__name__)


class AgentExecutionError(Exception):
    """Raised when the LLM agent fails during execution."""


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` fences that some LLMs wrap around JSON output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        return "\n".join(inner).strip()
    return stripped


def generic_token_counter(messages: List[BaseMessage]) -> int:
    """Estimate token count using 4 characters per token."""
    total_chars = 0
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total_chars += len(block.get("text", ""))
    return total_chars // 4


class BaseAgent(IAgent, ABC):
    """Base class with shared behavior for provider-specific agents."""

    def __init__(
        self,
        name: str,
        role_description: str,
        token: str,
        config: Optional[AgentConfig] = None,
        tools: Optional[List[Any]] = None,
    ):
        self._name = name
        self.role_description = role_description
        self.config = config or AgentConfig()
        self._tools: List[Any] = tools or []

        if not token and not self.config.base_url:
            raise ValueError(f"Cannot initialize '{self._name}': token is missing or empty.")

        self._llm = self._create_llm(token)
        self.react_agent = None

        if self._tools:
            logger.info(
                f"🛠️ Equipping agent '{self._name}' with {len(self._tools)} tools (ReAct archetype)."
            )
            self._setup_tool_executor()
        else:
            logger.info(f"⚡ Configuring agent '{self._name}' without tools (Reflex archetype).")

    @abstractmethod
    def _create_llm(self, token: str) -> Any:
        """Construct the provider-specific language model client."""
        pass

    @property
    @abstractmethod
    def _retry_exceptions(self) -> Tuple[Type[BaseException], ...]:
        """Exceptions that should trigger automatic retry."""
        pass

    @property
    def _token_counter(self):
        return generic_token_counter

    def _setup_tool_executor(self) -> None:
        """Create and attach a LangChain ReAct agent with the configured tools."""
        system_prompt = f"You are {self._name}. {self.role_description}"
        self.react_agent = create_llm_agent(
            model=self._llm,
            tools=self._tools,
            system_prompt=system_prompt,
        )

    def _trim_history(self, history: List[BaseMessage], max_tokens: int) -> List[BaseMessage]:
        """Truncate ``history`` to stay within ``max_tokens`` using a last-N strategy."""
        return trim_messages(
            history,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=self._token_counter,
            include_system=True,
        )

    def _build_messages(
        self,
        user_input: str,
        context: str,
        chat_history: Optional[List[BaseMessage]],
    ) -> List[BaseMessage]:
        """Assemble the final message list for an LLM call.

        Builds a ``[SystemMessage, *history, HumanMessage]`` sequence.
        Injects ``context`` either via the ``{context}`` placeholder in the
        role prompt or as an appended block when the placeholder is absent.

        Args:
            user_input: The employee's query or the upstream node's instruction.
            context: RAG chunks or structured JSON to ground the agent's answer.
            chat_history: Previous messages in the current node cycle.

        Returns:
            A list of ``BaseMessage`` objects ready for ``llm.invoke()``.
        """
        class _SafeDict(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        system_content = f"You are {self.name}. {self.role_description}"

        # Detect which known placeholders are present BEFORE formatting
        has_context_placeholder = "{context}" in system_content

        # Substitute {context} and {user_input} inline wherever they appear
        system_content = system_content.format_map(
            _SafeDict(context=context or "", user_input=user_input)
        )

        # Only append the standalone context block when the prompt doesn't already
        # embed {context} inline — avoids duplicating the context.
        if context and not has_context_placeholder:
            system_content += f"\n\nUSE THIS CONTEXT TO ANSWER:\n{context}"

        messages: List[BaseMessage] = [SystemMessage(content=system_content)]
        if chat_history:
            messages.extend(chat_history)
        messages.append(HumanMessage(content=user_input))
        return messages


    def _invoke_with_retries(
        self, messages: List[BaseMessage], config: Optional[RunnableConfig] = None
    ) -> str:
        """Invoke the LLM with up to 3 attempts and exponential back-off.

        Args:
            messages: The fully-assembled message list to send.
            config: Optional LangChain ``RunnableConfig`` (tracing callbacks, etc.).

        Returns:
            The LLM's text response as a plain string.

        Raises:
            The last caught exception if all retry attempts are exhausted.
        """
        attempts = 5
        for attempt in range(1, attempts + 1):
            try:
                response = self.llm.invoke(messages, config=config)
                return str(response.content)
            except self._retry_exceptions as error:
                if attempt >= attempts:
                    raise
                # Use a longer cap (60 s) so the back-off covers API-returned
                # retryDelay values (e.g. Gemini 429 asks for ~18 s) and
                # transient 503 UNAVAILABLE errors from high demand.
                wait_seconds = min(2 ** attempt * 5, 60)
                logger.warning(
                    "Retrying LLM request (%s/%s) after %s seconds due to %s",
                    attempt,
                    attempts,
                    wait_seconds,
                    type(error).__name__,
                )
                time.sleep(wait_seconds)

    def invoke_with_tools(
        self,
        messages: List[BaseMessage],
        config: Optional[RunnableConfig] = None,
    ) -> Any:
        """Invoke the underlying model with bound tools for LangGraph tool-calling loops."""
        if not self._tools:
            raise AgentExecutionError(f"Agent '{self.name}' has no tools configured.")

        try:
            tool_model = self._llm.bind_tools(self._tools)
            return tool_model.invoke(messages, config=config)
        except Exception as error:
            error_msg = f"Agent '{self.name}' failed during tool-aware invocation. Root cause: {error}"
            logger.error(error_msg)
            raise AgentExecutionError(error_msg) from error

    @property
    def name(self) -> str:
        return self._name

    @property
    def llm(self) -> Any:
        return self._llm

    @property
    def tools(self) -> List[Any]:
        return self._tools

    def run(
        self,
        user_input: str,
        context: str = "",
        chat_history: Optional[List[BaseMessage]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """Execute the agent's core logic and return a plain-text response.

        Uses the ReAct agent when tools are configured, otherwise calls the
        LLM directly via :meth:`_invoke_with_retries`.

        Args:
            user_input: The main query or instruction.
            context: Optional grounding data (RAG chunks, JSON facts).
            chat_history: Previous messages to include in the prompt window.
            config: Optional LangChain ``RunnableConfig`` for tracing.

        Returns:
            The agent's final text response.
        """
        logger.info(f"Agent '{self.name}' processing input...")
        history = chat_history or []
        if history:
            history = self._trim_history(history, max_tokens=2000)

        try:
            if self.react_agent:
                input_content = user_input
                if context:
                    input_content = f"REFERENCE CONTEXT:\n{context}\n\nUSER QUESTION: {user_input}"

                result = self.react_agent.invoke(
                    {"messages": history + [HumanMessage(content=input_content)]},
                    config=config,
                )
                return result["messages"][-1].content

            messages = self._build_messages(user_input, context, history)
            return self._invoke_with_retries(messages, config=config)
        except Exception as error:
            error_msg = f"Agent '{self.name}' failed. Root cause: {error}"
            logger.error(error_msg)
            raise AgentExecutionError(error_msg) from error

    def run_structured(
        self,
        schema: type,
        user_input: str,
        chat_history: Optional[List[BaseMessage]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> Any:
        logger.info(f"Agent '{self.name}' executing structured query (Tool-Aware).")
        history = chat_history or []
        if history:
            history = self._trim_history(history, max_tokens=3000)

        if self.tools and self.react_agent:
            json_instruction = (
                f"\n\nCRITICAL: After using tools, you MUST provide a final answer "
                f"strictly in JSON format matching this schema: {schema.model_json_schema()}. "
                "Do not include any other text, explanations or markdown blocks."
            )
            
            raw_output = self.run(
                user_input=f"{user_input}{json_instruction}",
                context="", 
                chat_history=history,
                config=config
            )
            return schema.model_validate_json(raw_output.strip())

        else:
            try:
                structured_llm = self._llm.with_structured_output(schema)
                messages = self._build_messages(user_input, context="", chat_history=history)
                attempts = 5
                for attempt in range(1, attempts + 1):
                    try:
                        return structured_llm.invoke(messages, config=config)
                    except self._retry_exceptions as rate_err:
                        if attempt >= attempts:
                            raise
                        wait_seconds = min(2 ** attempt * 5, 60)
                        logger.warning(
                            "Retrying structured LLM request (%s/%s) after %s s due to %s",
                            attempt, attempts, wait_seconds, type(rate_err).__name__,
                        )
                        time.sleep(wait_seconds)
            except NotImplementedError:
                logger.warning(
                    "Agent '%s': with_structured_output not supported — using JSON prompt fallback.",
                    self.name,
                )
                schema_fields = {
                    k: ("REQUIRED" if f.is_required() else "optional")
                    for k, f in schema.model_fields.items()
                }
                json_instruction = (
                    f"\n\nOutput ONLY a valid JSON object (no markdown, no explanation) "
                    f"with these fields filled with actual values from your analysis: "
                    f"{list(schema_fields.keys())}. "
                    f"Schema: {schema.model_json_schema()}"
                )
                messages = self._build_messages(
                    f"{user_input}{json_instruction}", context="", chat_history=history
                )
                raw = self._invoke_with_retries(messages, config=config)
                return schema.model_validate_json(_strip_markdown_fences(raw))
