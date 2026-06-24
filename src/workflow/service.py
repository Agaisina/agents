import asyncio
import logging
from typing import Optional

from src.config import settings
from src.agents.application.registry import AgentRegistry
from src.workflow.graph import WorkflowBuilder
from src.infrastructure.vector_db import VectorStoreManager
from src.infrastructure.checkpoint_factory import CheckpointFactory
from src.infrastructure.langfuse_monitor import LangFuseMonitor

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    This class encapsulates the entire multi-agent workflow
    It initializes all components and exposes methods to process user queries 
    both synchronously and via streaming (Server-Sent Events).
    Integrates LangFuse for full observability and monitoring.
    """
    def __init__(self, checkpointer=None):
        logger.info("⚙️ Initializing Agent Engine...")
        try:
            self.langfuse = LangFuseMonitor()
            self.db = VectorStoreManager()
            self.checkpointer = checkpointer or CheckpointFactory.get_checkpointer()
            self.registry = AgentRegistry(
                api_key=settings.token,
                model_id=settings.models.model_id,
                base_url=self._normalize_base_url(settings.models.get("base_url")),
                provider=settings.models.get("provider"),
            )

            builder = WorkflowBuilder(registry=self.registry, db_manager=self.db)
            self.graph = builder.compile(checkpointer=self.checkpointer)

            logger.info("✅ Agent Engine ready and compiled.")
        except Exception as e:
            logger.error("❌ Fatal error while building the engine: %s", e, exc_info=True)
            raise

    @classmethod
    async def create_async(cls) -> "AgentOrchestrator":
        """Async factory — use this from FastAPI lifespan.

        Creates an async-compatible checkpointer (``AsyncSqliteSaver`` or
        ``AsyncPostgresSaver``) so that ``graph.astream()``, ``graph.ainvoke()``,
        ``graph.aget_state()``, and ``graph.aupdate_state()`` all work correctly.
        """
        checkpointer = await CheckpointFactory.aget_checkpointer()
        return cls(checkpointer=checkpointer)

    @staticmethod
    def _normalize_base_url(raw_base_url: Optional[str]) -> Optional[str]:
        """Return ``None`` when ``raw_base_url`` is empty, whitespace, or the string ``"none"``."""
        if not raw_base_url or str(raw_base_url).strip().lower() == "none":
            return None
        return raw_base_url

    @staticmethod
    def _build_config(thread_id: str, langfuse_handler=None) -> dict:
        """Build the LangGraph ``RunnableConfig`` for a given thread.

        Attaches the LangFuse callback when available so every node invocation
        is traced automatically.
        """
        config = {"configurable": {"thread_id": thread_id}}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]
        return config

    def process_query(self, query: str, thread_id: str) -> dict:
        """
        Process a user query synchronously.
        Returns a dict with the response and the trace_id.
        """
        initial_state = {"messages": [HumanMessage(content=query)]}

        langfuse_handler = LangFuseMonitor.get_callback()
        config = self._build_config(thread_id, langfuse_handler)
        
        try:
            logger.info("⚙️ Executing graph with Langfuse tracing...")
            final_state = self.graph.invoke(initial_state, config=config)
            response = final_state.get("final_response", "Error: Empty response from the graph.")
            
            trace_id = None
            if langfuse_handler:
                trace_id = langfuse_handler.last_trace_id
                try:
                    LangFuseMonitor.flush()
                    logger.info("✅ Trace successfully flushed! Trace ID: %s", trace_id)
                except Exception as flush_err:
                    logger.warning("⚠️ LangFuse flush failed (trace may be lost): %s", flush_err)
            
            return {
                "respuesta": response,
                "trace_id": trace_id
            }
            
        except Exception as e:
            logger.error("❌ Error while executing the graph: %s", e, exc_info=True)
            if langfuse_handler:
                try:
                    LangFuseMonitor.flush()
                except Exception as flush_err:
                    logger.warning("⚠️ LangFuse flush failed after error: %s", flush_err)
            raise RuntimeError("The agent could not process the request.")


    def stream_query(self, query: str, thread_id: str):
        """
        Generator for reactive UIs (Streamlit / SSE).
        Streams graph events and stops automatically at interrupt_before breakpoints.
        """
        langfuse_handler = LangFuseMonitor.get_callback()
        config = self._build_config(thread_id, langfuse_handler)

        # Stream graph events until the generator is exhausted or the graph
        # hits an interrupt_before breakpoint (Human-in-the-loop).
        initial_state = {"messages": [HumanMessage(content=query)]}

        try:
            for event in self.graph.stream(initial_state, config=config):
                for node_name, node_state in event.items():
                    yield {"type": "node_update", "node": node_name}
                    
                    if node_name == "writer":
                        final_text = node_state.get("final_response", "")
                        words = final_text.split(" ")
                        for i, chunk in enumerate(words):
                            space = " " if i < len(words) - 1 else ""
                            yield {"type": "token", "content": chunk + space}
            
            # Human-in-the-loop
            current_state = self.graph.get_state(config)
            if len(current_state.next) > 0:
                logger.warning("⏸️ Graph execution suspended. Awaiting manager approval.")
                yield {"type": "interrupted"}
                
        except Exception as e:
            logger.error("❌ Error during stream execution: %s", e, exc_info=True)
            yield {"type": "error", "content": "An internal error occurred during streaming."}
        finally:
            if langfuse_handler:
                try:
                    LangFuseMonitor.flush()
                except Exception as flush_err:
                    logger.warning("⚠️ LangFuse flush failed in stream: %s", flush_err)

    def get_trip_state(self, thread_id: str) -> dict:
        """
        Extracts the frozen state of a pending trip so the Manager UI can display it.
        """
        config = self._build_config(thread_id, None)
        state = self.graph.get_state(config)
        current_values = state.values

        return {
            "is_pending": len(state.next) > 0,
            "operator_facts": current_values.get("operator_facts"),
            "planner_facts": current_values.get("planner_facts"),
        }

    def resume_suspended_query(self, thread_id: str, approve: bool, feedback: str = "") -> dict:
        """
        Resumes a paused graph. If rejected, injects the Manager's feedback as a HumanMessage
        so the Planner knows exactly what to fix.
        """
        langfuse_handler = LangFuseMonitor.get_callback()
        config = self._build_config(thread_id, langfuse_handler)

        action = "APPROVED" if approve else "REJECTED"
        logger.info("👤 Human Override for %s: %s - Reason: %s", thread_id, action, feedback)
        
        try:
            # 1. Build the manager decision message that agents will read
            if approve:
                system_instruction = "MANAGER_APPROVAL: The trip has been manually approved. Proceed to final summary."
            else:
                system_instruction = (
                    f"MANAGER_REJECTION: Your proposed trip was rejected. "
                    f"Reason: '{feedback}'. You must re-plan the trip adjusting to these instructions."
                )

            # 2. Append only the new manager message — add_messages reducer handles the merge
            manager_message = HumanMessage(content=system_instruction, name="Manager")
            self.graph.update_state(config, {
                "manager_override": approve,
                "messages": [manager_message],
            })
            
            logger.info("⚙️ Resuming suspended graph...")
            final_state = self.graph.invoke(None, config=config)
            
            response = final_state.get("final_response", "Error: Empty response.")
            
            return {"respuesta": response}
            
        except Exception as e:
            logger.error("❌ Error while resuming: %s", e, exc_info=True)
            raise
        finally:
            if langfuse_handler:
                LangFuseMonitor.flush()

    def shutdown(self):
        """
        Gracefully shutdown the orchestrator.
        Flushes any pending traces to LangFuse.
        """
        logger.info("Shutting down Agent Engine...")
        LangFuseMonitor.flush()
        logger.info("✅ Agent Engine shutdown complete.")

    # ------------------------------------------------------------------
    # Async API  (used by FastAPI — no threadpool required)
    # ------------------------------------------------------------------

    async def aprocess_query(self, query: str, thread_id: str) -> dict:
        """Asynchronously invoke the graph and return the final response.

        Args:
            query: The employee's travel request.
            thread_id: Unique conversation identifier used for checkpointing.

        Returns:
            A dict with keys ``respuesta`` (str) and ``trace_id`` (str | None).
        """
        initial_state = {"messages": [HumanMessage(content=query)]}
        langfuse_handler = LangFuseMonitor.get_callback()
        config = self._build_config(thread_id, langfuse_handler)

        try:
            logger.info("⚙️ Executing graph (async) with LangFuse tracing...")
            final_state = await self.graph.ainvoke(initial_state, config=config)
            response = final_state.get("final_response", "Error: Empty response from the graph.")

            trace_id = None
            if langfuse_handler:
                trace_id = langfuse_handler.last_trace_id
                try:
                    await asyncio.to_thread(LangFuseMonitor.flush)
                    logger.info("✅ Trace successfully flushed! Trace ID: %s", trace_id)
                except Exception as flush_err:
                    logger.warning("⚠️ LangFuse flush failed (trace may be lost): %s", flush_err)

            return {"respuesta": response, "trace_id": trace_id}

        except Exception as e:
            logger.error("❌ Error while executing the graph: %s", e, exc_info=True)
            if langfuse_handler:
                try:
                    await asyncio.to_thread(LangFuseMonitor.flush)
                except Exception as flush_err:
                    logger.warning("⚠️ LangFuse flush failed after error: %s", flush_err)
            raise RuntimeError("The agent could not process the request.")

    async def astream_query(self, query: str, thread_id: str):
        """
        Async generator for FastAPI SSE endpoints.
        Uses graph.astream() — no threadpool or run_in_executor needed.
        """
        langfuse_handler = LangFuseMonitor.get_callback()
        config = self._build_config(thread_id, langfuse_handler)
        initial_state = {"messages": [HumanMessage(content=query)]}

        try:
            async for event in self.graph.astream(initial_state, config=config):
                for node_name, node_state in event.items():
                    yield {"type": "node_update", "node": node_name}

                    if node_name == "writer":
                        final_text = node_state.get("final_response", "")
                        words = final_text.split(" ")
                        for i, chunk in enumerate(words):
                            space = " " if i < len(words) - 1 else ""
                            yield {"type": "token", "content": chunk + space}

            current_state = await self.graph.aget_state(config)
            if len(current_state.next) > 0:
                logger.warning("⏸️ Graph execution suspended. Awaiting manager approval.")
                yield {"type": "interrupted"}

        except Exception as e:
            logger.error("❌ Error during async stream execution: %s", e, exc_info=True)
            yield {"type": "error", "content": "An internal error occurred during streaming."}
        finally:
            if langfuse_handler:
                try:
                    await asyncio.to_thread(LangFuseMonitor.flush)
                except Exception as flush_err:
                    logger.warning("⚠️ LangFuse flush failed in stream: %s", flush_err)

    async def aresume_suspended_query(self, thread_id: str, approve: bool, feedback: str = "") -> dict:
        """Async version of resume_suspended_query using graph.aupdate_state / ainvoke."""
        langfuse_handler = LangFuseMonitor.get_callback()
        config = self._build_config(thread_id, langfuse_handler)

        action = "APPROVED" if approve else "REJECTED"
        logger.info("👤 Human Override for %s: %s - Reason: %s", thread_id, action, feedback)

        try:
            if approve:
                system_instruction = (
                    "MANAGER_APPROVAL: The trip has been manually approved. Proceed to final summary."
                )
            else:
                system_instruction = (
                    f"MANAGER_REJECTION: Your proposed trip was rejected. "
                    f"Reason: '{feedback}'. You must re-plan the trip adjusting to these instructions."
                )

            manager_message = HumanMessage(content=system_instruction, name="Manager")
            await self.graph.aupdate_state(config, {
                "manager_override": approve,
                "messages": [manager_message],
            })

            logger.info("⚙️ Resuming suspended graph (async)...")
            final_state = await self.graph.ainvoke(None, config=config)
            response = final_state.get("final_response", "Error: Empty response.")
            return {"respuesta": response}

        except Exception as e:
            logger.error("❌ Error while resuming: %s", e, exc_info=True)
            raise
        finally:
            if langfuse_handler:
                try:
                    await asyncio.to_thread(LangFuseMonitor.flush)
                except Exception as flush_err:
                    logger.warning("⚠️ LangFuse flush failed: %s", flush_err)