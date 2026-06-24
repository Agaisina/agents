import logging
import threading
from typing import Optional, Any
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from src.config import settings

logger = logging.getLogger(__name__)


class LangFuseMonitor:
    """
    Manages LangFuse integration for the agent system.
    Provides tracing and observability capabilities.
    """

    _instance: Optional["LangFuseMonitor"] = None
    _langfuse_client: Optional[Langfuse] = None
    _callback_handler: Any = None

    def __new__(cls):
        """Singleton pattern to ensure only one LangFuse client."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize LangFuse monitor (singleton)."""
        if self._langfuse_client is not None or self._callback_handler is not None:
            return

        if not settings.langfuse.get("enabled", False):
            logger.info("⊘ LangFuse monitoring is disabled")
            return

        try:
            public_key = settings.langfuse.get("public_key")
            secret_key = settings.langfuse.get("secret_key")

            if not public_key or not secret_key or str(public_key).strip() == "None":
                logger.warning(
                    "⚠️  LangFuse credentials not found or invalid. Set APP_LANGFUSE__PUBLIC_KEY and "
                    "APP_LANGFUSE__SECRET_KEY in Hugging Face or your .env file."
                )
                self._langfuse_client = None
                self._callback_handler = None
                return

            host = settings.langfuse.get("host", "https://cloud.langfuse.com")
            
            pk = str(public_key).strip()
            sk = str(secret_key).strip()
            h = str(host).strip()

            Langfuse(
                public_key=pk,
                secret_key=sk,
                host=h,
                flush_interval=10
            )

            self._langfuse_client = get_client()
            self._callback_handler = CallbackHandler()

            logger.info("✅ LangFuse monitoring initialized successfully")
            logger.info(f"   Host: {h}")
            logger.info(f"   Tag: {settings.langfuse.get('tag', 'agent-system')}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize LangFuse: {e}")
            self._langfuse_client = None
            self._callback_handler = None

    @classmethod
    def get_client(cls) -> Optional[Langfuse]:
        """Get the raw LangFuse client instance."""
        instance = cls()
        return instance._langfuse_client

    @classmethod
    def get_callback(cls) -> Any:
        """
        Returns a configured CallbackHandler to inject into LangGraph/LangChain.
        Returns None if credentials are not configured.
        """
        instance = cls()
        return instance._callback_handler

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if LangFuse monitoring is enabled and initialized."""
        return cls.get_client() is not None

    @classmethod
    def add_user_feedback(cls, trace_id: str, score: int, comment: str = ""):
        """
        Add user feedback to a trace for evaluation and improvement.
        """
        client = cls.get_client()
        if client is None:
            logger.warning("LangFuse not enabled, cannot add feedback")
            return

        try:
            client.score(
                trace_id=trace_id, 
                name="user_feedback", 
                value=score, 
                comment=comment
            )
            logger.info(f"✅ Feedback recorded for trace {trace_id}")
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")

    @classmethod
    def flush(cls, timeout: float = 5.0) -> None:
        """Flush pending traces to LangFuse.

        Runs the flush in a daemon thread with a hard timeout so a slow or
        unreachable LangFuse server never blocks the API response path.

        Args:
            timeout: Maximum seconds to wait.  Defaults to 5 s.
        """
        client = get_client()
        if client is None:
            return

        thread = threading.Thread(target=client.flush, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(
                "⚠️ LangFuse flush timed out after %.1fs — traces may be delayed", timeout
            )
