import sqlite3
import logging
from typing import Union
import psycopg
from src.config import settings

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = logging.getLogger(__name__)

class CheckpointFactory:
    """Decouples the Graph from the specific Database implementation."""
    @staticmethod
    def get_checkpointer() -> Union[SqliteSaver, PostgresSaver]:
        """Create and return the appropriate LangGraph checkpointer.

        Reads ``settings.database.type`` to determine the backend:

        - ``"sqlite"``: local file-based SQLite (default for development).
        - ``"postgres"``: Supabase / PostgreSQL (production).

        Returns:
            A ``SqliteSaver`` or ``PostgresSaver`` instance ready for use.

        Raises:
            ValueError: When ``postgres_uri`` is missing or the type is unknown.
        """
        db_type = settings.database.type.lower()

        if db_type == "sqlite":
            logger.info("🧠 Memory: Initializing local SQLite...")
            sqlite_path = settings.get("database.sqlite_path", "checkpoints.db")
            conn = sqlite3.connect(sqlite_path, check_same_thread=False)
            return SqliteSaver(conn)
            
        elif db_type == "postgres":
            logger.info("🧠 Memory: Connecting to Supabase PostgreSQL...")
            postgres_uri = settings.database.get("postgres_uri")
            if not postgres_uri or postgres_uri == "none":
                raise ValueError("'postgres_uri' not found. Add it to your settings.yaml or as an environment variable.")
            conn = psycopg.connect(postgres_uri, autocommit=True, prepare_threshold=None)
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()
            return checkpointer
            
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    @staticmethod
    async def aget_checkpointer() -> Union[AsyncSqliteSaver, AsyncPostgresSaver]:
        """Create and return an **async-compatible** LangGraph checkpointer.

        Required when using async graph methods (``ainvoke``, ``astream``,
        ``aget_state``, ``aupdate_state``). Uses the same ``settings.database.type``
        switch as :meth:`get_checkpointer`.

        Returns:
            An ``AsyncSqliteSaver`` or ``AsyncPostgresSaver`` instance.

        Raises:
            ValueError: When ``postgres_uri`` is missing or the type is unknown.
        """
        db_type = settings.database.type.lower()

        if db_type == "sqlite":
            import aiosqlite
            logger.info("🧠 Memory: Initializing async SQLite...")
            sqlite_path = settings.get("database.sqlite_path", "checkpoints.db")
            conn = await aiosqlite.connect(sqlite_path)
            return AsyncSqliteSaver(conn)

        elif db_type == "postgres":
            logger.info("🧠 Memory: Connecting to Supabase PostgreSQL (async)...")
            postgres_uri = settings.database.get("postgres_uri")
            if not postgres_uri or postgres_uri == "none":
                raise ValueError("'postgres_uri' not found. Add it to your settings.yaml or as an environment variable.")
            conn = await psycopg.AsyncConnection.connect(
                postgres_uri, autocommit=True, prepare_threshold=None
            )
            checkpointer = AsyncPostgresSaver(conn)
            await checkpointer.setup()
            return checkpointer

        else:
            raise ValueError(f"Unsupported database type: {db_type}")