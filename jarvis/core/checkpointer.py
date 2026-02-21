"""Shared LangGraph checkpointer — PostgresSaver backed by personal_agent_hub.

All Tier 2 graphs and Tier 3 agent subgraphs use this to persist checkpoint
state across server restarts. Replaces the per-graph MemorySaver instances
which were in-memory only (lost on restart).

Usage in a graph module:
    from jarvis.core.checkpointer import get_checkpointer

    async def build_my_graph():
        checkpointer = await get_checkpointer()
        return graph.compile(checkpointer=checkpointer)

The checkpointer lazily creates its table on first use (setup()).
Connection pool is shared across all graphs in the same process.
"""
from __future__ import annotations

import asyncio

import structlog

log = structlog.get_logger()

_checkpointer = None
_checkpointer_lock = asyncio.Lock()


async def get_checkpointer():
    """Return a shared AsyncPostgresSaver instance (singleton per process).

    Creates the checkpoint tables on first call via setup().
    Uses psycopg async connection pool under the hood.
    """
    global _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    async with _checkpointer_lock:
        if _checkpointer is not None:
            return _checkpointer

        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        from config.settings import get_settings

        settings = get_settings()

        # AsyncPostgresSaver expects a psycopg connection string (not SQLAlchemy)
        # Our postgres_url is already in the right format: postgresql://localhost/personal_agent_hub
        conn_string = settings.postgres_url

        saver = AsyncPostgresSaver.from_conn_string(conn_string)
        await saver.setup()

        _checkpointer = saver
        log.info("checkpointer.postgres_ready", url=conn_string.split("@")[-1] if "@" in conn_string else conn_string)
        return _checkpointer
