"""AutoMem — PostgreSQL + pgvector memory layer."""
import structlog
from datetime import datetime
from typing import Any
import psycopg2
import psycopg2.extras

log = structlog.get_logger()


class AutoMem:
    """Persistent memory via PostgreSQL + pgvector.

    Stores conversations, patterns, and notes with semantic search.
    Compatible with the existing personal_agent_hub schema.
    """

    def __init__(self, db_url: str | None = None):
        from config.settings import get_settings
        settings = get_settings()
        self.db_url = db_url or settings.postgres_url
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
            self._conn.autocommit = False
        return self._conn

    def add_conversation(self, role: str, content: str, embedding: list[float] | None = None) -> int:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (role, content, embedding, created_at)
                VALUES (%s, %s, %s, %s) RETURNING id
                """,
                (role, content, embedding, datetime.utcnow()),
            )
            row_id = cur.fetchone()[0]
        conn.commit()
        log.debug("conversation.stored", role=role, id=row_id)
        return row_id

    def search_conversations(self, query_embedding: list[float], limit: int = 5) -> list[dict]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, role, content, created_at,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM conversations
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, limit),
            )
            return [dict(r) for r in cur.fetchall()]

    def add_pattern(self, name: str, description: str, confidence: float, metadata: dict | None = None) -> int:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patterns (name, description, confidence, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
                """,
                (name, description, confidence, psycopg2.extras.Json(metadata or {}), datetime.utcnow()),
            )
            row_id = cur.fetchone()[0]
        conn.commit()
        log.info("pattern.stored", name=name, confidence=confidence)
        return row_id

    def get_patterns(self, min_confidence: float = 0.7) -> list[dict]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM patterns WHERE confidence >= %s ORDER BY confidence DESC",
                (min_confidence,),
            )
            return [dict(r) for r in cur.fetchall()]

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
