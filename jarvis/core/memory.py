"""AutoMem v2 — PostgreSQL + pgvector memory layer (async-first).

Ported from jarvis_v1 utils/automem.py with asyncpg replacing psycopg2.
Compatible with the existing personal_agent_hub schema (14 tables).
Does NOT recreate tables — connects and uses the existing schema.

Connection: postgresql://localhost/personal_agent_hub
"""

import json
import uuid
import asyncio
from datetime import datetime
from typing import Any, Optional

import httpx
import structlog

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

async def get_embedding(text: str, base_url: str = "http://localhost:11434") -> list[float]:
    """Generate embedding via Ollama nomic-embed-text (768-dim)."""
    payload = {"model": "nomic-embed-text", "prompt": text}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{base_url}/api/embeddings", json=payload)
        resp.raise_for_status()
        return resp.json()["embedding"]


# ---------------------------------------------------------------------------
# AutoMem v2
# ---------------------------------------------------------------------------

class AutoMem:
    """PostgreSQL + pgvector backed conversation memory and pattern learning.

    Async-first: uses asyncpg connection pool.
    Compatible with the existing personal_agent_hub schema.
    """

    def __init__(self, db_url: str | None = None):
        from config.settings import get_settings
        settings = get_settings()
        self.db_url = db_url or settings.postgres_url
        self._pool = None
        self._pool_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _get_pool(self):
        """Return (or lazily create) the asyncpg connection pool."""
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is not None:
                return self._pool
            import asyncpg
            # asyncpg expects postgresql:// DSN; strip any trailing options
            dsn = self.db_url
            self._pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10)
            # Register pgvector codec for this pool so vector columns work
            async with self._pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            log.info("automem.pool_created", dsn=dsn)
        return self._pool

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def _execute(
        self,
        query: str,
        *args,
        fetch: str | None = None,
    ) -> Any:
        """Run a query against the pool. fetch: 'one' | 'all' | None."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                if fetch == "one":
                    return await conn.fetchrow(query, *args)
                elif fetch == "all":
                    return await conn.fetch(query, *args)
                else:
                    return await conn.execute(query, *args)
            except Exception as exc:
                log.error("automem.query_failed", error=str(exc), query=query[:120])
                raise

    # ------------------------------------------------------------------
    # Embedding (convenience wrapper)
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text via Ollama."""
        return await get_embedding(text)

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    async def store_conversation(
        self,
        session_id: str,
        summary: str,
        embedding: list[float],
        context: dict[str, Any],
        outcome: str = "general",
    ) -> str:
        """Store a conversation with embedding for semantic search.

        Returns the conversation UUID.
        """
        # asyncpg requires vector as a string representation for pgvector
        vec = f"[{','.join(str(v) for v in embedding)}]"
        row = await self._execute(
            """
            INSERT INTO conversations (session_id, summary, embedding, context, outcome)
            VALUES ($1, $2, $3::vector, $4::jsonb, $5)
            RETURNING id
            """,
            session_id, summary, vec, json.dumps(context), outcome,
            fetch="one",
        )
        conv_id = str(row["id"])
        log.info("conversation.stored", id=conv_id, session=session_id)
        return conv_id

    async def find_similar_conversations(
        self,
        embedding: list[float],
        limit: int = 5,
        min_similarity: float = 0.7,
        outcome_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find conversations semantically similar to the given embedding."""
        vec = f"[{','.join(str(v) for v in embedding)}]"
        if outcome_filter:
            rows = await self._execute(
                """
                SELECT id, session_id, summary, outcome, context,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM conversations
                WHERE outcome = $2
                  AND 1 - (embedding <=> $1::vector) >= $3
                ORDER BY embedding <=> $1::vector
                LIMIT $4
                """,
                vec, outcome_filter, min_similarity, limit,
                fetch="all",
            )
        else:
            rows = await self._execute(
                """
                SELECT id, session_id, summary, outcome, context,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM conversations
                WHERE 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                vec, min_similarity, limit,
                fetch="all",
            )
        results = [
            {
                "id": str(r["id"]),
                "session_id": r["session_id"],
                "summary": r["summary"],
                "outcome": r["outcome"],
                "context": json.loads(r["context"]) if isinstance(r["context"], str) else r["context"],
                "similarity": float(r["similarity"]),
            }
            for r in (rows or [])
        ]
        log.info("conversations.searched", count=len(results))
        return results

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    async def extract_pattern(
        self,
        conversation_id: str,
        pattern_type: str,
        description: str,
        confidence: float,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Extract a reusable pattern from a conversation. Returns pattern UUID."""
        vec = f"[{','.join(str(v) for v in embedding)}]"
        row = await self._execute(
            """
            INSERT INTO patterns
                (pattern_type, description, confidence_score, embedding,
                 source_conversation_id, metadata)
            VALUES ($1, $2, $3, $4::vector, $5, $6::jsonb)
            RETURNING id
            """,
            pattern_type, description, confidence, vec,
            uuid.UUID(conversation_id), json.dumps(metadata or {}),
            fetch="one",
        )
        pattern_id = str(row["id"])
        log.info("pattern.extracted", id=pattern_id, type=pattern_type, confidence=confidence)
        return pattern_id

    async def find_similar_patterns(
        self,
        embedding: list[float],
        pattern_type: str | None = None,
        min_confidence: float = 0.5,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find patterns similar to embedding, sorted by relevance * similarity."""
        vec = f"[{','.join(str(v) for v in embedding)}]"
        if pattern_type:
            rows = await self._execute(
                """
                SELECT id, pattern_type, description, confidence_score, relevance_score,
                       is_golden_rule, last_used_at, success_count, total_uses, metadata,
                       1 - (embedding <=> $1::vector) AS similarity,
                       (1 - (embedding <=> $1::vector)) * relevance_score AS weighted_score
                FROM patterns
                WHERE confidence_score >= $2
                  AND pattern_type = $3
                ORDER BY weighted_score DESC
                LIMIT $4
                """,
                vec, min_confidence, pattern_type, limit,
                fetch="all",
            )
        else:
            rows = await self._execute(
                """
                SELECT id, pattern_type, description, confidence_score, relevance_score,
                       is_golden_rule, last_used_at, success_count, total_uses, metadata,
                       1 - (embedding <=> $1::vector) AS similarity,
                       (1 - (embedding <=> $1::vector)) * relevance_score AS weighted_score
                FROM patterns
                WHERE confidence_score >= $2
                ORDER BY weighted_score DESC
                LIMIT $3
                """,
                vec, min_confidence, limit,
                fetch="all",
            )
        return [_pattern_row_to_dict(r) for r in (rows or [])]

    async def hybrid_search_patterns(
        self,
        query: str,
        embedding: list[float],
        pattern_type: str | None = None,
        min_confidence: float = 0.7,
        limit: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Hybrid search: pgvector cosine + BM25 full-text (5-10% better retrieval)."""
        vec = f"[{','.join(str(v) for v in embedding)}]"
        type_filter = "AND pattern_type = $5" if pattern_type else ""
        params: list[Any] = [vec, query, min_confidence, limit]
        if pattern_type:
            params.append(pattern_type)

        sql = f"""
        WITH vector_scores AS (
            SELECT id,
                   1 - (embedding <=> $1::vector) AS vector_score
            FROM patterns
            WHERE confidence_score >= $3
            {type_filter}
        ),
        keyword_scores AS (
            SELECT id,
                   ts_rank(to_tsvector('english', description),
                           plainto_tsquery('english', $2)) AS keyword_score
            FROM patterns
            WHERE confidence_score >= $3
            {type_filter}
        )
        SELECT p.id, p.pattern_type, p.description, p.confidence_score,
               p.relevance_score, p.is_golden_rule, p.last_used_at,
               p.success_count, p.total_uses, p.metadata,
               COALESCE(vs.vector_score, 0) AS vector_score,
               COALESCE(ks.keyword_score, 0) AS keyword_score,
               ({vector_weight} * COALESCE(vs.vector_score, 0) +
                {keyword_weight} * COALESCE(ks.keyword_score, 0)) AS hybrid_score
        FROM patterns p
        LEFT JOIN vector_scores vs ON p.id = vs.id
        LEFT JOIN keyword_scores ks ON p.id = ks.id
        WHERE p.confidence_score >= $3
        {type_filter}
        ORDER BY hybrid_score DESC
        LIMIT $4
        """
        rows = await self._execute(sql, *params, fetch="all")
        results = []
        for r in (rows or []):
            d = _pattern_row_to_dict(r)
            d["vector_score"] = float(r.get("vector_score") or 0)
            d["keyword_score"] = float(r.get("keyword_score") or 0)
            d["hybrid_score"] = float(r.get("hybrid_score") or 0)
            results.append(d)
        return results

    async def get_golden_rules_for_task(
        self,
        task_embedding: list[float],
        task_type: str | None = None,
        min_similarity: float = 0.6,
    ) -> list[dict[str, Any]]:
        """Find golden rules relevant to a task via semantic search."""
        vec = f"[{','.join(str(v) for v in task_embedding)}]"
        if task_type:
            rows = await self._execute(
                """
                SELECT id, pattern_type, description, confidence_score,
                       success_count, total_uses, metadata,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM patterns
                WHERE is_golden_rule = TRUE
                  AND 1 - (embedding <=> $1::vector) >= $2
                  AND pattern_type = $3
                ORDER BY embedding <=> $1::vector
                LIMIT 10
                """,
                vec, min_similarity, task_type,
                fetch="all",
            )
        else:
            rows = await self._execute(
                """
                SELECT id, pattern_type, description, confidence_score,
                       success_count, total_uses, metadata,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM patterns
                WHERE is_golden_rule = TRUE
                  AND 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT 10
                """,
                vec, min_similarity,
                fetch="all",
            )
        return [_pattern_row_to_dict(r) for r in (rows or [])]

    async def get_golden_rules(
        self,
        pattern_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Load active golden rules."""
        if pattern_type:
            rows = await self._execute(
                """
                SELECT id, pattern_type, description, confidence_score,
                       success_count, total_uses, metadata, promoted_at, last_used_at,
                       CASE WHEN total_uses > 0
                            THEN (success_count::numeric / total_uses::numeric)
                            ELSE 0 END AS success_rate
                FROM patterns
                WHERE is_golden_rule = TRUE AND pattern_type = $1
                ORDER BY success_rate DESC, total_uses DESC
                LIMIT $2
                """,
                pattern_type, limit,
                fetch="all",
            )
        else:
            rows = await self._execute(
                """
                SELECT id, pattern_type, description, confidence_score,
                       success_count, total_uses, metadata, promoted_at, last_used_at,
                       CASE WHEN total_uses > 0
                            THEN (success_count::numeric / total_uses::numeric)
                            ELSE 0 END AS success_rate
                FROM patterns
                WHERE is_golden_rule = TRUE
                ORDER BY success_rate DESC, total_uses DESC
                LIMIT $1
                """,
                limit,
                fetch="all",
            )
        log.info("golden_rules.loaded", count=len(rows or []))
        return [_pattern_row_to_dict(r) for r in (rows or [])]

    async def promote_to_golden_rule(self, pattern_id: str) -> bool:
        """Promote a pattern to golden rule if eligible (5+ uses, 90%+ success)."""
        row = await self._execute(
            "SELECT success_count, total_uses, is_golden_rule FROM patterns WHERE id = $1",
            uuid.UUID(pattern_id),
            fetch="one",
        )
        if not row:
            return False
        if row["is_golden_rule"]:
            return False
        total = int(row["total_uses"])
        successes = int(row["success_count"])
        if total < 5:
            return False
        if successes / total < 0.90:
            return False
        await self._execute(
            "UPDATE patterns SET is_golden_rule = TRUE, promoted_at = now() WHERE id = $1",
            uuid.UUID(pattern_id),
        )
        log.info("pattern.promoted_golden_rule", id=pattern_id)
        return True

    async def record_pattern_use(self, pattern_id: str, success: bool) -> dict[str, Any]:
        """Increment pattern use counters and check for golden rule promotion."""
        success_inc = 1 if success else 0
        row = await self._execute(
            """
            UPDATE patterns
            SET total_uses = total_uses + 1,
                success_count = success_count + $1,
                last_used_at = now()
            WHERE id = $2
            RETURNING total_uses, success_count, is_golden_rule, relevance_score
            """,
            success_inc, uuid.UUID(pattern_id),
            fetch="one",
        )
        if not row:
            return {"updated": False}
        promoted = False
        if not row["is_golden_rule"]:
            promoted = await self.promote_to_golden_rule(pattern_id)
        return {
            "updated": True,
            "total_uses": int(row["total_uses"]),
            "success_count": int(row["success_count"]),
            "relevance_score": float(row["relevance_score"]),
            "promoted": promoted,
        }

    # ------------------------------------------------------------------
    # Revenue tracking
    # ------------------------------------------------------------------

    async def track_revenue(
        self,
        strategy_id: str,
        amount: float,
        metadata: dict[str, Any] | None = None,
        date: datetime | None = None,
    ) -> str:
        """Record a revenue event for a strategy. Returns revenue UUID."""
        revenue_date = (date or datetime.now()).date()
        row = await self._execute(
            """
            INSERT INTO revenue_results (strategy_id, revenue_amount, date, metadata)
            VALUES ($1, $2, $3, $4::jsonb)
            RETURNING id
            """,
            uuid.UUID(strategy_id), amount, revenue_date, json.dumps(metadata or {}),
            fetch="one",
        )
        revenue_id = str(row["id"])
        log.info("revenue.tracked", id=revenue_id, amount=amount)
        return revenue_id

    async def find_profitable_patterns(
        self,
        min_revenue: float = 100.0,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Find patterns that generated revenue above threshold in the last N days."""
        rows = await self._execute(
            """
            SELECT p.id, p.pattern_type, p.description, p.confidence_score,
                   COUNT(DISTINCT r.id) AS revenue_events,
                   SUM(r.revenue_amount) AS total_revenue,
                   AVG(r.revenue_amount) AS avg_revenue,
                   MAX(r.date) AS last_revenue_date
            FROM patterns p
            JOIN bot_strategies bs ON bs.pattern_id = p.id
            JOIN revenue_results r ON r.strategy_id = bs.id
            WHERE r.date >= CURRENT_DATE - $1 * INTERVAL '1 day'
            GROUP BY p.id, p.pattern_type, p.description, p.confidence_score
            HAVING SUM(r.revenue_amount) >= $2
            ORDER BY total_revenue DESC
            """,
            days, min_revenue,
            fetch="all",
        )
        return [
            {
                "id": str(r["id"]),
                "pattern_type": r["pattern_type"],
                "description": r["description"],
                "confidence_score": float(r["confidence_score"]),
                "revenue_events": int(r["revenue_events"]),
                "total_revenue": float(r["total_revenue"]),
                "avg_revenue": float(r["avg_revenue"]),
                "last_revenue_date": r["last_revenue_date"],
            }
            for r in (rows or [])
        ]

    # ------------------------------------------------------------------
    # Confidence audit
    # ------------------------------------------------------------------

    async def log_confidence_decision(
        self,
        bot_name: str,
        task_type: str,
        base_score: float,
        validation_score: float,
        historical_score: float,
        reflexive_score: float,
        final_confidence: float,
        decision: str,
        golden_rules_applied: list[str] | None = None,
        guardrail_flags: list[str] | None = None,
        reasoning: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Log a confidence gate decision to the audit trail. Returns audit UUID."""
        row = await self._execute(
            """
            INSERT INTO confidence_audit_log
                (bot_name, task_type, base_score, validation_score, historical_score,
                 reflexive_score, final_confidence, decision, golden_rules_applied,
                 guardrail_flags, reasoning, context)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10::jsonb,$11,$12::jsonb)
            RETURNING id
            """,
            bot_name, task_type, base_score, validation_score, historical_score,
            reflexive_score, final_confidence, decision,
            json.dumps(golden_rules_applied or []),
            json.dumps(guardrail_flags or []),
            reasoning,
            json.dumps(context or {}),
            fetch="one",
        )
        audit_id = str(row["id"])
        log.info("confidence.logged", id=audit_id, decision=decision, final=final_confidence)
        return audit_id

    async def update_audit_outcome(
        self,
        audit_id: str,
        outcome: str,
        revenue_impact: float | None = None,
    ) -> bool:
        """Close the feedback loop on a confidence audit entry."""
        await self._execute(
            """
            UPDATE confidence_audit_log
            SET outcome = $1, revenue_impact = $2, resolved_at = now()
            WHERE id = $3
            """,
            outcome, revenue_impact, uuid.UUID(audit_id),
        )
        log.info("audit.outcome_updated", id=audit_id, outcome=outcome)
        return True

    # ------------------------------------------------------------------
    # Personal notes
    # ------------------------------------------------------------------

    async def add_note(
        self,
        title: str,
        content: str = "",
        priority: str = "normal",
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Add a personal note with embedding for semantic search. Returns note UUID."""
        full_text = f"{title}\n\n{content}" if content else title
        embedding = await self.embed(full_text)
        vec = f"[{','.join(str(v) for v in embedding)}]"
        row = await self._execute(
            """
            INSERT INTO personal_notes (title, content, embedding, priority, tags, context)
            VALUES ($1, $2, $3::vector, $4, $5, $6::jsonb)
            RETURNING id
            """,
            title, content, vec, priority, tags or [], json.dumps(context or {}),
            fetch="one",
        )
        note_id = str(row["id"])
        log.info("note.added", id=note_id, title=title)
        return note_id

    async def search_notes(
        self,
        query: str,
        min_similarity: float = 0.5,
        status: str = "open",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Semantic search across personal notes."""
        embedding = await self.embed(query)
        vec = f"[{','.join(str(v) for v in embedding)}]"
        rows = await self._execute(
            """
            SELECT id, title, content, priority, status, tags, context, created_at,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM personal_notes
            WHERE status = $2
              AND 1 - (embedding <=> $1::vector) >= $3
            ORDER BY embedding <=> $1::vector
            LIMIT $4
            """,
            vec, status, min_similarity, limit,
            fetch="all",
        )
        return [
            {
                "id": str(r["id"]),
                "title": r["title"],
                "content": r["content"],
                "priority": r["priority"],
                "status": r["status"],
                "tags": list(r["tags"] or []),
                "context": r["context"],
                "created_at": r["created_at"],
                "similarity": float(r["similarity"]),
            }
            for r in (rows or [])
        ]

    async def get_notes(
        self,
        status: str = "open",
        priority: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get notes filtered by status and priority."""
        if priority:
            rows = await self._execute(
                """
                SELECT id, title, content, priority, status, tags, context, created_at
                FROM personal_notes
                WHERE status = $1 AND priority = $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                status, priority, limit,
                fetch="all",
            )
        else:
            rows = await self._execute(
                """
                SELECT id, title, content, priority, status, tags, context, created_at
                FROM personal_notes
                WHERE status = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                status, limit,
                fetch="all",
            )
        return [
            {
                "id": str(r["id"]),
                "title": r["title"],
                "content": r["content"],
                "priority": r["priority"],
                "status": r["status"],
                "tags": list(r["tags"] or []),
                "context": r["context"],
                "created_at": r["created_at"],
            }
            for r in (rows or [])
        ]

    async def update_note_status(
        self,
        note_id: str,
        status: str,
        completion_notes: str | None = None,
    ) -> bool:
        """Update note status."""
        if status == "completed":
            await self._execute(
                "UPDATE personal_notes SET status = $1, completed_at = now() WHERE id = $2",
                status, uuid.UUID(note_id),
            )
        else:
            await self._execute(
                "UPDATE personal_notes SET status = $1 WHERE id = $2",
                status, uuid.UUID(note_id),
            )
        log.info("note.status_updated", id=note_id, status=status)
        return True

    # ------------------------------------------------------------------
    # Bot operations
    # ------------------------------------------------------------------

    async def register_bot_account(
        self,
        bot_name: str,
        bot_type: str = "specialist",
        platform: str | None = None,
        account_handle: str | None = None,
        credential_ref: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Register or upsert a bot account. Returns account UUID."""
        row = await self._execute(
            """
            INSERT INTO bot_accounts
                (bot_name, bot_type, platform, account_handle, credential_ref, config)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            ON CONFLICT (bot_name, platform, account_handle)
            DO UPDATE SET config = COALESCE(EXCLUDED.config, bot_accounts.config),
                          updated_at = now()
            RETURNING id
            """,
            bot_name, bot_type, platform, account_handle,
            credential_ref, json.dumps(config or {}),
            fetch="one",
        )
        account_id = str(row["id"])
        log.info("bot_account.registered", id=account_id, bot=bot_name)
        return account_id

    async def get_analytics_summary(self, days: int = 30) -> dict[str, Any]:
        """High-level analytics summary."""
        conv = await self._execute(
            "SELECT COUNT(*) AS count FROM conversations WHERE timestamp >= CURRENT_DATE - $1 * INTERVAL '1 day'",
            days, fetch="one",
        )
        patt = await self._execute(
            "SELECT COUNT(*) AS count FROM patterns WHERE discovered_at >= CURRENT_DATE - $1 * INTERVAL '1 day'",
            days, fetch="one",
        )
        rev = await self._execute(
            """
            SELECT COUNT(*) AS events,
                   COALESCE(SUM(revenue_amount), 0) AS total,
                   COALESCE(AVG(revenue_amount), 0) AS avg
            FROM revenue_results
            WHERE date >= CURRENT_DATE - $1 * INTERVAL '1 day'
            """,
            days, fetch="one",
        )
        return {
            "period_days": days,
            "conversations": int(conv["count"]) if conv else 0,
            "patterns": int(patt["count"]) if patt else 0,
            "revenue_events": int(rev["events"]) if rev else 0,
            "total_revenue": float(rev["total"]) if rev else 0.0,
            "avg_revenue_per_event": float(rev["avg"]) if rev else 0.0,
        }

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------

    async def get_active_guardrails(
        self,
        guardrail_type: str | None = None,
        bot_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load active guardrail configurations."""
        conditions = ["is_active = TRUE"]
        params: list[Any] = []
        idx = 1

        if guardrail_type:
            idx += 1
            conditions.append(f"guardrail_type = ${idx}")
            params.append(guardrail_type)
        if bot_name:
            idx += 1
            conditions.append(f"(bot_name = ${idx} OR bot_name IS NULL)")
            params.append(bot_name)

        where = " AND ".join(conditions)
        rows = await self._execute(
            f"SELECT id, name, guardrail_type, description, config, bot_name FROM guardrails WHERE {where} ORDER BY guardrail_type, name",
            *params,
            fetch="all",
        )
        return [
            {
                "id": str(r["id"]),
                "name": r["name"],
                "guardrail_type": r["guardrail_type"],
                "description": r["description"],
                "config": r["config"],
                "bot_name": r["bot_name"],
            }
            for r in (rows or [])
        ]

    # ------------------------------------------------------------------
    # Pattern search (convenience)
    # ------------------------------------------------------------------

    async def search_patterns(
        self,
        context: str,
        limit: int = 10,
        min_confidence: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Semantic search for patterns related to a context string."""
        try:
            embedding = await self.embed(context)
        except Exception as exc:
            log.warning("search_patterns.embed_failed", error=str(exc))
            return []
        return await self.find_similar_patterns(
            embedding=embedding,
            min_confidence=min_confidence,
            limit=limit,
        )

    async def get_patterns_by_context(
        self,
        context: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get patterns by exact pattern_type match."""
        rows = await self._execute(
            """
            SELECT id, pattern_type, description, confidence_score,
                   success_count, total_uses, is_golden_rule, last_used_at, metadata
            FROM patterns
            WHERE pattern_type = $1
            ORDER BY confidence_score DESC, total_uses DESC
            LIMIT $2
            """,
            context, limit,
            fetch="all",
        )
        return [_pattern_row_to_dict(r) for r in (rows or [])]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pattern_row_to_dict(r: Any) -> dict[str, Any]:
    """Convert an asyncpg Record for a patterns row into a plain dict."""
    d: dict[str, Any] = {
        "id": str(r["id"]),
        "pattern_type": r["pattern_type"],
        "description": r["description"],
        "confidence_score": float(r["confidence_score"]) if r["confidence_score"] is not None else 0.0,
        "is_golden_rule": bool(r["is_golden_rule"]) if "is_golden_rule" in r.keys() else False,
        "metadata": r["metadata"] if "metadata" in r.keys() else {},
    }
    for optional in ("relevance_score", "success_count", "total_uses",
                     "last_used_at", "success_rate", "similarity",
                     "weighted_score", "promoted_at"):
        if optional in r.keys():
            val = r[optional]
            if val is not None and isinstance(val, (int, float)):
                d[optional] = float(val)
            else:
                d[optional] = val
    return d
