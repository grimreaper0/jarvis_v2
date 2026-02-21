#!/usr/bin/env python3.13
"""Backfill Neo4j knowledge graph from existing PostgreSQL data.

Migrates: bot_accounts, bot_activity, conversations, confidence_audit_log,
          viral_opportunities, roadmap_items, roadmap_dependencies,
          personal_notes, patterns

Run: PYTHONPATH=. venv/bin/python scripts/backfill_neo4j.py
"""
import asyncio
import json
import structlog
from datetime import datetime
from typing import Any

log = structlog.get_logger()

PSQL = "/opt/homebrew/Cellar/postgresql@17/17.7_1/bin/psql"


async def backfill() -> None:
    import asyncpg
    from jarvis.core.knowledge_graph import KnowledgeGraph
    from config.settings import get_settings

    settings = get_settings()
    kg = KnowledgeGraph()
    await kg.connect()
    await kg.ensure_indexes()

    pool = await asyncpg.create_pool(settings.postgres_url)

    stats: dict[str, int] = {}

    # ── Bot accounts ──────────────────────────────────────────────────────
    log.info("backfill.bot_accounts")
    rows = await pool.fetch("SELECT * FROM bot_accounts")
    for r in rows:
        await kg.merge_bot(
            name=r["bot_name"],
            platform=r.get("platform"),
            status=r.get("status", "unknown"),
            pg_id=_safe(r["id"]),
        )
    stats["bot_accounts"] = len(rows)

    # ── Bot activity (batched — 21K+ rows) ────────────────────────────────
    # Schema: id(uuid), bot_name, account_id, activity_type, status, title,
    #         detail(jsonb), content_id, audit_id, revenue_amount,
    #         error_message, started_at, completed_at, created_at
    log.info("backfill.bot_activity")
    count = 0
    batch_size = 500
    offset = 0
    while True:
        rows = await pool.fetch(
            "SELECT id, bot_name, activity_type, status, created_at, "
            "       detail, title "
            "FROM bot_activity ORDER BY created_at LIMIT $1 OFFSET $2",
            batch_size, offset,
        )
        if not rows:
            break
        for r in rows:
            detail = r.get("detail")
            topics = _extract_topics(detail)
            props = {}
            if r.get("title"):
                props["title"] = r["title"]
            if detail and isinstance(detail, dict):
                for k in ("symbol", "strategy", "side", "platform", "source"):
                    if k in detail:
                        props[k] = str(detail[k])

            await kg.add_activity(
                pg_id=_safe(r["id"]),
                bot_name=r["bot_name"],
                activity_type=r["activity_type"],
                status=r["status"],
                created_at=_safe(r["created_at"]) or "",
                topics=topics,
                **props,
            )
            count += 1
        offset += batch_size
        if count % 2000 == 0:
            log.info("backfill.bot_activity.progress", count=count)
    stats["bot_activity"] = count

    # ── Conversations ─────────────────────────────────────────────────────
    # Schema: id, session_id, timestamp, summary, embedding, context, outcome,
    #         created_at, search_vector
    log.info("backfill.conversations")
    rows = await pool.fetch(
        "SELECT id, session_id, summary, context, outcome, created_at FROM conversations"
    )
    for r in rows:
        # Extract topics from summary text
        summary = r.get("summary") or ""
        title = summary[:100] if summary else "untitled"
        context = r.get("context")
        topics = []
        if context and isinstance(context, dict):
            topics = _extract_topics(context)
        await kg.add_conversation(
            pg_id=_safe(r["id"]),
            title=title,
            topics=topics,
            session_id=_safe(r.get("session_id")) or "",
            outcome=r.get("outcome") or "",
            created_at=_safe(r.get("created_at")) or "",
        )
    stats["conversations"] = len(rows)

    # ── Confidence audit log ──────────────────────────────────────────────
    # Schema: id, bot_name, task_type, base_score, validation_score,
    #         historical_score, reflexive_score, final_confidence, decision,
    #         golden_rules_applied, guardrail_flags, reasoning, context,
    #         outcome, revenue_impact, created_at, resolved_at
    log.info("backfill.confidence_audit_log")
    count = 0
    offset = 0
    while True:
        rows = await pool.fetch(
            "SELECT id, task_type, decision, final_confidence, base_score, "
            "       validation_score, historical_score, reflexive_score, "
            "       bot_name, created_at "
            "FROM confidence_audit_log ORDER BY created_at LIMIT $1 OFFSET $2",
            batch_size, offset,
        )
        if not rows:
            break
        for r in rows:
            await kg.add_confidence_decision(
                pg_id=_safe(r["id"]),
                task_type=r.get("task_type") or "unknown",
                decision=r.get("decision") or "unknown",
                final_score=float(r.get("final_confidence") or 0.0),
                base_score=float(r.get("base_score") or 0.0),
                validation_score=float(r.get("validation_score") or 0.0),
                historical_score=float(r.get("historical_score") or 0.0),
                reflexive_score=float(r.get("reflexive_score") or 0.0),
                bot_name=r.get("bot_name") or "",
                created_at=_safe(r.get("created_at")) or "",
            )
            count += 1
        offset += batch_size
    stats["confidence_audit_log"] = count

    # ── Viral opportunities ───────────────────────────────────────────────
    # Schema: id, tool_name, source, url, score, github_stars_24h, etc.
    log.info("backfill.viral_opportunities")
    rows = await pool.fetch(
        "SELECT id, tool_name, source, url, score, alert_tier, created_at "
        "FROM viral_opportunities"
    )
    for r in rows:
        await kg.add_viral_opportunity(
            pg_id=_safe(r["id"]),
            title=r.get("tool_name") or "untitled",
            platform=r.get("source") or "unknown",
            score=float(r.get("score") or 0.0),
            source_url=r.get("url") or "",
            alert_tier=r.get("alert_tier") or "",
            created_at=_safe(r.get("created_at")) or "",
        )
    stats["viral_opportunities"] = len(rows)

    # ── Roadmap items ─────────────────────────────────────────────────────
    log.info("backfill.roadmap_items")
    rows = await pool.fetch(
        "SELECT id, title, phase, status, priority, category, created_at "
        "FROM roadmap_items"
    )
    for r in rows:
        await kg.add_roadmap_item(
            pg_id=_safe(r["id"]),
            title=r.get("title") or "untitled",
            phase=r.get("phase") or "unknown",
            status=r.get("status") or "unknown",
            priority=r.get("priority") or "normal",
            category=r.get("category") or "",
            created_at=_safe(r.get("created_at")) or "",
        )
    stats["roadmap_items"] = len(rows)

    # ── Roadmap dependencies ──────────────────────────────────────────────
    log.info("backfill.roadmap_dependencies")
    rows = await pool.fetch(
        "SELECT blocker_id, blocked_id, dependency_type "
        "FROM roadmap_dependencies"
    )
    for r in rows:
        if r.get("dependency_type") == "blocks":
            await kg.link_roadmap_blocks(_safe(r["blocker_id"]), _safe(r["blocked_id"]))
    stats["roadmap_dependencies"] = len(rows)

    # ── Personal notes ────────────────────────────────────────────────────
    log.info("backfill.personal_notes")
    rows = await pool.fetch(
        "SELECT id, title, tags, priority, created_at FROM personal_notes"
    )
    for r in rows:
        tags = r.get("tags")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []
        topics = tags if isinstance(tags, list) else []
        await kg.add_personal_note(
            pg_id=_safe(r["id"]),
            title=r.get("title") or "untitled",
            topics=topics,
            priority=r.get("priority", "normal"),
            created_at=_safe(r.get("created_at")) or "",
        )
    stats["personal_notes"] = len(rows)

    # ── Patterns ──────────────────────────────────────────────────────────
    # Schema: id, pattern_type, description, embedding, confidence_score,
    #         discovered_at, source_conversation_id, metadata, created_at,
    #         is_golden_rule, success_count, total_uses, promoted_at, etc.
    log.info("backfill.patterns")
    rows = await pool.fetch(
        "SELECT id, pattern_type, description, confidence_score, "
        "       is_golden_rule, success_count, total_uses, "
        "       source_conversation_id, created_at "
        "FROM patterns"
    )
    for r in rows:
        await kg.add_pattern(
            pg_id=_safe(r["id"]),
            pattern_type=r.get("pattern_type") or "unknown",
            description=r.get("description") or "",
            confidence=float(r.get("confidence_score") or 0.0),
            is_golden_rule=r.get("is_golden_rule", False),
            success_count=r.get("success_count", 0),
            total_uses=r.get("total_uses", 0),
            created_at=_safe(r.get("created_at")) or "",
        )
        if r.get("source_conversation_id"):
            await kg.link_conversation_to_pattern(
                _safe(r["source_conversation_id"]), _safe(r["id"])
            )
    stats["patterns"] = len(rows)

    # ── Final stats ───────────────────────────────────────────────────────
    await pool.close()

    graph_stats = await kg.get_graph_stats()
    await kg.close()

    log.info("backfill.complete", pg_stats=stats, neo4j_stats=graph_stats)
    print("\n=== BACKFILL COMPLETE ===")
    print(f"PostgreSQL → Neo4j migration:")
    for table, count in sorted(stats.items()):
        print(f"  {table}: {count} rows")
    print(f"\nNeo4j graph:")
    for label, count in sorted(graph_stats["nodes"].items()):
        print(f"  {label}: {count} nodes")
    for rel_type, count in sorted(graph_stats["relationships"].items()):
        print(f"  {rel_type}: {count} relationships")


def _safe(val: Any) -> Any:
    """Convert asyncpg types to Neo4j-compatible Python types."""
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _safe_dict(row: dict) -> dict:
    """Convert all values in a dict to Neo4j-safe types."""
    return {k: _safe(v) for k, v in row.items() if v is not None}


def _extract_topics(details: Any) -> list[str]:
    """Extract topic keywords from activity details."""
    if not details:
        return []
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(details, dict):
        return []
    topics = []
    for key in ("topic", "symbol", "strategy", "platform", "source", "category"):
        val = details.get(key)
        if val and isinstance(val, str):
            topics.append(val.lower().strip())
    tags = details.get("tags") or details.get("hashtags")
    if isinstance(tags, list):
        topics.extend(t.lower().strip().lstrip("#") for t in tags if isinstance(t, str))
    return topics


if __name__ == "__main__":
    asyncio.run(backfill())
