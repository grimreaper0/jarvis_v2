"""Confidence evaluation subgraph (Tier 2 LangGraph).

Standalone 4-layer scoring as a reusable graph.  Each layer runs as an
individual async node so scores are checkpointed independently, giving
a full audit trail per layer.

Layers:
    score_base       — task clarity (recognized type, description, context richness)
    score_validation — domain validation heuristics
    score_historical — AutoMem pattern search + golden rule boosts
    score_reflexive  — coherence check across first three layers
    finalize         — weighted combination → decision

Thresholds (loaded from Settings):
    >= confidence_execute  → "execute"
    >= confidence_delegate → "delegate"
    <  confidence_delegate → "clarify"
"""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from jarvis.core.state import ConfidenceState

log = structlog.get_logger()

# Layer weights — must sum to 1.0
WEIGHT_BASE: float = 0.40
WEIGHT_VALIDATION: float = 0.20
WEIGHT_HISTORICAL: float = 0.30
WEIGHT_REFLEXIVE: float = 0.10

# Baseline for novel tasks with no historical match
HISTORICAL_BASELINE: float = 0.30

# Known task types for base scoring
KNOWN_TASK_TYPES: frozenset[str] = frozenset(
    {
        "content_post", "content_story", "content_reel", "content_video",
        "trade_stock", "trade_crypto", "hashtag_research", "seo_optimization",
        "trend_analysis", "revenue_analysis", "pattern_extraction",
        "backtest", "portfolio_rebalance", "newsletter", "newsletter_issue",
        "tiktok_video", "tiktok_trend",
    }
)


# ─────────────────────────── Node implementations ────────────────────────────


async def score_base(state: ConfidenceState) -> dict[str, Any]:
    """
    Layer 1 — Base confidence from task clarity.

    Scoring:
      Recognised task type (in KNOWN_TASK_TYPES)  → +0.35
      Substantive description (20+ chars)          → +0.35
      Non-empty context dict                       → +0.30
    """
    # Pull task meta from messages or state kwargs
    task_meta: dict[str, Any] = _extract_task_meta(state)

    task_type: str = task_meta.get("task_type", "")
    description: str = task_meta.get("description", "")
    context: dict[str, Any] = task_meta.get("context", {})

    score = 0.0
    if task_type in KNOWN_TASK_TYPES:
        score += 0.35
    if len(description.strip()) >= 20:
        score += 0.35
    if context and len(context) > 0:
        score += 0.30

    base = round(min(1.0, score), 4)
    log.debug("confidence.score_base", task_type=task_type, score=base)
    return {"base_score": base}


async def score_validation(state: ConfidenceState) -> dict[str, Any]:
    """
    Layer 2 — Validation score from domain checks.

    Default heuristics (no custom callables in graph mode):
      Baseline                               → 0.50
      api_available in context               → +0.20
      data_fresh in context                  → +0.15
      within_limits is False                 → -0.30
    """
    task_meta: dict[str, Any] = _extract_task_meta(state)
    context: dict[str, Any] = task_meta.get("context", {})

    score = 0.50  # baseline
    if context.get("api_available", False):
        score += 0.20
    if context.get("data_fresh", False):
        score += 0.15
    if not context.get("within_limits", True):
        score -= 0.30

    validation = round(max(0.0, min(1.0, score)), 4)
    log.debug("confidence.score_validation", score=validation)
    return {"validation_score": validation}


async def score_historical(state: ConfidenceState) -> dict[str, Any]:
    """
    Layer 3 — Historical score from AutoMem pattern matching.

    Starts at HISTORICAL_BASELINE (0.30) for novel tasks.
    Boosted by matching patterns (up to +0.20) and golden rules (+0.15 each,
    capped at +0.30 total).
    """
    task_meta: dict[str, Any] = _extract_task_meta(state)
    task_type: str = task_meta.get("task_type", "")
    description: str = task_meta.get("description", "")

    score = HISTORICAL_BASELINE

    if not task_type and not description:
        log.debug("confidence.score_historical.no_task_meta", score=score)
        return {"historical_score": round(score, 4)}

    try:
        from jarvis.core.memory import AutoMem

        mem = AutoMem()
        # v2 AutoMem uses get_patterns_by_context for exact type matches
        # (find_similar_patterns requires a pre-computed embedding vector)
        type_matches = await mem.get_patterns_by_context(context=task_type, limit=10)

        # Keyword filter on description for secondary matching
        desc_keywords = set(description.lower().split()) if description else set()
        desc_matches = [
            p for p in type_matches
            if desc_keywords and any(
                kw in p.get("description", "").lower()
                for kw in desc_keywords
                if len(kw) > 4
            )
        ]

        # Union: start with type matches, enrich with desc filter
        combined = type_matches  # type matches already deduplicated by DB

        if combined:
            # Pattern boost: +0.05 per match, cap at +0.20
            pattern_boost = min(0.20, 0.05 * len(combined))
            score += pattern_boost

            # Golden rule boost: +0.15 per golden rule, cap at +0.30
            golden_count = sum(1 for p in combined if p.get("is_golden_rule", False))
            if golden_count:
                golden_boost = min(0.30, 0.15 * golden_count)
                score += golden_boost

            log.info(
                "confidence.score_historical.boosted",
                task_type=task_type,
                matched_patterns=len(combined),
                golden_rules=golden_count,
                desc_keyword_hits=len(desc_matches),
                score=round(score, 4),
            )

    except Exception as exc:
        log.warning("confidence.score_historical.failed", error=str(exc))

    historical = round(min(1.0, score), 4)
    return {"historical_score": historical}


async def score_reflexive(state: ConfidenceState) -> dict[str, Any]:
    """
    Layer 4 — Reflexive coherence check.

    Computes spread of the first three layer scores:
      spread <= 0.15  → high agreement  → avg + 0.10 bonus
      spread <= 0.30  → moderate        → avg (no change)
      spread >  0.30  → high divergence → avg - 0.15 penalty
    """
    base: float = state.get("base_score", 0.5)
    validation: float = state.get("validation_score", 0.5)
    historical: float = state.get("historical_score", 0.5)

    scores = [base, validation, historical]
    spread = max(scores) - min(scores)
    avg = sum(scores) / len(scores)

    if spread <= 0.15:
        reflexive = min(1.0, avg + 0.10)
    elif spread <= 0.30:
        reflexive = avg
    else:
        reflexive = max(0.0, avg - 0.15)

    reflexive = round(reflexive, 4)
    log.debug(
        "confidence.score_reflexive",
        spread=round(spread, 4),
        avg=round(avg, 4),
        reflexive=reflexive,
    )
    return {"reflexive_score": reflexive}


async def finalize(state: ConfidenceState) -> dict[str, Any]:
    """
    Weighted combination of all four layers → final score → decision.

    Audits result to confidence_audit_log via ConfidenceGate.
    """
    base: float = state.get("base_score", 0.5)
    validation: float = state.get("validation_score", 0.5)
    historical: float = state.get("historical_score", 0.5)
    reflexive: float = state.get("reflexive_score", 0.5)

    final = (
        base * WEIGHT_BASE
        + validation * WEIGHT_VALIDATION
        + historical * WEIGHT_HISTORICAL
        + reflexive * WEIGHT_REFLEXIVE
    )
    final = round(max(0.0, min(1.0, final)), 4)

    from config.settings import get_settings

    settings = get_settings()

    if final >= settings.confidence_execute:
        decision = "execute"
    elif final >= settings.confidence_delegate:
        decision = "delegate"
    else:
        decision = "clarify"

    log.info(
        "confidence.finalized",
        base=base,
        validation=validation,
        historical=historical,
        reflexive=reflexive,
        final=final,
        decision=decision,
    )

    # Audit via v2 ConfidenceGate (no bot_name arg) — non-fatal
    try:
        from jarvis.core.confidence import ConfidenceGate

        gate = ConfidenceGate()
        task_meta = _extract_task_meta(state)
        gate.evaluate(
            base=base,
            validation=validation,
            historical=historical,
            reflexive=reflexive,
            context=f"confidence_graph:{task_meta.get('task_type', 'unknown')}",
        )
    except Exception as exc:
        log.warning("confidence.finalize.audit_failed", error=str(exc))

    return {"final_score": final, "decision": decision}


async def clarify_node(state: ConfidenceState) -> dict[str, Any]:
    """Human-in-the-loop: pause execution and surface decision to operator.

    Called only when confidence < confidence_delegate threshold (<0.60).
    Graph execution is frozen here until the caller resumes via:

        graph.ainvoke(Command(resume=user_choice), config=thread_config)

    The resume value becomes `user_decision` and overrides `decision` so
    downstream code sees what the operator actually chose.
    """
    task_meta = _extract_task_meta(state)
    task_type = task_meta.get("task_type", "unknown")
    final_score = state.get("final_score", 0.0)

    # Pause graph — caller receives this dict via get_state().tasks[0].interrupts[0].value
    user_choice = interrupt({
        "question": (
            f"Confidence too low ({final_score:.0%}) to act autonomously on '{task_type}'. "
            f"What should I do?"
        ),
        "options": ["execute", "delegate", "skip", "abort"],
        "context": {
            "task_type": task_type,
            "confidence": final_score,
            "layers": {
                "base": state.get("base_score"),
                "validation": state.get("validation_score"),
                "historical": state.get("historical_score"),
                "reflexive": state.get("reflexive_score"),
            },
        },
    })

    log.info(
        "confidence.clarify.resumed",
        user_choice=user_choice,
        task_type=task_type,
    )
    return {"decision": user_choice, "user_decision": user_choice}


def _route_after_finalize(state: ConfidenceState) -> str:
    """Conditional edge from finalize: execute/delegate go to END, clarify pauses."""
    decision = state.get("decision", "clarify")
    if decision in ("execute", "delegate"):
        return decision
    return "clarify"


# ─────────────────────────── Helpers ─────────────────────────────────────────


def _extract_task_meta(state: ConfidenceState) -> dict[str, Any]:
    """
    Pull task metadata from the last HumanMessage in state['messages'] if present,
    otherwise fall back to state-level keys injected by the caller.
    """
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        # LangChain messages expose .content; also handle plain dicts
        content = getattr(last, "content", None) or (
            last.get("content") if isinstance(last, dict) else ""
        )
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

    # Fall back: caller may have embedded task_type etc. at top-level state keys
    return {
        "task_type": state.get("decision", ""),  # re-used field; default empty
        "description": "",
        "context": {},
    }


# ─────────────────────────── Graph builder ───────────────────────────────────


def build_confidence_graph():
    """Build and compile the 4-layer confidence evaluation subgraph.

    Flow:
        score_base → score_validation → score_historical → score_reflexive → finalize
            ├── execute  → END  (confidence >= 0.90)
            ├── delegate → END  (confidence 0.60-0.89)
            └── clarify  → clarify_node → END  (confidence < 0.60, interrupt() called)
    """
    graph = StateGraph(ConfidenceState)

    graph.add_node("score_base", score_base)
    graph.add_node("score_validation", score_validation)
    graph.add_node("score_historical", score_historical)
    graph.add_node("score_reflexive", score_reflexive)
    graph.add_node("finalize", finalize)
    graph.add_node("clarify_node", clarify_node)

    graph.set_entry_point("score_base")
    graph.add_edge("score_base", "score_validation")
    graph.add_edge("score_validation", "score_historical")
    graph.add_edge("score_historical", "score_reflexive")
    graph.add_edge("score_reflexive", "finalize")

    graph.add_conditional_edges(
        "finalize",
        _route_after_finalize,
        {
            "execute": END,
            "delegate": END,
            "clarify": "clarify_node",
        },
    )
    graph.add_edge("clarify_node", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


confidence_graph = build_confidence_graph()


# ─────────────────────────── Graph Runner ────────────────────────────────────


class ConfidenceGraphRunner:
    """Consume from 'confidence_eval' Redis queue and run the confidence subgraph."""

    QUEUE = "confidence_eval"

    def __init__(self) -> None:
        self.graph = build_confidence_graph()
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis

            from config.settings import get_settings

            settings = get_settings()
            self._redis = aioredis.from_url(settings.redis_url)
        return self._redis

    async def run_forever(self) -> None:
        """Consume from Redis queue and run confidence graph for each request indefinitely."""
        log.info("confidence_runner.starting", queue=self.QUEUE)
        r = await self._get_redis()

        while True:
            try:
                item = await r.blpop(self.QUEUE, timeout=5)
                if item is None:
                    continue

                _, raw = item
                try:
                    task: dict[str, Any] = json.loads(raw)
                except json.JSONDecodeError as exc:
                    log.error("confidence_runner.bad_json", error=str(exc))
                    continue

                thread_id: str = task.get("id") or str(uuid4())
                config = {"configurable": {"thread_id": thread_id}}

                # Embed task meta as a human message for _extract_task_meta
                from langchain_core.messages import HumanMessage

                task_message = HumanMessage(content=json.dumps(task))

                initial_state: ConfidenceState = {
                    "messages": [task_message],
                    "confidence": 0.0,
                    "error": None,
                    "base_score": 0.0,
                    "validation_score": 0.0,
                    "historical_score": 0.0,
                    "reflexive_score": 0.0,
                    "final_score": 0.0,
                    "decision": "clarify",
                    "user_decision": None,
                }

                try:
                    result = await self.graph.ainvoke(initial_state, config=config)

                    # Check if graph was interrupted (confidence too low, needs human input)
                    snapshot = self.graph.get_state(config)
                    if snapshot.next:
                        interrupt_val: dict = {}
                        for task_info in snapshot.tasks:
                            if task_info.interrupts:
                                interrupt_val = task_info.interrupts[0].value
                                break
                        log.warning(
                            "confidence_runner.interrupted",
                            thread_id=thread_id,
                            question=interrupt_val.get("question"),
                        )
                        reply_queue: str | None = task.get("reply_queue")
                        if reply_queue:
                            reply = json.dumps({
                                "id": thread_id,
                                "status": "waiting_for_input",
                                "interrupt": interrupt_val,
                                "correlation_id": task.get("correlation_id"),
                            })
                            await r.rpush(reply_queue, reply)
                        continue

                    log.info(
                        "confidence_runner.completed",
                        thread_id=thread_id,
                        final_score=result.get("final_score"),
                        decision=result.get("decision"),
                    )

                    # Optionally push result back to a response queue
                    reply_queue = task.get("reply_queue")
                    if reply_queue:
                        reply = json.dumps(
                            {
                                "id": thread_id,
                                "status": "complete",
                                "base_score": result.get("base_score"),
                                "validation_score": result.get("validation_score"),
                                "historical_score": result.get("historical_score"),
                                "reflexive_score": result.get("reflexive_score"),
                                "final_score": result.get("final_score"),
                                "decision": result.get("decision"),
                                "correlation_id": task.get("correlation_id"),
                            }
                        )
                        await r.rpush(reply_queue, reply)

                except Exception as graph_exc:
                    log.error(
                        "confidence_runner.graph_error",
                        error=str(graph_exc),
                        thread_id=thread_id,
                    )

            except Exception as loop_exc:
                log.error("confidence_runner.loop_error", error=str(loop_exc))

import asyncio

if __name__ == "__main__":
    asyncio.run(ConfidenceGraphRunner().run_forever())

