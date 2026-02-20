"""Revenue opportunity evaluation graph (Tier 2 LangGraph).

Decision flow:
    load_opportunity
      -> enrich_opportunity  (AutoMem historical context)
      -> evaluate_confidence (4-layer ConfidenceGate scoring)
      -> route_decision      (conditional: execute / delegate / skip)
      -> execute_opportunity / delegate_opportunity / skip_opportunity
"""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from jarvis.core.confidence import ConfidenceGate
from jarvis.core.memory import AutoMem
from jarvis.core.state import RevenueState

log = structlog.get_logger()

# Maps opportunity task_type -> Tier 3 Redis queue name
QUEUE_MAP: dict[str, str] = {
    "content_post": "instagram_task",
    "content_story": "instagram_task",
    "content_reel": "instagram_task",
    "content_video": "youtube_task",
    "seo_optimization": "youtube_task",
    "trade_stock": "trading_signal",
    "trade_crypto": "trading_signal",
    "backtest": "trading_signal",
    "newsletter": "newsletter_task",
    "newsletter_issue": "newsletter_task",
    "tiktok_video": "tiktok_task",
    "tiktok_trend": "tiktok_task",
}

# Required fields every opportunity must have
REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"task_type", "description"}
)


# ─────────────────────────── Node implementations ────────────────────────────


async def load_opportunity(state: RevenueState) -> dict[str, Any]:
    """Pull opportunity from state and validate required fields."""
    opp: dict[str, Any] = state.get("opportunity", {})

    missing = REQUIRED_FIELDS - set(opp.keys())
    if missing:
        error_msg = f"Opportunity missing required fields: {sorted(missing)}"
        log.warning("revenue.load.invalid", missing=sorted(missing))
        return {
            "opportunity": opp,
            "evaluation_score": 0.0,
            "action": "skip",
            "error": error_msg,
        }

    # Normalise optional fields with safe defaults
    opp.setdefault("estimated_value", 0.0)
    opp.setdefault("confidence_score", 0.0)
    opp.setdefault("context", {})
    opp.setdefault("source", "unknown")

    log.info(
        "revenue.load.ok",
        task_type=opp["task_type"],
        source=opp["source"],
        description=opp["description"][:80],
    )
    return {"opportunity": opp, "error": None}


async def enrich_opportunity(state: RevenueState) -> dict[str, Any]:
    """Search AutoMem for similar past opportunities and add historical context."""
    if state.get("error"):
        return {}

    opp = state["opportunity"]
    task_type: str = opp["task_type"]
    description: str = opp["description"]

    try:
        mem = AutoMem()
        # v2 AutoMem: find_similar_patterns requires an embedding vector.
        # Use get_patterns_by_context (exact type match) as a lighter alternative.
        matching = await mem.get_patterns_by_context(context=task_type, limit=5)

        historical_context: dict[str, Any] = {
            "similar_patterns_count": len(matching),
            "avg_historical_confidence": (
                sum(p.get("confidence_score", 0.0) for p in matching) / len(matching)
                if matching else 0.0
            ),
            "top_patterns": [
                {
                    "name": p.get("description", "")[:60],
                    "confidence": p.get("confidence_score", 0.0),
                }
                for p in matching[:3]
            ],
        }

        # Merge into opportunity context
        updated_opp = dict(opp)
        updated_opp["context"] = {**opp.get("context", {}), "historical": historical_context}

        log.info(
            "revenue.enrich.ok",
            task_type=task_type,
            matching_patterns=len(matching),
        )
        return {"opportunity": updated_opp}

    except Exception as exc:
        # Enrichment failure is non-fatal — proceed with unenriched state
        log.warning("revenue.enrich.failed", error=str(exc))
        return {}


async def evaluate_confidence(state: RevenueState) -> dict[str, Any]:
    """Run the 4-layer ConfidenceGate scoring against the opportunity."""
    if state.get("error"):
        return {"evaluation_score": 0.0, "confidence": 0.0}

    opp = state["opportunity"]
    task_type: str = opp["task_type"]
    description: str = opp["description"]
    context: dict[str, Any] = opp.get("context", {})

    # Layer 1 — Base: task clarity
    base_score = _score_base(task_type, description, context)

    # Layer 2 — Validation: data quality heuristics
    validation_score = _score_validation(task_type, context)

    # Layer 3 — Historical: pattern match boost
    historical_score = _score_historical(opp)

    # Layer 4 — Reflexive: coherence check across first three layers
    reflexive_score = _score_reflexive(base_score, validation_score, historical_score)

    # Weighted combination matching v1 weights
    WEIGHT_BASE = 0.40
    WEIGHT_VALIDATION = 0.20
    WEIGHT_HISTORICAL = 0.30
    WEIGHT_REFLEXIVE = 0.10

    final = (
        base_score * WEIGHT_BASE
        + validation_score * WEIGHT_VALIDATION
        + historical_score * WEIGHT_HISTORICAL
        + reflexive_score * WEIGHT_REFLEXIVE
    )
    final = max(0.0, min(1.0, round(final, 4)))

    log.info(
        "revenue.confidence.evaluated",
        task_type=task_type,
        base=round(base_score, 4),
        validation=round(validation_score, 4),
        historical=round(historical_score, 4),
        reflexive=round(reflexive_score, 4),
        final=final,
    )

    # Attempt audit log — non-fatal (v2 ConfidenceGate takes no bot_name)
    try:
        from jarvis.core.confidence import ConfidenceGate

        gate = ConfidenceGate()
        gate.evaluate(
            base=base_score,
            validation=validation_score,
            historical=historical_score,
            reflexive=reflexive_score,
            context=f"revenue_graph:{task_type}:{description[:60]}",
        )
    except Exception as exc:
        log.warning("revenue.audit.failed", error=str(exc))

    return {
        "evaluation_score": final,
        "confidence": final,
    }


async def route_decision(state: RevenueState) -> dict[str, Any]:
    """Set action based on evaluation score.  Routing edge reads state['action']."""
    if state.get("error"):
        return {"action": "skip"}

    score: float = state.get("evaluation_score", 0.0)

    from config.settings import get_settings
    settings = get_settings()

    if score >= settings.confidence_execute:
        action = "execute"
    elif score >= settings.confidence_delegate:
        action = "delegate"
    else:
        action = "skip"

    log.info("revenue.route", score=score, action=action)
    return {"action": action}


def _route_decision_edge(state: RevenueState) -> str:
    """Conditional edge function — returns the name of the next node."""
    return state.get("action", "skip")


async def execute_opportunity(state: RevenueState) -> dict[str, Any]:
    """Route approved opportunity to the correct Tier 3 agent queue."""
    opp = state["opportunity"]
    task_type: str = opp["task_type"]
    queue_name: str = QUEUE_MAP.get(task_type, "general_task")

    payload = json.dumps(
        {
            "id": str(uuid4()),
            "task_type": task_type,
            "description": opp["description"],
            "context": opp.get("context", {}),
            "confidence": state.get("evaluation_score", 0.0),
            "review_required": False,
            "source": opp.get("source", "revenue_graph"),
        }
    )

    try:
        import redis.asyncio as aioredis

        from config.settings import get_settings

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        await r.rpush(queue_name, payload)
        await r.aclose()
        log.info(
            "revenue.execute.queued",
            queue=queue_name,
            task_type=task_type,
            description=opp["description"][:80],
        )
    except Exception as exc:
        log.error("revenue.execute.redis_failed", error=str(exc), queue=queue_name)

    return {}


async def delegate_opportunity(state: RevenueState) -> dict[str, Any]:
    """Push opportunity to human review queue with full context."""
    opp = state["opportunity"]

    payload = json.dumps(
        {
            "id": str(uuid4()),
            "task_type": opp["task_type"],
            "description": opp["description"],
            "context": opp.get("context", {}),
            "confidence": state.get("evaluation_score", 0.0),
            "review_required": True,
            "reason": "confidence_below_execute_threshold",
            "source": opp.get("source", "revenue_graph"),
        }
    )

    try:
        import redis.asyncio as aioredis

        from config.settings import get_settings

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        await r.rpush("human_review", payload)
        await r.aclose()
        log.info(
            "revenue.delegate.queued",
            task_type=opp["task_type"],
            score=state.get("evaluation_score", 0.0),
        )
    except Exception as exc:
        log.error("revenue.delegate.redis_failed", error=str(exc))

    return {"delegated_to": "human_review"}


async def skip_opportunity(state: RevenueState) -> dict[str, Any]:
    """Log skip reason to AutoMem audit and discard opportunity."""
    opp = state.get("opportunity", {})
    reason = state.get("error") or "confidence_below_delegate_threshold"

    log.info(
        "revenue.skip",
        task_type=opp.get("task_type", "unknown"),
        score=state.get("evaluation_score", 0.0),
        reason=reason,
    )

    # v2 AutoMem: extract_pattern requires an embedding + conversation_id.
    # Log the skip via structured log only; a future integration can persist
    # these through the full audit pipeline when embeddings are available.
    log.info(
        "revenue.skip.logged",
        task_type=opp.get("task_type", "unknown"),
        reason=reason,
        score=state.get("evaluation_score", 0.0),
        source=opp.get("source", "revenue_graph"),
    )

    return {}


# ─────────────────────────── Helper scoring functions ────────────────────────

KNOWN_TASK_TYPES: frozenset[str] = frozenset(
    {
        "content_post", "content_story", "content_reel", "content_video",
        "trade_stock", "trade_crypto", "hashtag_research", "seo_optimization",
        "trend_analysis", "revenue_analysis", "pattern_extraction",
        "backtest", "portfolio_rebalance", "newsletter", "newsletter_issue",
        "tiktok_video", "tiktok_trend",
    }
)


def _score_base(task_type: str, description: str, context: dict[str, Any]) -> float:
    score = 0.0
    if task_type in KNOWN_TASK_TYPES:
        score += 0.35
    if len(description.strip()) >= 20:
        score += 0.35
    if context and len(context) > 0:
        score += 0.30
    return min(1.0, score)


def _score_validation(task_type: str, context: dict[str, Any]) -> float:
    score = 0.5
    if context.get("api_available", False):
        score += 0.20
    if context.get("data_fresh", False):
        score += 0.15
    if not context.get("within_limits", True):
        score -= 0.30
    return max(0.0, min(1.0, score))


def _score_historical(opp: dict[str, Any]) -> float:
    """Derive historical score from opportunity metadata and enriched context."""
    baseline = 0.30
    historical = opp.get("context", {}).get("historical", {})
    avg_conf = historical.get("avg_historical_confidence", 0.0)
    count = historical.get("similar_patterns_count", 0)

    if count == 0:
        return baseline

    boost = min(0.20, 0.05 * count)
    return min(1.0, baseline + avg_conf * 0.30 + boost)


def _score_reflexive(base: float, validation: float, historical: float) -> float:
    scores = [base, validation, historical]
    spread = max(scores) - min(scores)
    avg = sum(scores) / len(scores)
    if spread <= 0.15:
        return min(1.0, avg + 0.10)
    elif spread <= 0.30:
        return avg
    else:
        return max(0.0, avg - 0.15)


# ─────────────────────────── Graph builder ───────────────────────────────────


def build_revenue_graph():
    """Build and compile the revenue opportunity evaluation graph."""
    graph = StateGraph(RevenueState)

    graph.add_node("load_opportunity", load_opportunity)
    graph.add_node("enrich_opportunity", enrich_opportunity)
    graph.add_node("evaluate_confidence", evaluate_confidence)
    graph.add_node("route_decision", route_decision)
    graph.add_node("execute_opportunity", execute_opportunity)
    graph.add_node("delegate_opportunity", delegate_opportunity)
    graph.add_node("skip_opportunity", skip_opportunity)

    graph.set_entry_point("load_opportunity")
    graph.add_edge("load_opportunity", "enrich_opportunity")
    graph.add_edge("enrich_opportunity", "evaluate_confidence")
    graph.add_edge("evaluate_confidence", "route_decision")

    graph.add_conditional_edges(
        "route_decision",
        _route_decision_edge,
        {
            "execute": "execute_opportunity",
            "delegate": "delegate_opportunity",
            "skip": "skip_opportunity",
        },
    )

    graph.add_edge("execute_opportunity", END)
    graph.add_edge("delegate_opportunity", END)
    graph.add_edge("skip_opportunity", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


revenue_graph = build_revenue_graph()


# ─────────────────────────── Graph Runner ────────────────────────────────────


class RevenueGraphRunner:
    """Consume from the 'revenue_opportunity' Redis queue and run the graph for each task."""

    QUEUE = "revenue_opportunity"

    def __init__(self) -> None:
        self.graph = build_revenue_graph()
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis

            from config.settings import get_settings

            settings = get_settings()
            self._redis = aioredis.from_url(settings.redis_url)
        return self._redis

    async def run_forever(self) -> None:
        """Consume from Redis queue and run graph for each task indefinitely."""
        log.info("revenue_runner.starting", queue=self.QUEUE)
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
                    log.error("revenue_runner.bad_json", error=str(exc))
                    continue

                thread_id: str = task.get("id") or str(uuid4())
                config = {"configurable": {"thread_id": thread_id}}

                # Build initial state
                initial_state: RevenueState = {
                    "messages": [],
                    "confidence": task.get("confidence_score", 0.0),
                    "error": None,
                    "opportunity": task,
                    "evaluation_score": 0.0,
                    "action": "skip",
                    "delegated_to": None,
                }

                try:
                    result = await self.graph.ainvoke(initial_state, config=config)
                    log.info(
                        "revenue_runner.completed",
                        thread_id=thread_id,
                        action=result.get("action", "unknown"),
                        score=result.get("evaluation_score", 0.0),
                    )

                    # Log result summary — full pattern persistence requires
                    # embeddings (Ollama) which are wired at the worker level.
                    log.info(
                        "revenue_runner.result_summary",
                        thread_id=thread_id,
                        task_type=task.get("task_type"),
                        action=result.get("action"),
                        score=result.get("evaluation_score", 0.0),
                        delegated_to=result.get("delegated_to"),
                    )

                except Exception as graph_exc:
                    log.error("revenue_runner.graph_error", error=str(graph_exc), thread_id=thread_id)

            except Exception as loop_exc:
                log.error("revenue_runner.loop_error", error=str(loop_exc))
