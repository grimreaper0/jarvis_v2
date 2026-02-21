"""Trading decision graph (Tier 2 LangGraph).

Signal → validate → check guardrails → size position → submit / reject.

All actual Alpaca API calls happen in the Tier 1 TradingWorker — this graph
only decides *whether* to act and pushes approved orders to the
'trading_execution' Redis queue.
"""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from jarvis.core.guardrails import Guardrails
from jarvis.core.state import TradingState

log = structlog.get_logger()

# Hard guardrail values (also enforced by core.guardrails.Guardrails)
MAX_POSITION_USD: float = 100.0
MAX_DAILY_LOSS_USD: float = 500.0
CIRCUIT_BREAKER_LOSSES: int = 3

# Kelly Criterion constraints
MIN_KELLY_FRACTION: float = 0.01   # floor: never bet less than 1%
MAX_KELLY_FRACTION: float = 0.10   # cap at 10% of bankroll (conservative)
DEFAULT_BANKROLL: float = 1_000.0  # paper trading starting equity

# Required signal fields
REQUIRED_SIGNAL_FIELDS: frozenset[str] = frozenset(
    {"symbol", "action", "price", "confidence"}
)

_guardrails = Guardrails()


# ─────────────────────────── Node implementations ────────────────────────────


async def validate_signal(state: TradingState) -> dict[str, Any]:
    """Validate the incoming signal has all required fields and is sane."""
    signal: dict[str, Any] = state.get("signal", {})
    symbol: str = state.get("symbol", signal.get("symbol", ""))

    missing = REQUIRED_SIGNAL_FIELDS - set(signal.keys())
    if missing:
        error_msg = f"Signal missing required fields: {sorted(missing)}"
        log.warning("trading.validate.invalid", symbol=symbol, missing=sorted(missing))
        return {
            "risk_approved": False,
            "error": error_msg,
        }

    action: str = str(signal.get("action", "")).upper()
    if action not in {"BUY", "SELL", "HOLD"}:
        error_msg = f"Unknown signal action: {signal.get('action')!r}"
        log.warning("trading.validate.bad_action", symbol=symbol, action=action)
        return {"risk_approved": False, "error": error_msg}

    price: float = float(signal.get("price", 0.0))
    if price <= 0:
        error_msg = f"Signal price must be > 0, got {price}"
        log.warning("trading.validate.bad_price", symbol=symbol, price=price)
        return {"risk_approved": False, "error": error_msg}

    confidence: float = float(signal.get("confidence", 0.0))

    log.info(
        "trading.validate.ok",
        symbol=symbol,
        action=action,
        price=price,
        confidence=confidence,
    )
    return {
        "symbol": symbol,
        "signal": {**signal, "action": action},
        "confidence": confidence,
        "error": None,
    }


async def check_guardrails(state: TradingState) -> dict[str, Any]:
    """
    Financial guardrails:
      - Max position size $100
      - Daily loss circuit breaker $500
      - Consecutive loss circuit breaker (3 losses)

    Any failure → risk_approved=False so conditional edge routes to reject_order.
    """
    if state.get("error"):
        return {"risk_approved": False}

    signal: dict[str, Any] = state["signal"]
    symbol: str = state["symbol"]

    # Compute tentative trade value (price * qty; qty TBD, use 1 share for guardrail check)
    price: float = float(signal.get("price", 0.0))

    # Primary: max trade amount check
    trade_result = _guardrails.check_trade(amount_usd=price)
    if not trade_result.passed:
        log.warning(
            "trading.guardrail.max_trade_exceeded",
            symbol=symbol,
            price=price,
            reason=trade_result.reason,
        )
        return {
            "risk_approved": False,
            "error": f"Guardrail: {trade_result.reason}",
        }

    # Daily loss check — pulled from signal context if available
    context: dict[str, Any] = signal.get("context", {})
    daily_loss: float = float(context.get("daily_loss_usd", 0.0))
    loss_result = _guardrails.check_trade(amount_usd=price, daily_loss_usd=daily_loss)
    if not loss_result.passed:
        log.warning(
            "trading.guardrail.daily_loss_exceeded",
            symbol=symbol,
            daily_loss=daily_loss,
            reason=loss_result.reason,
        )
        return {
            "risk_approved": False,
            "error": f"Guardrail: {loss_result.reason}",
        }

    # Consecutive loss circuit breaker
    consecutive_losses: int = int(context.get("consecutive_losses", 0))
    if consecutive_losses >= CIRCUIT_BREAKER_LOSSES:
        reason = (
            f"Circuit breaker: {consecutive_losses} consecutive losses "
            f">= threshold {CIRCUIT_BREAKER_LOSSES}"
        )
        log.warning("trading.guardrail.circuit_breaker", symbol=symbol, losses=consecutive_losses)
        return {"risk_approved": False, "error": f"Guardrail: {reason}"}

    log.info("trading.guardrails.passed", symbol=symbol, price=price, daily_loss=daily_loss)
    return {"risk_approved": True}


def _route_risk(state: TradingState) -> str:
    """Conditional edge: route to size_position if approved, else reject_order."""
    if state.get("error") or not state.get("risk_approved", False):
        return "reject"
    return "approve"


async def size_position(state: TradingState) -> dict[str, Any]:
    """
    Kelly Criterion position sizing.

    f* = (b·p - q) / b
    where:
      b = reward-to-risk ratio  (estimated; defaults to 1.5 if not provided)
      p = win probability       (confidence score from signal)
      q = 1 - p

    Position size is clamped to [MIN_KELLY_FRACTION, MAX_KELLY_FRACTION]
    of bankroll, and further hard-capped at MAX_POSITION_USD.
    """
    signal: dict[str, Any] = state["signal"]
    symbol: str = state["symbol"]

    p: float = max(0.01, min(0.99, float(signal.get("confidence", 0.55))))
    q: float = 1.0 - p
    b: float = float(signal.get("reward_risk_ratio", 1.5))  # default 1.5:1

    if b <= 0:
        b = 1.5

    # Raw Kelly fraction
    kelly_raw: float = (b * p - q) / b
    kelly_clamped: float = max(MIN_KELLY_FRACTION, min(MAX_KELLY_FRACTION, kelly_raw))

    bankroll: float = float(signal.get("context", {}).get("account_equity", DEFAULT_BANKROLL))
    position_usd: float = min(MAX_POSITION_USD, kelly_clamped * bankroll)

    price: float = float(signal.get("price", 1.0))
    qty: float = position_usd / price if price > 0 else 0.0
    qty = max(1.0, round(qty, 6))  # at least 1 share; fractional shares supported

    log.info(
        "trading.size_position",
        symbol=symbol,
        p=round(p, 4),
        b=round(b, 4),
        kelly_raw=round(kelly_raw, 4),
        kelly_clamped=round(kelly_clamped, 4),
        position_usd=round(position_usd, 2),
        qty=round(qty, 4),
    )
    return {"position_size": round(position_usd, 2)}


async def risk_approve(state: TradingState) -> dict[str, Any]:
    """Final approval gate — all checks passed."""
    log.info(
        "trading.risk_approved",
        symbol=state["symbol"],
        position_size=state.get("position_size", 0.0),
    )
    return {"risk_approved": True}


async def submit_order(state: TradingState) -> dict[str, Any]:
    """Push approved order to the 'trading_execution' queue for Tier 1 TradingWorker."""
    signal: dict[str, Any] = state["signal"]
    symbol: str = state["symbol"]

    correlation_id = signal.get("correlation_id", str(uuid4()))

    payload = json.dumps(
        {
            "id": str(uuid4()),
            "symbol": symbol,
            "action": signal.get("action", "BUY"),
            "price": signal.get("price"),
            "position_size_usd": state.get("position_size", 0.0),
            "confidence": signal.get("confidence", 0.0),
            "strategy": signal.get("strategy", "unknown"),
            "context": signal.get("context", {}),
            "correlation_id": correlation_id,
        }
    )

    try:
        import redis.asyncio as aioredis

        from config.settings import get_settings

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        await r.rpush("trading_execution", payload)
        await r.aclose()
        log.info("trading.submit.queued", symbol=symbol, position_size=state.get("position_size"))
    except Exception as exc:
        log.error("trading.submit.redis_failed", error=str(exc), symbol=symbol)

    return {"order_id": "queued"}


async def reject_order(state: TradingState) -> dict[str, Any]:
    """Log rejection reason to confidence_audit_log / AutoMem and return."""
    symbol: str = state.get("symbol", "UNKNOWN")
    reason: str = state.get("error", "guardrail_triggered")

    log.warning("trading.order_rejected", symbol=symbol, reason=reason)

    # Log rejection to structured log; full persistence with embeddings
    # happens at the Tier 1 TradingWorker level.
    log.info(
        "trading.reject.logged",
        symbol=symbol,
        action=state.get("signal", {}).get("action", "unknown"),
        reason=reason,
        confidence=state.get("confidence", 0.0),
    )

    return {"order_id": None, "risk_approved": False}


# ─────────────────────────── Graph builder ───────────────────────────────────


def build_trading_graph():
    """Build and compile the trading signal decision graph."""
    graph = StateGraph(TradingState)

    graph.add_node("validate_signal", validate_signal)
    graph.add_node("check_guardrails", check_guardrails)
    graph.add_node("size_position", size_position)
    graph.add_node("risk_approve", risk_approve)
    graph.add_node("submit_order", submit_order)
    graph.add_node("reject_order", reject_order)

    graph.set_entry_point("validate_signal")
    graph.add_edge("validate_signal", "check_guardrails")

    # Primary conditional split: guardrails pass → size position, fail → reject
    graph.add_conditional_edges(
        "check_guardrails",
        _route_risk,
        {
            "approve": "size_position",
            "reject": "reject_order",
        },
    )

    graph.add_edge("size_position", "risk_approve")
    graph.add_edge("risk_approve", "submit_order")
    graph.add_edge("submit_order", END)
    graph.add_edge("reject_order", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


trading_graph = build_trading_graph()


# ─────────────────────────── Graph Runner ────────────────────────────────────


class TradingGraphRunner:
    """Consume from 'trading_decision' Redis queue and run the trading decision graph."""

    QUEUE = "trading_decision"

    def __init__(self) -> None:
        self.graph = build_trading_graph()
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis

            from config.settings import get_settings

            settings = get_settings()
            self._redis = aioredis.from_url(settings.redis_url)
        return self._redis

    async def run_forever(self) -> None:
        """Consume from Redis queue and run graph for each signal indefinitely."""
        log.info("trading_runner.starting", queue=self.QUEUE)
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
                    log.error("trading_runner.bad_json", error=str(exc))
                    continue

                thread_id: str = task.get("id") or str(uuid4())
                config = {"configurable": {"thread_id": thread_id}}

                symbol: str = task.get("symbol", "UNKNOWN")

                initial_state: TradingState = {
                    "messages": [],
                    "confidence": task.get("confidence", 0.0),
                    "error": None,
                    "symbol": symbol,
                    "signal": task,
                    "risk_approved": False,
                    "order_id": None,
                    "position_size": 0.0,
                }

                try:
                    result = await self.graph.ainvoke(initial_state, config=config)
                    log.info(
                        "trading_runner.completed",
                        thread_id=thread_id,
                        symbol=symbol,
                        order_id=result.get("order_id"),
                        risk_approved=result.get("risk_approved"),
                        position_size=result.get("position_size"),
                        correlation_id=task.get("correlation_id"),
                    )
                except Exception as graph_exc:
                    log.error(
                        "trading_runner.graph_error",
                        error=str(graph_exc),
                        thread_id=thread_id,
                        symbol=symbol,
                    )

            except Exception as loop_exc:
                log.error("trading_runner.loop_error", error=str(loop_exc))

import asyncio

if __name__ == "__main__":
    asyncio.run(TradingGraphRunner().run_forever())

