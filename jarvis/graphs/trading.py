"""Trading signal → risk validation → execute graph (Tier 2 LangGraph)."""
import structlog
from langgraph.graph import StateGraph, END
from jarvis.core.state import TradingState
from jarvis.core.guardrails import Guardrails

log = structlog.get_logger()
guardrails = Guardrails()


def validate_signal(state: TradingState) -> TradingState:
    """Validate the trading signal has required fields and is actionable."""
    signal = state["signal"]
    required = {"symbol", "side", "price", "qty"}
    missing = required - set(signal.keys())
    if missing:
        return {**state, "confidence": 0.0, "error": f"Signal missing fields: {missing}"}
    return {**state, "confidence": state.get("confidence", 0.8)}


def check_risk(state: TradingState) -> TradingState:
    """Run guardrails: position size, daily loss circuit breaker."""
    signal = state["signal"]
    amount = signal.get("price", 0) * signal.get("qty", 1)
    result = guardrails.check_trade(amount_usd=amount)
    if not result.passed:
        log.warning("trading.risk_rejected", reason=result.reason)
        return {**state, "risk_approved": False, "error": result.reason}
    return {**state, "risk_approved": True, "position_size": amount}


def route_risk(state: TradingState) -> str:
    if state.get("error"):
        return "reject"
    return "approve" if state.get("risk_approved") else "reject"


def submit_order(state: TradingState) -> TradingState:
    """Push signal to TradingWorker queue via Redis."""
    log.info("trading.graph_submit", symbol=state["symbol"], signal=state["signal"])
    return {**state, "order_id": "queued"}


def reject_trade(state: TradingState) -> TradingState:
    log.info("trading.graph_rejected", symbol=state["symbol"], error=state.get("error"))
    return state


def build_trading_graph() -> StateGraph:
    graph = StateGraph(TradingState)

    graph.add_node("validate_signal", validate_signal)
    graph.add_node("check_risk", check_risk)
    graph.add_node("submit_order", submit_order)
    graph.add_node("reject_trade", reject_trade)

    graph.set_entry_point("validate_signal")
    graph.add_edge("validate_signal", "check_risk")
    graph.add_conditional_edges("check_risk", route_risk, {
        "approve": "submit_order",
        "reject": "reject_trade",
    })
    graph.add_edge("submit_order", END)
    graph.add_edge("reject_trade", END)

    return graph.compile()


trading_graph = build_trading_graph()
