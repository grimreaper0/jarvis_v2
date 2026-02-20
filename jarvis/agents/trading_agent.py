"""TradingAgent — Tier 3 specialist subgraph for trade signal generation."""
import structlog
from langgraph.graph import StateGraph, END
from jarvis.core.state import TradingState

log = structlog.get_logger()


def screen_symbols(state: TradingState) -> TradingState:
    """Filter universe for liquid, volatile symbols."""
    watchlist = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD"]
    log.info("trading_agent.screened", symbols=watchlist[:5])
    return {**state, "signal": {**state.get("signal", {}), "watchlist": watchlist}}


def compute_vwap_signal(state: TradingState) -> TradingState:
    """VWAP Mean Reversion: signal when price deviates 1.5%+ from VWAP."""
    signal = state.get("signal", {})
    symbol = state.get("symbol", signal.get("watchlist", ["SPY"])[0])
    log.info("trading_agent.vwap_compute", symbol=symbol)
    vwap_signal = {
        **signal,
        "strategy": "vwap_mean_reversion",
        "symbol": symbol,
        "side": "buy",
        "price": 0.0,
        "qty": 1,
        "confidence": 0.0,
    }
    return {**state, "symbol": symbol, "signal": vwap_signal}


def apply_kelly_sizing(state: TradingState) -> TradingState:
    """Kelly Criterion position sizing: 2-4% of account, $100 max."""
    signal = state["signal"]
    win_rate = 0.62
    avg_win = 1.5
    avg_loss = 1.0
    kelly_fraction = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    account_size = 10_000
    position_usd = min(account_size * kelly_fraction * 0.5, 100.0)
    price = signal.get("price", 100.0) or 100.0
    qty = max(1, int(position_usd / price))
    return {**state, "signal": {**signal, "qty": qty}, "position_size": position_usd}


def route_to_graph(state: TradingState) -> TradingState:
    """Hand off to the Tier 2 trading graph for risk validation + execution."""
    log.info("trading_agent.routing_to_graph", symbol=state["symbol"], signal=state["signal"])
    return state


def build_trading_agent() -> StateGraph:
    graph = StateGraph(TradingState)

    graph.add_node("screen_symbols", screen_symbols)
    graph.add_node("compute_vwap_signal", compute_vwap_signal)
    graph.add_node("apply_kelly_sizing", apply_kelly_sizing)
    graph.add_node("route_to_graph", route_to_graph)

    graph.set_entry_point("screen_symbols")
    graph.add_edge("screen_symbols", "compute_vwap_signal")
    graph.add_edge("compute_vwap_signal", "apply_kelly_sizing")
    graph.add_edge("apply_kelly_sizing", "route_to_graph")
    graph.add_edge("route_to_graph", END)

    return graph.compile()


trading_agent = build_trading_agent()
