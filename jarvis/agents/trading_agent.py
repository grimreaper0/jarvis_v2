"""TradingAgent — Tier 3 specialist subgraph for VWAP mean reversion signal generation.

Nodes:
  1. screen_symbols    — Filter universe to high-liquidity, high-volatility candidates
  2. calculate_vwap    — Compute VWAP + RSI for each symbol using bar data
  3. generate_signal   — VWAP mean reversion signal (deviation > 1.5% = signal)
  4. assess_confidence — ConfidenceGate-style evaluation of the signal
  5. route_to_tier2    — If confidence >= 0.6, push to 'trading_decision' queue (Tier 2)

VWAP formula (ported from v1 indicators/vwap.py):
    Typical Price = (High + Low + Close) / 3
    VWAP = Σ(Typical Price × Volume) / Σ(Volume)  [cumulative from market open]

Signal logic (ported from v1 strategies/vwap_live.py):
    BUY  — price < VWAP by >=1.5%, RSI < 30, volume > 1.2× avg
    SELL — price > VWAP by >=1.5%, RSI > 70, volume > 1.2× avg
"""
from __future__ import annotations

import asyncio
import json
import structlog
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from jarvis.core.router import LLMRouter, LLMRequest

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TradingAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    symbol: str
    watchlist: list[str]
    bars: dict[str, list[dict]]  # symbol -> list of bar dicts
    vwap_data: dict[str, dict]   # symbol -> {vwap, rsi, price, volume_ratio, deviation_pct}
    signal: dict[str, Any]
    position_size: float
    confidence: float
    approved: bool
    error: str | None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEGA_CAP_FALLBACK = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

LARGE_CAP_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "JNJ", "WMT", "PG", "MA", "HD", "CVX",
    "AMD", "INTC", "QCOM", "NFLX", "PYPL",
    "BAC", "WFC", "GS", "MS", "C",
    "SPY", "QQQ", "IWM",
]

SCREEN_CRITERIA = {
    "min_volume": 500_000,
    "min_price": 5.0,
    "max_price": 1000.0,
}

VWAP_DEVIATION_THRESHOLD = 0.015   # 1.5% deviation triggers signal
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
VOLUME_MULTIPLIER = 1.2            # Must exceed 1.2× 20-bar avg volume

KELLY_WIN_RATE = 0.62
KELLY_AVG_WIN = 1.5
KELLY_AVG_LOSS = 1.0
MAX_POSITION_USD = 100.0
ACCOUNT_SIZE_USD = 10_000.0

CONFIDENCE_EXECUTE = 0.90
CONFIDENCE_DELEGATE = 0.60


# ---------------------------------------------------------------------------
# VWAP + RSI computation (pure Python — no backtrader dependency)
# ---------------------------------------------------------------------------

def _compute_vwap(bars: list[dict]) -> float:
    """Cumulative VWAP: Σ(typical_price × volume) / Σ(volume)."""
    cum_pv = 0.0
    cum_v = 0.0
    for bar in bars:
        h = float(bar.get("h", bar.get("high", 0)))
        l = float(bar.get("l", bar.get("low", 0)))
        c = float(bar.get("c", bar.get("close", 0)))
        v = float(bar.get("v", bar.get("volume", 1)))
        typical = (h + l + c) / 3.0
        cum_pv += typical * v
        cum_v += v
    return cum_pv / cum_v if cum_v > 0 else 0.0


def _compute_rsi(bars: list[dict], period: int = 14) -> float:
    """Wilder RSI from close prices."""
    closes = [float(b.get("c", b.get("close", 0))) for b in bars]
    if len(closes) < period + 1:
        return 50.0  # Neutral when insufficient data

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas[-period:]]
    losses = [abs(min(d, 0)) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def _compute_volume_ratio(bars: list[dict], window: int = 20) -> float:
    """Current bar volume / 20-bar rolling average volume."""
    volumes = [float(b.get("v", b.get("volume", 0))) for b in bars]
    if not volumes:
        return 1.0
    avg_vol = sum(volumes[-window:]) / min(len(volumes), window)
    if avg_vol == 0:
        return 1.0
    return round(volumes[-1] / avg_vol, 4)


def _kelly_position_size(account_size: float) -> float:
    """Kelly Criterion position sizing, capped at MAX_POSITION_USD."""
    kelly_f = KELLY_WIN_RATE - (1 - KELLY_WIN_RATE) / (KELLY_AVG_WIN / KELLY_AVG_LOSS)
    position = account_size * kelly_f * 0.5
    return round(min(position, MAX_POSITION_USD), 2)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def screen_symbols(state: TradingAgentState) -> dict:
    """Filter stock universe to high-liquidity, tradeable candidates."""
    watchlist = LARGE_CAP_UNIVERSE[:15]

    try:
        import yfinance as yf

        screened: list[str] = []
        for ticker in LARGE_CAP_UNIVERSE[:30]:
            try:
                info = yf.Ticker(ticker).fast_info
                price = float(getattr(info, "last_price", 0) or 0)
                volume = float(getattr(info, "three_month_average_volume", 0) or 0)
                if (SCREEN_CRITERIA["min_price"] <= price <= SCREEN_CRITERIA["max_price"]
                        and volume >= SCREEN_CRITERIA["min_volume"]):
                    screened.append(ticker)
                    if len(screened) >= 10:
                        break
            except Exception:
                continue

        watchlist = screened if screened else MEGA_CAP_FALLBACK
        log.info("trading_agent.screen_symbols", count=len(watchlist), symbols=watchlist[:5])
    except Exception as exc:
        log.warning("trading_agent.screen_symbols.failed", error=str(exc))
        watchlist = MEGA_CAP_FALLBACK

    symbol = state.get("symbol") or watchlist[0]
    return {"watchlist": watchlist, "symbol": symbol}


def calculate_vwap(state: TradingAgentState) -> dict:
    """Fetch 5-min bars and compute VWAP + RSI for the primary symbol."""
    symbol = state.get("symbol", "SPY")
    bars_data: dict[str, list[dict]] = {}
    vwap_data: dict[str, dict] = {}

    symbols_to_analyze = [symbol] + [s for s in state.get("watchlist", [])[:5] if s != symbol]

    for sym in symbols_to_analyze:
        try:
            import yfinance as yf

            ticker = yf.Ticker(sym)
            hist = ticker.history(period="1d", interval="5m")

            if len(hist) < 5:
                log.warning("trading_agent.calculate_vwap.insufficient_data", symbol=sym, bars=len(hist))
                continue

            raw_bars = []
            for idx, row in hist.iterrows():
                raw_bars.append({
                    "h": float(row["High"]),
                    "l": float(row["Low"]),
                    "c": float(row["Close"]),
                    "v": float(row["Volume"]),
                })

            bars_data[sym] = raw_bars
            vwap = _compute_vwap(raw_bars)
            rsi = _compute_rsi(raw_bars)
            volume_ratio = _compute_volume_ratio(raw_bars)
            current_price = raw_bars[-1]["c"]
            deviation_pct = (current_price - vwap) / vwap if vwap > 0 else 0.0

            vwap_data[sym] = {
                "vwap": round(vwap, 4),
                "rsi": rsi,
                "price": round(current_price, 4),
                "volume_ratio": volume_ratio,
                "deviation_pct": round(deviation_pct * 100, 4),
            }
            log.debug(
                "trading_agent.vwap_computed",
                symbol=sym,
                price=round(current_price, 2),
                vwap=round(vwap, 2),
                deviation_pct=round(deviation_pct * 100, 2),
                rsi=rsi,
            )
        except Exception as exc:
            log.warning("trading_agent.calculate_vwap.error", symbol=sym, error=str(exc))

    if not vwap_data:
        vwap_data[symbol] = {
            "vwap": 0.0, "rsi": 50.0, "price": 0.0,
            "volume_ratio": 1.0, "deviation_pct": 0.0,
        }

    log.info("trading_agent.calculate_vwap", symbols_computed=len(vwap_data))
    return {"bars": bars_data, "vwap_data": vwap_data}


def generate_signal(state: TradingAgentState) -> dict:
    """VWAP mean reversion signal: deviation > 1.5% with RSI + volume confirmation."""
    vwap_data = state.get("vwap_data", {})
    symbol = state.get("symbol", "SPY")

    best_signal: dict[str, Any] = {
        "strategy": "vwap_mean_reversion",
        "symbol": symbol,
        "side": None,
        "price": 0.0,
        "vwap": 0.0,
        "deviation_pct": 0.0,
        "rsi": 50.0,
        "volume_ratio": 1.0,
        "reason": "no_signal",
        "qty": 0,
        "confidence": 0.0,
    }

    for sym, data in vwap_data.items():
        price = data["price"]
        vwap = data["vwap"]
        rsi = data["rsi"]
        volume_ratio = data["volume_ratio"]
        deviation_pct = data["deviation_pct"]

        if price == 0 or vwap == 0:
            continue

        is_oversold = rsi < RSI_OVERSOLD
        is_below_vwap = deviation_pct < -(VWAP_DEVIATION_THRESHOLD * 100)
        has_volume = volume_ratio >= VOLUME_MULTIPLIER

        is_overbought = rsi > RSI_OVERBOUGHT
        is_above_vwap = deviation_pct > (VWAP_DEVIATION_THRESHOLD * 100)

        if is_oversold and is_below_vwap and has_volume:
            best_signal.update({
                "symbol": sym,
                "side": "buy",
                "price": price,
                "vwap": vwap,
                "deviation_pct": deviation_pct,
                "rsi": rsi,
                "volume_ratio": volume_ratio,
                "reason": f"VWAP BUY: {deviation_pct:.1f}% below VWAP, RSI {rsi:.0f}",
                "confidence": 0.75,
            })
            symbol = sym
            log.info("trading_agent.buy_signal", symbol=sym, deviation_pct=deviation_pct, rsi=rsi)
            break

        elif is_overbought and is_above_vwap and has_volume:
            best_signal.update({
                "symbol": sym,
                "side": "sell",
                "price": price,
                "vwap": vwap,
                "deviation_pct": deviation_pct,
                "rsi": rsi,
                "volume_ratio": volume_ratio,
                "reason": f"VWAP SELL: {deviation_pct:.1f}% above VWAP, RSI {rsi:.0f}",
                "confidence": 0.70,
            })
            symbol = sym
            log.info("trading_agent.sell_signal", symbol=sym, deviation_pct=deviation_pct, rsi=rsi)
            break

    log.info("trading_agent.generate_signal", side=best_signal["side"], symbol=best_signal["symbol"])
    return {"signal": best_signal, "symbol": best_signal["symbol"]}


async def assess_confidence(state: TradingAgentState) -> dict:
    """4-layer confidence assessment: base + validation + historical + reflexive."""
    signal = state.get("signal", {})
    side = signal.get("side")

    if not side:
        return {"confidence": 0.0, "position_size": 0.0}

    # Layer 1: Base — from signal generator
    base = signal.get("confidence", 0.0)

    # Layer 2: Validation — check data quality
    validation = 0.0
    if signal.get("price", 0) > 0 and signal.get("vwap", 0) > 0:
        validation += 0.5
    if signal.get("volume_ratio", 0) >= VOLUME_MULTIPLIER:
        validation += 0.3
    rsi = signal.get("rsi", 50)
    if (side == "buy" and rsi < 35) or (side == "sell" and rsi > 65):
        validation += 0.2

    # Layer 3: Historical — check AutoMem patterns (lightweight)
    historical = 0.5  # Default neutral
    try:
        import psycopg2
        import psycopg2.extras
        from config.settings import get_settings

        settings = get_settings()
        conn = psycopg2.connect(settings.postgres_url)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT AVG(confidence_score) as avg_conf
            FROM patterns
            WHERE context ILIKE %s AND success_count > 0
            """,
            (f"%vwap%",),
        )
        row = cur.fetchone()
        conn.close()
        if row and row["avg_conf"]:
            historical = float(row["avg_conf"])
    except Exception:
        pass

    # Layer 4: Reflexive — LLM second opinion on the signal
    reflexive = 0.5
    try:
        router = LLMRouter()
        prompt = (
            f"Evaluate this VWAP trading signal briefly:\n"
            f"Symbol: {signal.get('symbol')}, Side: {side}\n"
            f"Price: {signal.get('price'):.2f}, VWAP: {signal.get('vwap'):.2f}\n"
            f"Deviation: {signal.get('deviation_pct'):.2f}%, RSI: {signal.get('rsi'):.1f}\n"
            f"Volume ratio: {signal.get('volume_ratio'):.2f}x\n\n"
            f"Rate confidence 0.0-1.0 for this mean-reversion trade. "
            f"Reply with ONLY a number like: 0.72"
        )
        resp = await router.complete(LLMRequest(prompt=prompt, reasoning=True, max_tokens=50))
        text = resp.content.strip().split("\n")[-1].strip()
        reflexive = min(max(float(text), 0.0), 1.0)
    except Exception as exc:
        log.warning("trading_agent.reflexive_confidence.failed", error=str(exc))

    weights = {"base": 0.40, "validation": 0.20, "historical": 0.30, "reflexive": 0.10}
    final = (
        base * weights["base"]
        + validation * weights["validation"]
        + historical * weights["historical"]
        + reflexive * weights["reflexive"]
    )
    final = round(final, 4)

    position_size = _kelly_position_size(ACCOUNT_SIZE_USD) if final >= CONFIDENCE_DELEGATE else 0.0
    qty = max(1, int(position_size / max(signal.get("price", 100), 1))) if position_size > 0 else 0
    signal["qty"] = qty
    signal["final_confidence"] = final

    log.info(
        "trading_agent.assess_confidence",
        base=round(base, 3),
        validation=round(validation, 3),
        historical=round(historical, 3),
        reflexive=round(reflexive, 3),
        final=final,
        qty=qty,
    )
    return {"confidence": final, "position_size": position_size, "signal": signal}


async def route_to_tier2(state: TradingAgentState) -> dict:
    """Push signal to 'trading_decision' queue if confidence >= 0.6."""
    confidence = state.get("confidence", 0.0)
    signal = state.get("signal", {})
    approved = confidence >= CONFIDENCE_DELEGATE and signal.get("side") is not None

    log.info(
        "trading_agent.route_to_tier2",
        confidence=confidence,
        approved=approved,
        symbol=state.get("symbol"),
        side=signal.get("side"),
    )

    if approved:
        try:
            import redis.asyncio as aioredis
            from config.settings import get_settings

            settings = get_settings()
            r = aioredis.from_url(settings.redis_url)
            payload = json.dumps({
                "source": "trading_agent",
                "symbol": state.get("symbol"),
                "signal": signal,
                "confidence": confidence,
                "position_size": state.get("position_size"),
            })
            await r.lpush("trading_decision", payload)
            await r.aclose()
            log.info("trading_agent.pushed_to_tier2", symbol=state.get("symbol"), confidence=confidence)
        except Exception as exc:
            log.warning("trading_agent.redis_push.failed", error=str(exc))

    return {"approved": approved}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_trading_agent():
    """Build and compile the Trading specialist subgraph."""
    graph = StateGraph(TradingAgentState)

    graph.add_node("screen_symbols", screen_symbols)
    graph.add_node("calculate_vwap", calculate_vwap)
    graph.add_node("generate_signal", generate_signal)
    graph.add_node("assess_confidence", assess_confidence)
    graph.add_node("route_to_tier2", route_to_tier2)

    graph.set_entry_point("screen_symbols")
    graph.add_edge("screen_symbols", "calculate_vwap")
    graph.add_edge("calculate_vwap", "generate_signal")
    graph.add_edge("generate_signal", "assess_confidence")
    graph.add_edge("assess_confidence", "route_to_tier2")
    graph.add_edge("route_to_tier2", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# AgentRunner — Redis consumer on "trading_task" queue
# ---------------------------------------------------------------------------

class AgentRunner:
    def __init__(self):
        self.agent = build_trading_agent()

    async def run(self):
        import redis.asyncio as aioredis
        from config.settings import get_settings

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        log.info("trading_runner.started", queue="trading_task")

        while True:
            try:
                raw = await r.brpop("trading_task", timeout=10)
                if raw is None:
                    continue

                _, data = raw
                task = json.loads(data)
                log.info("trading_runner.task_received", symbol=task.get("symbol"))

                initial_state: TradingAgentState = {
                    "messages": [],
                    "symbol": task.get("symbol", ""),
                    "watchlist": task.get("watchlist", []),
                    "bars": {},
                    "vwap_data": {},
                    "signal": {},
                    "position_size": 0.0,
                    "confidence": 0.0,
                    "approved": False,
                    "error": None,
                }

                result = await self.agent.ainvoke(initial_state)
                log.info(
                    "trading_runner.task_complete",
                    approved=result.get("approved"),
                    confidence=result.get("confidence"),
                    signal_side=result.get("signal", {}).get("side"),
                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("trading_runner.error", error=str(exc))
                await asyncio.sleep(5)

        await r.aclose()


trading_agent = build_trading_agent()


if __name__ == "__main__":
    async def _main():
        runner = AgentRunner()
        await runner.run()

    asyncio.run(_main())
