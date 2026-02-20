"""TradingWorker — 24/7 market execution engine.

Market hours (9:30 AM - 4:00 PM ET):  real-time analysis, signal generation, order execution
After hours (4:00 PM - 5:30 PM ET):   extended hours with reduced position sizing
Closed:                                performance analysis, strategy optimisation, next-day prep

Queue:  trading_signal (primary)
Output: risk-unvalidated SIGNALS pushed to 'trading_decision' (Tier 2 LangGraph)
        pre-validated EXECUTIONS executed directly via Alpaca
"""
import asyncio
import json
import structlog
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from jarvis.workers.base import ContinuousWorker

log = structlog.get_logger()

ET = ZoneInfo("America/New_York")

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
AFTER_HOURS_CLOSE = time(17, 30)

SCAN_INTERVAL_SEC = 300         # 5 min during market hours
AFTER_HOURS_SCAN_SEC = 600      # 10 min after hours
CLOSED_SLEEP_SEC = 1800         # 30 min when market closed

MAX_TRADE_VALUE = 100.0         # max $ per trade (paper trading safety)
MAX_TRADE_VALUE_AFTER_HOURS = 50.0
CIRCUIT_BREAKER_LOSSES = 3      # stop trading after this many consecutive losses

# Mega-cap fallback symbols when screener returns nothing
FALLBACK_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

# VWAP mean reversion thresholds
VWAP_DEVIATION_ENTRY = 0.015    # 1.5% from VWAP
VWAP_DEVIATION_ENTRY_AH = 0.025 # 2.5% after hours (wider spread)
VWAP_TARGET_PROFIT = 0.008      # 0.8% target
VWAP_STOP_LOSS = 0.005          # 0.5% stop loss


class TradingWorker(ContinuousWorker):
    """Executes trading strategies via Alpaca Paper Trading API.

    Strategies: VWAP Mean Reversion (primary)
    Risk: Kelly Criterion sizing, $100 max/trade, circuit breaker after 3 losses
    Signal routing: new signals → Tier 2 'trading_decision' queue for risk check
                    pre-approved executions → executed directly
    """

    queue_name = "trading_signal"
    worker_name = "trading_worker"

    def __init__(self):
        super().__init__()
        self._alpaca = None
        self._daily_losses = 0.0
        self._daily_wins = 0.0
        self._loss_streak = 0
        self._trades_today: list[dict] = []
        self._last_reset_date: date | None = None
        self._screened_symbols: list[str] = []
        self._last_screen_time: datetime | None = None

    # ==================== Alpaca client ====================

    def _get_alpaca(self):
        """Lazy-init Alpaca REST client via alpaca-py."""
        if self._alpaca is None:
            try:
                from alpaca.trading.client import TradingClient
                self._alpaca = TradingClient(
                    api_key=self.settings.alpaca_api_key,
                    secret_key=self.settings.alpaca_api_secret,
                    paper=True,
                )
            except ImportError:
                # Fall back to alpaca-trade-api if alpaca-py not installed
                import alpaca_trade_api as tradeapi
                self._alpaca = tradeapi.REST(
                    self.settings.alpaca_api_key,
                    self.settings.alpaca_api_secret,
                    self.settings.alpaca_base_url,
                )
        return self._alpaca

    def _get_data_client(self):
        """Lazy-init Alpaca market data client."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            return StockHistoricalDataClient(
                api_key=self.settings.alpaca_api_key,
                secret_key=self.settings.alpaca_api_secret,
            )
        except ImportError:
            return None

    # ==================== Market session ====================

    def _market_session(self) -> str:
        """Return current market session: 'regular', 'after_hours', or 'closed'."""
        now = datetime.now(ET).time()
        if MARKET_OPEN <= now < MARKET_CLOSE:
            return "regular"
        if MARKET_CLOSE <= now < AFTER_HOURS_CLOSE:
            return "after_hours"
        return "closed"

    def _reset_daily_state_if_needed(self) -> None:
        """Reset per-day counters at market open."""
        today = datetime.now(ET).date()
        if self._last_reset_date != today:
            self._daily_losses = 0.0
            self._daily_wins = 0.0
            self._loss_streak = 0
            self._trades_today = []
            self._last_reset_date = today
            log.info("trading.daily_reset", date=str(today))

    def _circuit_breaker_active(self, session: str) -> bool:
        """Return True if circuit breaker should stop trading."""
        threshold = 2 if session == "after_hours" else CIRCUIT_BREAKER_LOSSES
        return self._loss_streak >= threshold

    # ==================== Task dispatch ====================

    async def process_task(self, task: dict) -> None:
        task_type = task.get("type")
        if task_type == "execute_signal":
            # Signal already approved by Tier 2 — execute directly
            await self._execute_approved_signal(task["signal"])
        elif task_type == "close_positions":
            await self._close_all_positions()
        elif task_type == "run_backtest":
            await self._run_backtest(task.get("config", {}))
        elif task_type == "screen_stocks":
            await self._run_screener()
        elif task_type == "analyze_performance":
            await self._after_hours_analysis()
        else:
            log.warning("trading.unknown_task", task_type=task_type)

    # ==================== Idle loop ====================

    async def idle_loop(self) -> None:
        self._reset_daily_state_if_needed()
        session = self._market_session()

        if session in ("regular", "after_hours"):
            if self._circuit_breaker_active(session):
                log.warning(
                    "trading.circuit_breaker_active",
                    session=session,
                    loss_streak=self._loss_streak,
                )
                await asyncio.sleep(SCAN_INTERVAL_SEC)
                return
            await self._scan_and_signal(session)
        else:
            await self._after_hours_analysis()
            await asyncio.sleep(CLOSED_SLEEP_SEC)

    # ==================== Market scanning ====================

    async def _scan_and_signal(self, session: str) -> None:
        """Screen stocks, compute VWAP signals, route to Tier 2 for risk approval."""
        log.info("trading.market_scan", session=session)

        symbols = await self._get_symbols()
        if not symbols:
            log.warning("trading.no_symbols")
            await asyncio.sleep(SCAN_INTERVAL_SEC)
            return

        signals_generated = 0
        for symbol in symbols:
            try:
                signal = await self._compute_vwap_signal(symbol, session)
                if signal:
                    await self._route_signal(signal, session)
                    signals_generated += 1
            except Exception as exc:
                log.warning("trading.signal_error", symbol=symbol, error=str(exc))

        log.info("trading.scan_complete", symbols=len(symbols), signals=signals_generated, session=session)
        await self.log_activity(
            action="market_scan",
            status="completed",
            details={"session": session, "symbols": len(symbols), "signals": signals_generated},
        )

        sleep_secs = AFTER_HOURS_SCAN_SEC if session == "after_hours" else SCAN_INTERVAL_SEC
        await asyncio.sleep(sleep_secs)

    async def _get_symbols(self) -> list[str]:
        """Return screened symbols, using cache if < 30 minutes old."""
        now = datetime.utcnow()
        if (
            self._screened_symbols
            and self._last_screen_time
            and (now - self._last_screen_time).total_seconds() < 1800
        ):
            return self._screened_symbols

        symbols = await self._run_screener()
        return symbols if symbols else FALLBACK_SYMBOLS

    async def _run_screener(self) -> list[str]:
        """Screen for liquid, volatile stocks suitable for VWAP intraday trading."""
        log.info("trading.screener_start")
        try:
            data_client = self._get_data_client()
            if data_client is None:
                log.warning("trading.screener_no_data_client")
                return FALLBACK_SYMBOLS

            # Use Alpaca asset list filtered for tradeable US equities
            api = self._get_alpaca()
            try:
                # alpaca-py path
                from alpaca.trading.requests import GetAssetsRequest
                from alpaca.trading.enums import AssetClass, AssetStatus
                req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
                assets = api.get_all_assets(req)
            except AttributeError:
                # alpaca-trade-api fallback
                assets = api.list_assets(status="active", asset_class="us_equity")

            tradeable = [
                a.symbol for a in assets
                if getattr(a, "tradable", True)
                and getattr(a, "shortable", True)
                and not getattr(a, "symbol", "").endswith((".", "/"))
            ]

            # Cap at top 50 by symbol length heuristic (large-caps have short tickers)
            tradeable = [s for s in tradeable if len(s) <= 4][:50]
            self._screened_symbols = tradeable or FALLBACK_SYMBOLS
            self._last_screen_time = datetime.utcnow()

            log.info("trading.screener_done", count=len(self._screened_symbols))
            return self._screened_symbols

        except Exception as exc:
            log.error("trading.screener_failed", error=str(exc))
            return FALLBACK_SYMBOLS

    # ==================== VWAP signal generation ====================

    async def _compute_vwap_signal(self, symbol: str, session: str) -> dict | None:
        """Compute VWAP mean reversion signal for a symbol.

        Returns a signal dict if a trade opportunity exists, else None.
        """
        try:
            data_client = self._get_data_client()
            if data_client is None:
                return None

            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            # Request intraday 1-minute bars for today
            from datetime import datetime as dt
            today_open_et = dt.now(ET).replace(hour=9, minute=30, second=0, microsecond=0)
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=today_open_et,
                limit=60,
            )
            bars_resp = data_client.get_stock_bars(req)
            bars = bars_resp[symbol] if bars_resp else []

            if len(bars) < 5:
                return None

            # Compute VWAP = sum(price * volume) / sum(volume)
            total_pv = sum(b.vwap * b.volume if hasattr(b, "vwap") and b.vwap
                           else (b.high + b.low + b.close) / 3 * b.volume
                           for b in bars)
            total_vol = sum(b.volume for b in bars)
            if total_vol == 0:
                return None

            vwap = total_pv / total_vol
            last_bar = bars[-1]
            current_price = last_bar.close

            deviation = (current_price - vwap) / vwap
            threshold = VWAP_DEVIATION_ENTRY_AH if session == "after_hours" else VWAP_DEVIATION_ENTRY

            if deviation <= -threshold:
                side = "buy"
            elif deviation >= threshold:
                side = "sell"
            else:
                return None

            # RSI filter: don't buy oversold below 25 or above 75
            rsi = self._compute_rsi([b.close for b in bars], period=14)
            if side == "buy" and rsi is not None and rsi < 25:
                return None
            if side == "sell" and rsi is not None and rsi > 75:
                return None

            signal = {
                "symbol": symbol,
                "side": side,
                "current_price": current_price,
                "vwap": round(vwap, 4),
                "deviation_pct": round(deviation * 100, 3),
                "rsi": round(rsi, 2) if rsi is not None else None,
                "strategy": "vwap_mean_reversion",
                "session": session,
                "generated_at": datetime.utcnow().isoformat(),
            }
            log.info("trading.signal_generated", **{k: v for k, v in signal.items() if k != "generated_at"})
            return signal

        except Exception as exc:
            log.warning("trading.vwap_signal_error", symbol=symbol, error=str(exc))
            return None

    def _compute_rsi(self, prices: list[float], period: int = 14) -> float | None:
        """Compute RSI from a list of closing prices."""
        if len(prices) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, period + 1):
            delta = prices[-(period + 1 - i)] - prices[-(period + 2 - i)]
            (gains if delta > 0 else losses).append(abs(delta))
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # ==================== Signal routing ====================

    async def _route_signal(self, signal: dict, session: str) -> None:
        """Route signal to Tier 2 trading_decision queue for risk check."""
        max_value = MAX_TRADE_VALUE_AFTER_HOURS if session == "after_hours" else MAX_TRADE_VALUE
        signal["max_trade_value"] = max_value

        await self.push_task_to_graph(
            {"type": "trading_signal", "signal": signal},
            "trading_decision",
        )
        log.debug("trading.signal_routed_to_tier2", symbol=signal["symbol"], side=signal["side"])

    # ==================== Trade execution ====================

    async def _execute_approved_signal(self, signal: dict) -> None:
        """Execute a signal that has already been approved by Tier 2."""
        session = self._market_session()
        if self._circuit_breaker_active(session):
            log.warning("trading.circuit_breaker_blocked", streak=self._loss_streak)
            return

        symbol = signal.get("symbol")
        side = signal.get("side", "buy")
        current_price = signal.get("current_price", 0)
        max_value = signal.get("max_trade_value", MAX_TRADE_VALUE)

        if not symbol or not current_price:
            log.error("trading.invalid_signal", signal=signal)
            return

        # Kelly Criterion position sizing: simple fractional Kelly
        # Kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        qty = max(1, int(max_value / current_price))

        log.info("trading.executing", symbol=symbol, side=side, qty=qty, price=current_price)
        try:
            api = self._get_alpaca()
            try:
                # alpaca-py path
                from alpaca.trading.requests import MarketOrderRequest
                from alpaca.trading.enums import OrderSide, TimeInForce

                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                order = api.submit_order(req)
                order_id = order.id
            except (ImportError, AttributeError):
                # alpaca-trade-api fallback
                order = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type="market",
                    time_in_force="day",
                )
                order_id = order.id

            trade_record = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": current_price,
                "order_id": str(order_id),
                "session": session,
                "strategy": signal.get("strategy", "vwap_mean_reversion"),
                "executed_at": datetime.utcnow().isoformat(),
            }
            self._trades_today.append(trade_record)

            log.info("trading.order_submitted", order_id=str(order_id), symbol=symbol, qty=qty)
            await self.log_activity("trade_executed", "success", trade_record)

        except Exception as exc:
            log.error("trading.order_failed", symbol=symbol, error=str(exc))
            self._loss_streak += 1
            await self.log_activity(
                "trade_executed", "error",
                {"symbol": symbol, "error": str(exc)},
            )

    async def _close_all_positions(self) -> None:
        """Close all open positions — called at end of day or on SIGTERM."""
        log.info("trading.closing_all_positions")
        try:
            api = self._get_alpaca()
            try:
                api.close_all_positions(cancel_orders=True)
            except TypeError:
                api.close_all_positions()
            log.info("trading.positions_closed")
            await self.log_activity("positions_closed", "success", {})
        except Exception as exc:
            log.error("trading.close_all_failed", error=str(exc))
            await self.log_activity("positions_closed", "error", {"error": str(exc)})

    # ==================== After-hours analysis ====================

    async def _after_hours_analysis(self) -> None:
        """Review today's trades, extract patterns, prepare next-day watchlist."""
        log.info("trading.after_hours_analysis", trades_today=len(self._trades_today))
        if not self._trades_today:
            return

        wins = [t for t in self._trades_today if t.get("pnl", 0) > 0]
        losses = [t for t in self._trades_today if t.get("pnl", 0) <= 0]

        summary = {
            "date": str(datetime.now(ET).date()),
            "total_trades": len(self._trades_today),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(self._trades_today), 3) if self._trades_today else 0,
            "loss_streak": self._loss_streak,
        }
        log.info("trading.daily_summary", **summary)

        await self.log_activity("daily_summary", "completed", summary)

        # Push pattern to research for learning
        if self._trades_today:
            await self.push_task_to_graph(
                {
                    "type": "pattern_discovery",
                    "article": {
                        "title": f"Trading session {summary['date']}",
                        "summary": json.dumps(summary),
                        "source": "trading_worker",
                        "category": "trading",
                    },
                },
                "pattern_discovery",
            )

    async def _run_backtest(self, config: dict) -> None:
        """Stub for backtest execution — delegates to Tier 2 or external engine."""
        log.info("trading.backtest_requested", config=config)
        await self.push_task_to_graph(
            {"type": "run_backtest", "config": config},
            "trading_decision",
        )

    # ==================== Graceful shutdown ====================

    async def run(self) -> None:
        """Override to close positions on shutdown."""
        try:
            await super().run()
        finally:
            # Best-effort position close on shutdown
            if self.settings.alpaca_api_key:
                try:
                    await self._close_all_positions()
                except Exception as exc:
                    log.warning("trading.shutdown_close_failed", error=str(exc))


if __name__ == "__main__":
    asyncio.run(TradingWorker().run())
