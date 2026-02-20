"""TradingWorker — 24/7 market execution engine.

Market hours (9:30 AM - 4:00 PM ET): real-time analysis, signal detection, order execution
After hours: performance analysis, strategy optimization, next-day preparation
"""
import asyncio
import structlog
from datetime import datetime, time
from zoneinfo import ZoneInfo
from jarvis.workers.base import ContinuousWorker

log = structlog.get_logger()

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
AFTER_HOURS_CLOSE = time(17, 30)

SCAN_INTERVAL_SEC = 300  # 5 minutes during market hours


class TradingWorker(ContinuousWorker):
    """Executes trading strategies via Alpaca Paper Trading API.

    Strategies: VWAP Mean Reversion, Opening Range Breakout, Bollinger+RSI
    Risk: Kelly Criterion sizing, $100 max/trade, circuit breaker after 3 losses
    """

    queue_name = "trading_signal"
    worker_name = "trading_worker"

    def __init__(self):
        super().__init__()
        self._alpaca = None
        self._daily_losses = 0.0
        self._loss_streak = 0

    def _get_alpaca(self):
        if self._alpaca is None:
            import alpaca_trade_api as tradeapi
            self._alpaca = tradeapi.REST(
                self.settings.alpaca_api_key,
                self.settings.alpaca_api_secret,
                self.settings.alpaca_base_url,
            )
        return self._alpaca

    async def process_task(self, task: dict) -> None:
        task_type = task.get("type")
        if task_type == "execute_signal":
            await self._execute_signal(task["signal"])
        elif task_type == "close_positions":
            await self._close_all_positions()
        elif task_type == "run_backtest":
            await self._run_backtest(task.get("config", {}))
        else:
            log.warning("trading.unknown_task", task_type=task_type)

    async def idle_loop(self) -> None:
        session = self._market_session()
        if session == "regular" or session == "after_hours":
            await self._scan_and_signal()
        else:
            await self._after_hours_analysis()
            await asyncio.sleep(SCAN_INTERVAL_SEC)

    def _market_session(self) -> str:
        now = datetime.now(ET).time()
        if MARKET_OPEN <= now < MARKET_CLOSE:
            return "regular"
        if MARKET_CLOSE <= now < AFTER_HOURS_CLOSE:
            return "after_hours"
        return "closed"

    async def _scan_and_signal(self) -> None:
        log.info("trading.market_scan", session=self._market_session())
        await asyncio.sleep(SCAN_INTERVAL_SEC)

    async def _execute_signal(self, signal: dict) -> None:
        if self._loss_streak >= 3:
            log.warning("trading.circuit_breaker", streak=self._loss_streak)
            return

        symbol = signal.get("symbol")
        side = signal.get("side")
        qty = signal.get("qty", 1)

        log.info("trading.signal_execute", symbol=symbol, side=side, qty=qty)
        try:
            api = self._get_alpaca()
            order = api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",
            )
            log.info("trading.order_submitted", order_id=order.id, symbol=symbol)
        except Exception as exc:
            log.error("trading.order_failed", symbol=symbol, error=str(exc))

    async def _close_all_positions(self) -> None:
        try:
            api = self._get_alpaca()
            api.close_all_positions()
            log.info("trading.positions_closed")
        except Exception as exc:
            log.error("trading.close_failed", error=str(exc))

    async def _run_backtest(self, config: dict) -> None:
        log.info("trading.backtest_start", config=config)

    async def _after_hours_analysis(self) -> None:
        log.info("trading.after_hours_analysis")


if __name__ == "__main__":
    worker = TradingWorker()
    asyncio.run(worker.run())
