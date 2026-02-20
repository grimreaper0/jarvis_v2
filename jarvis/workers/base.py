"""ContinuousWorker — base class for always-on Tier 1 workers.

Key contract:
- NO LangGraph imports anywhere in this file or subclasses
- Workers poll Redis queues; on idle they run their default scrape/scan loop
- supervisord restarts on crash — workers must be stateless across restarts
"""
import asyncio
import signal
import structlog
from abc import ABC, abstractmethod

log = structlog.get_logger()

IDLE_LOOP_INTERVAL_SEC = 3600  # 1 hour default idle cycle


class ContinuousWorker(ABC):
    """Base class for always-on workers that process Redis queues.

    Subclasses implement:
      process_task(task) — handle one task from the queue
      idle_loop()        — work to do when queue is empty (scraping, scanning)
    """

    queue_name: str = "default_queue"
    worker_name: str = "base_worker"

    def __init__(self):
        from config.settings import get_settings
        self.settings = get_settings()
        self._running = False
        self._redis = None

    def _get_redis(self):
        if self._redis is None:
            import redis
            self._redis = redis.from_url(self.settings.redis_url, decode_responses=True)
        return self._redis

    @abstractmethod
    async def process_task(self, task: dict) -> None:
        """Process a single task popped from the Redis queue."""

    @abstractmethod
    async def idle_loop(self) -> None:
        """Work to perform when the Redis queue is empty."""

    async def run(self) -> None:
        self._running = True
        log.info("worker.started", name=self.worker_name, queue=self.queue_name)
        self._register_signals()

        while self._running:
            task = self._pop_task()
            if task:
                try:
                    await self.process_task(task)
                except Exception as exc:
                    log.error("worker.task_error", name=self.worker_name, error=str(exc), task=task)
            else:
                try:
                    await self.idle_loop()
                except Exception as exc:
                    log.error("worker.idle_error", name=self.worker_name, error=str(exc))
                await asyncio.sleep(5)

    def _pop_task(self) -> dict | None:
        try:
            r = self._get_redis()
            raw = r.lpop(self.queue_name)
            if raw:
                import json
                return json.loads(raw)
        except Exception as exc:
            log.warning("worker.redis_pop_error", error=str(exc))
        return None

    def push_task(self, task: dict, queue: str | None = None) -> None:
        import json
        target = queue or self.queue_name
        self._get_redis().rpush(target, json.dumps(task))

    def _register_signals(self) -> None:
        def _stop(signum, frame):
            log.info("worker.stopping", name=self.worker_name, signal=signum)
            self._running = False

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
