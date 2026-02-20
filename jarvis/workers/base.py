"""ContinuousWorker — base class for always-on Tier 1 workers.

Key contract:
- NO LangGraph imports anywhere in this file or subclasses
- Workers poll Redis queues; on idle they run their default scrape/scan loop
- supervisord restarts on crash — workers must be stateless across restarts
"""
import asyncio
import json
import signal
import structlog
from abc import ABC, abstractmethod
from datetime import datetime

import redis.asyncio as aioredis

log = structlog.get_logger()

IDLE_LOOP_INTERVAL_SEC = 3600  # 1 hour default idle cycle
HEARTBEAT_TTL_SEC = 120        # heartbeat key expires after 2 minutes
HEARTBEAT_INTERVAL_SEC = 60    # refresh heartbeat every 60 seconds


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
        self._redis: aioredis.Redis | None = None
        self._sync_redis = None
        self._db_pool = None

    # ==================== Redis (async) ====================

    async def _get_async_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                max_connections=10,
            )
        return self._redis

    def _get_redis(self):
        """Sync Redis — kept for legacy callers only. Prefer async Redis."""
        if self._sync_redis is None:
            import redis as sync_redis
            self._sync_redis = sync_redis.from_url(
                self.settings.redis_url, decode_responses=True
            )
        return self._sync_redis

    # ==================== PostgreSQL ====================

    async def _get_db(self):
        """Return an async psycopg connection (psycopg v3)."""
        import psycopg
        return await psycopg.AsyncConnection.connect(self.settings.postgres_url)

    # ==================== Fleet registry ====================

    async def announce_online(self) -> None:
        """Publish presence to the fleet registry channel."""
        r = await self._get_async_redis()
        payload = json.dumps({
            "worker": self.worker_name,
            "queue": self.queue_name,
            "status": "online",
            "ts": datetime.utcnow().isoformat(),
        })
        await r.publish("fleet:announcements", payload)
        log.info("worker.online", name=self.worker_name)

    async def announce_offline(self) -> None:
        """Publish departure to the fleet registry channel."""
        r = await self._get_async_redis()
        payload = json.dumps({
            "worker": self.worker_name,
            "queue": self.queue_name,
            "status": "offline",
            "ts": datetime.utcnow().isoformat(),
        })
        await r.publish("fleet:announcements", payload)
        log.info("worker.offline", name=self.worker_name)

    async def heartbeat(self) -> None:
        """Set a TTL key so fleet dashboard knows this worker is alive."""
        r = await self._get_async_redis()
        key = f"heartbeat:{self.worker_name}"
        await r.setex(key, HEARTBEAT_TTL_SEC, datetime.utcnow().isoformat())

    async def _heartbeat_loop(self) -> None:
        """Background task: refresh heartbeat TTL every 60 seconds."""
        while self._running:
            try:
                await self.heartbeat()
            except Exception as exc:
                log.warning("worker.heartbeat_error", name=self.worker_name, error=str(exc))
            await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)

    # ==================== Tier 2 task dispatch ====================

    async def push_task_to_graph(self, task: dict, graph_name: str) -> None:
        """Push a task to a Tier 2 LangGraph worker queue.

        Tier 2 workers watch Redis queues named after their graph (e.g.
        "revenue_opportunity", "trading_decision", "content_gate").
        """
        r = await self._get_async_redis()
        await r.rpush(graph_name, json.dumps(task))
        log.debug("worker.pushed_to_graph", graph=graph_name, task_type=task.get("type"))

    # ==================== Activity logging ====================

    async def log_activity(
        self,
        action: str,
        status: str,
        details: dict | None = None,
    ) -> None:
        """Write one row to the bot_activity table in PostgreSQL."""
        try:
            async with await self._get_db() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO bot_activity
                            (bot_name, activity_type, status, details_json, created_at)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            self.worker_name,
                            action,
                            status,
                            json.dumps(details or {}),
                            datetime.utcnow(),
                        ),
                    )
                    await conn.commit()
        except Exception as exc:
            log.warning(
                "worker.log_activity_failed",
                name=self.worker_name,
                action=action,
                error=str(exc),
            )

    # ==================== Core run loop ====================

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

        await self.announce_online()
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            while self._running:
                task = await self._pop_task()
                if task:
                    try:
                        await self.process_task(task)
                    except Exception as exc:
                        log.error(
                            "worker.task_error",
                            name=self.worker_name,
                            error=str(exc),
                            task=task,
                        )
                        await self.log_activity(
                            action="task_error",
                            status="error",
                            details={"error": str(exc), "task": task},
                        )
                else:
                    try:
                        await self.idle_loop()
                    except Exception as exc:
                        log.error(
                            "worker.idle_error",
                            name=self.worker_name,
                            error=str(exc),
                        )
                    await asyncio.sleep(5)
        finally:
            self._running = False
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            await self.announce_offline()
            if self._redis:
                await self._redis.aclose()

    async def _pop_task(self) -> dict | None:
        try:
            r = await self._get_async_redis()
            raw = await r.lpop(self.queue_name)
            if raw:
                return json.loads(raw)
        except Exception as exc:
            log.warning("worker.redis_pop_error", error=str(exc))
        return None

    async def push_task_async(self, task: dict, queue: str | None = None) -> None:
        """Async version of push_task."""
        r = await self._get_async_redis()
        target = queue or self.queue_name
        await r.rpush(target, json.dumps(task))

    def push_task(self, task: dict, queue: str | None = None) -> None:
        """Sync push — available for legacy / non-async callers."""
        target = queue or self.queue_name
        self._get_redis().rpush(target, json.dumps(task))

    def _register_signals(self) -> None:
        def _stop(signum, frame):
            log.info("worker.stopping", name=self.worker_name, signal=signum)
            self._running = False

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
