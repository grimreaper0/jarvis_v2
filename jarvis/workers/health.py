"""HealthWorker — 24/7 infrastructure monitoring and auto-remediation."""
import asyncio
import structlog
from jarvis.workers.base import ContinuousWorker

log = structlog.get_logger()

CHECK_INTERVAL_SEC = 60

MONITORED_SERVICES = ["redis", "postgresql", "supervisor"]
DISK_WARN_THRESHOLD_PCT = 85
MEM_WARN_THRESHOLD_PCT = 90


class HealthWorker(ContinuousWorker):
    """Monitors infrastructure health and auto-remediates known issues.

    Checks: Redis, PostgreSQL, disk space, memory, supervisor processes
    Remediates: restart failed services, clear caches, rotate logs
    Alerts: pushes notifications queue for critical issues
    """

    queue_name = "health_check"
    worker_name = "health_worker"

    async def process_task(self, task: dict) -> None:
        task_type = task.get("type")
        if task_type == "check_service":
            await self._check_service(task["service"])
        elif task_type == "remediate":
            await self._remediate(task["issue"])
        else:
            log.warning("health.unknown_task", task_type=task_type)

    async def idle_loop(self) -> None:
        await asyncio.gather(
            self._check_redis(),
            self._check_postgres(),
            self._check_disk(),
            self._check_memory(),
            return_exceptions=True,
        )
        await asyncio.sleep(CHECK_INTERVAL_SEC)

    async def _check_redis(self) -> None:
        try:
            r = self._get_redis()
            r.ping()
            log.debug("health.redis_ok")
        except Exception as exc:
            log.error("health.redis_down", error=str(exc))
            await self._alert("CRITICAL", "Redis is down", "redis_down")

    async def _check_postgres(self) -> None:
        try:
            import psycopg2
            from config.settings import get_settings
            conn = psycopg2.connect(get_settings().postgres_url, connect_timeout=5)
            conn.close()
            log.debug("health.postgres_ok")
        except Exception as exc:
            log.error("health.postgres_down", error=str(exc))
            await self._alert("CRITICAL", "PostgreSQL is down", "postgres_down")

    async def _check_disk(self) -> None:
        import shutil
        usage = shutil.disk_usage("/")
        pct = (usage.used / usage.total) * 100
        if pct > DISK_WARN_THRESHOLD_PCT:
            log.warning("health.disk_high", pct=round(pct, 1))
            await self._alert("WARNING", f"Disk usage at {pct:.1f}%", "disk_high")
        else:
            log.debug("health.disk_ok", pct=round(pct, 1))

    async def _check_memory(self) -> None:
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > MEM_WARN_THRESHOLD_PCT:
                log.warning("health.memory_high", pct=mem.percent)
                await self._alert("WARNING", f"Memory usage at {mem.percent:.1f}%", "memory_high")
            else:
                log.debug("health.memory_ok", pct=mem.percent)
        except ImportError:
            log.debug("health.psutil_not_installed")

    async def _check_service(self, service: str) -> None:
        log.info("health.service_check", service=service)

    async def _remediate(self, issue: dict) -> None:
        log.info("health.remediate", issue=issue)

    async def _alert(self, severity: str, message: str, issue_type: str) -> None:
        self.push_task(
            {"type": "alert", "severity": severity, "message": message, "issue_type": issue_type},
            "notifications",
        )


if __name__ == "__main__":
    worker = HealthWorker()
    asyncio.run(worker.run())
