"""HealthWorker — 24/7 infrastructure monitoring and auto-remediation.

Queue:  health_check
Idle:   full check cycle every 5 minutes (CHECK_INTERVAL_SEC)
Checks: Redis, PostgreSQL, disk space, memory, LLM inference (mlx_lm), supervisor processes
Remediates: restart failed supervisor processes, purge Redis, rotate old logs
Alerts: pushes to 'notifications' queue + structlog
"""
import asyncio
import json
import shutil
import subprocess
import structlog
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from jarvis.workers.base import ContinuousWorker

log = structlog.get_logger()

CHECK_INTERVAL_SEC = 300        # 5 minutes between full check cycles
DISK_WARN_THRESHOLD_PCT = 80    # warn above 80% used
DISK_CRIT_THRESHOLD_PCT = 90    # critical above 90% used
MEM_WARN_THRESHOLD_PCT = 80
MEM_CRIT_THRESHOLD_PCT = 90
MEM_RUNAWAY_MB = 4096           # kill processes using more than this
REDIS_MEM_THRESHOLD_PCT = 80
POSTGRES_CONN_WARN_PCT = 75     # % of max_connections
LOG_ROTATE_DAYS = 7             # compress logs older than this many days


@dataclass
class CheckResult:
    component: str
    status: str             # healthy | degraded | critical
    message: str
    metrics: dict
    timestamp: str
    remediation_attempted: bool = False
    remediation_success: bool = False
    remediation_message: str = ""


class HealthWorker(ContinuousWorker):
    """Infrastructure health monitoring and auto-remediation.

    Runs a full check cycle during idle_loop(), sleeping CHECK_INTERVAL_SEC
    between cycles. Individual on-demand checks are dispatched via process_task().
    """

    queue_name = "health_check"
    worker_name = "health_worker"

    def __init__(self):
        super().__init__()
        self._project_root = Path(__file__).parent.parent.parent
        self._log_dir = self._project_root / "logs"
        self._backup_dir = Path.home() / "backups" / "automem"

    # ==================== Task dispatch ====================

    async def process_task(self, task: dict) -> None:
        task_type = task.get("type")
        if task_type == "check_service":
            result = await self._check_by_name(task.get("service", ""))
            if result:
                await self._handle_result(result)
        elif task_type == "remediate":
            await self._remediate_by_name(task.get("issue", ""))
        else:
            log.warning("health.unknown_task", task_type=task_type)

    # ==================== Idle loop ====================

    async def idle_loop(self) -> None:
        log.info("health.cycle_start")
        results = await self._run_all_checks()
        await self._remediate_issues(results)
        await self._send_alerts(results)
        await self._log_results(results)

        healthy = sum(1 for r in results if r.status == "healthy")
        degraded = sum(1 for r in results if r.status == "degraded")
        critical = sum(1 for r in results if r.status == "critical")

        log.info(
            "health.cycle_done",
            healthy=healthy,
            degraded=degraded,
            critical=critical,
        )
        await asyncio.sleep(CHECK_INTERVAL_SEC)

    # ==================== Individual checks ====================

    async def _run_all_checks(self) -> list[CheckResult]:
        checks = await asyncio.gather(
            self._check_redis(),
            self._check_postgres(),
            self._check_neo4j(),
            self._check_disk(),
            self._check_memory(),
            self._check_llm_inference(),
            self._check_supervisor(),
            return_exceptions=False,
        )
        return list(checks)

    async def _check_by_name(self, name: str) -> CheckResult | None:
        dispatch = {
            "redis": self._check_redis,
            "postgresql": self._check_postgres,
            "postgres": self._check_postgres,
            "disk": self._check_disk,
            "memory": self._check_memory,
            "neo4j": self._check_neo4j,
            "llm": self._check_llm_inference,
            "llm_inference": self._check_llm_inference,
            "supervisor": self._check_supervisor,
        }
        fn = dispatch.get(name)
        return await fn() if fn else None

    async def _check_redis(self) -> CheckResult:
        ts = datetime.utcnow().isoformat()
        try:
            r = await self._get_async_redis()
            await r.ping()
            info = await r.info("memory")
            used_mb = info["used_memory"] / (1024 * 1024)
            max_bytes = info.get("maxmemory", 0)
            mem_pct = (info["used_memory"] / max_bytes * 100) if max_bytes > 0 else 0
            key_count = await r.dbsize()

            metrics = {
                "used_mb": round(used_mb, 2),
                "mem_pct": round(mem_pct, 2),
                "key_count": key_count,
            }
            if mem_pct >= REDIS_MEM_THRESHOLD_PCT:
                return CheckResult("redis", "critical", f"Memory at {mem_pct:.1f}%", metrics, ts)
            elif mem_pct >= REDIS_MEM_THRESHOLD_PCT * 0.75:
                return CheckResult("redis", "degraded", f"Memory elevated at {mem_pct:.1f}%", metrics, ts)
            return CheckResult("redis", "healthy", f"OK ({key_count} keys, {used_mb:.1f}MB)", metrics, ts)
        except Exception as exc:
            log.error("health.redis_check_failed", error=str(exc))
            return CheckResult("redis", "critical", f"Connection failed: {exc}", {}, ts)

    async def _check_postgres(self) -> CheckResult:
        ts = datetime.utcnow().isoformat()
        try:
            import psycopg
            async with await psycopg.AsyncConnection.connect(self.settings.postgres_url) as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT
                          (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') max_conn,
                          (SELECT count(*) FROM pg_stat_activity) active_conn,
                          (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') running,
                          (SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock') locks
                    """)
                    row = await cur.fetchone()
                    max_conn, active_conn, running, locks = row

                    await cur.execute(
                        "SELECT pg_database_size(%s) / (1024*1024)",
                        (self.settings.postgres_url.rsplit("/", 1)[-1],),
                    )
                    db_mb = (await cur.fetchone())[0]

            conn_pct = (active_conn / max_conn) * 100
            metrics = {
                "max_conn": max_conn,
                "active_conn": active_conn,
                "conn_pct": round(conn_pct, 1),
                "running_queries": running,
                "waiting_locks": locks,
                "db_size_mb": round(db_mb, 1),
            }
            if conn_pct >= POSTGRES_CONN_WARN_PCT * 1.25 or locks > 5:
                return CheckResult("postgres", "critical",
                                   f"Stressed: {conn_pct:.1f}% conn, {locks} locks", metrics, ts)
            elif conn_pct >= POSTGRES_CONN_WARN_PCT:
                return CheckResult("postgres", "degraded",
                                   f"Load elevated: {conn_pct:.1f}% connections", metrics, ts)
            return CheckResult("postgres", "healthy",
                               f"OK ({active_conn}/{max_conn} connections)", metrics, ts)
        except Exception as exc:
            log.error("health.postgres_check_failed", error=str(exc))
            return CheckResult("postgres", "critical", f"Check failed: {exc}", {}, ts)

    async def _check_disk(self) -> CheckResult:
        ts = datetime.utcnow().isoformat()
        try:
            usage = shutil.disk_usage("/")
            used_pct = (usage.used / usage.total) * 100
            free_gb = usage.free / (1024 ** 3)

            # Log sizes of key directories
            log_mb = self._dir_size_mb(self._log_dir)
            backup_mb = self._dir_size_mb(self._backup_dir)

            metrics = {
                "total_gb": round(usage.total / (1024 ** 3), 2),
                "used_pct": round(used_pct, 1),
                "free_gb": round(free_gb, 2),
                "log_dir_mb": log_mb,
                "backup_dir_mb": backup_mb,
            }
            if used_pct >= DISK_CRIT_THRESHOLD_PCT:
                return CheckResult("disk", "critical", f"Disk {used_pct:.1f}% used", metrics, ts)
            elif used_pct >= DISK_WARN_THRESHOLD_PCT:
                return CheckResult("disk", "degraded", f"Disk {used_pct:.1f}% used", metrics, ts)
            return CheckResult("disk", "healthy", f"OK ({free_gb:.1f}GB free)", metrics, ts)
        except Exception as exc:
            log.error("health.disk_check_failed", error=str(exc))
            return CheckResult("disk", "critical", f"Check failed: {exc}", {}, ts)

    async def _check_memory(self) -> CheckResult:
        ts = datetime.utcnow().isoformat()
        try:
            import psutil
            mem = psutil.virtual_memory()
            pct = mem.percent

            high_procs = []
            for proc in psutil.process_iter(["pid", "name", "memory_info"]):
                try:
                    mb = proc.info["memory_info"].rss / (1024 * 1024)
                    if mb > 2048:
                        high_procs.append({"pid": proc.info["pid"], "name": proc.info["name"], "mb": round(mb, 1)})
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            runaway = [p for p in high_procs if p["mb"] > MEM_RUNAWAY_MB]
            metrics = {
                "used_pct": round(pct, 1),
                "available_gb": round(mem.available / (1024 ** 3), 2),
                "high_mem_procs": high_procs,
                "runaway_procs": runaway,
            }
            if runaway or pct >= MEM_CRIT_THRESHOLD_PCT:
                return CheckResult("memory", "critical",
                                   f"Memory {pct:.1f}% used, {len(runaway)} runaway procs", metrics, ts)
            elif pct >= MEM_WARN_THRESHOLD_PCT:
                return CheckResult("memory", "degraded", f"Memory {pct:.1f}% used", metrics, ts)
            return CheckResult("memory", "healthy", f"OK ({pct:.1f}% used)", metrics, ts)
        except ImportError:
            return CheckResult("memory", "healthy", "psutil not installed — skipping", {}, ts)
        except Exception as exc:
            log.error("health.memory_check_failed", error=str(exc))
            return CheckResult("memory", "critical", f"Check failed: {exc}", {}, ts)

    async def _check_neo4j(self) -> CheckResult:
        """Verify Neo4j knowledge graph is responding."""
        ts = datetime.utcnow().isoformat()
        try:
            from jarvis.core.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            await kg.connect()
            stats = await kg.get_graph_stats()
            await kg.close()
            total_nodes = sum(stats["nodes"].values())
            total_rels = sum(stats["relationships"].values())
            metrics = {"total_nodes": total_nodes, "total_relationships": total_rels}
            return CheckResult("neo4j", "healthy", f"OK ({total_nodes} nodes, {total_rels} rels)", metrics, ts)
        except Exception as exc:
            log.error("health.neo4j_check_failed", error=str(exc))
            return CheckResult("neo4j", "critical", f"Neo4j unreachable: {exc}", {}, ts)

    async def _check_llm_inference(self) -> CheckResult:
        """Verify local LLM inference (mlx_lm on port 8001) is responding."""
        ts = datetime.utcnow().isoformat()
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.settings.vllm_local_base_url}/models")
                resp.raise_for_status()
                data = resp.json()
                models = data.get("data", [])
            model_ids = [m.get("id", "?") for m in models]
            metrics = {"model_count": len(models), "models": model_ids}
            return CheckResult("llm_inference", "healthy", f"OK ({', '.join(model_ids)})", metrics, ts)
        except Exception as exc:
            log.error("health.llm_check_failed", error=str(exc))
            return CheckResult("llm_inference", "critical", f"LLM inference unreachable: {exc}", {}, ts)

    async def _check_supervisor(self) -> CheckResult:
        """Check supervisor-managed processes via supervisorctl."""
        ts = datetime.utcnow().isoformat()
        try:
            result = subprocess.run(
                ["supervisorctl", "status"],
                capture_output=True, text=True, timeout=10,
            )
            processes = []
            failed = []
            for line in result.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) < 2:
                    continue
                name, status = parts[0], parts[1]
                processes.append({"name": name, "status": status})
                if status not in ("RUNNING", "STARTING"):
                    failed.append(name)

            metrics = {"total": len(processes), "failed": failed, "processes": processes}
            if failed:
                return CheckResult("supervisor", "degraded",
                                   f"Down: {', '.join(failed)}", metrics, ts)
            return CheckResult("supervisor", "healthy", f"All {len(processes)} procs running", metrics, ts)
        except FileNotFoundError:
            return CheckResult("supervisor", "degraded", "supervisorctl not found", {}, ts)
        except subprocess.TimeoutExpired:
            return CheckResult("supervisor", "critical", "supervisorctl timeout", {}, ts)
        except Exception as exc:
            log.error("health.supervisor_check_failed", error=str(exc))
            return CheckResult("supervisor", "critical", f"Check failed: {exc}", {}, ts)

    # ==================== Auto-remediation ====================

    async def _remediate_issues(self, results: list[CheckResult]) -> None:
        for result in results:
            if result.status in ("degraded", "critical"):
                if result.component == "redis":
                    await self._remediate_redis(result)
                elif result.component == "disk":
                    await self._remediate_disk(result)
                elif result.component == "memory":
                    await self._remediate_memory(result)
                elif result.component == "supervisor":
                    await self._remediate_supervisor(result)

    async def _remediate_by_name(self, issue: str) -> None:
        dispatch = {
            "redis": self._remediate_redis,
            "disk": self._remediate_disk,
            "memory": self._remediate_memory,
            "supervisor": self._remediate_supervisor,
        }
        fn = dispatch.get(issue)
        if fn:
            dummy = CheckResult(issue, "degraded", "manual trigger", {}, datetime.utcnow().isoformat())
            await fn(dummy)

    async def _remediate_redis(self, result: CheckResult) -> None:
        try:
            r = await self._get_async_redis()
            await r.execute_command("MEMORY", "PURGE")
            result.remediation_attempted = True
            result.remediation_success = True
            result.remediation_message = "Purged expired Redis keys"
            log.info("health.redis_remediated")
        except Exception as exc:
            result.remediation_attempted = True
            result.remediation_success = False
            result.remediation_message = f"Redis remediation failed: {exc}"
            log.error("health.redis_remediation_failed", error=str(exc))

    async def _remediate_disk(self, result: CheckResult) -> None:
        try:
            cutoff = datetime.now() - timedelta(days=LOG_ROTATE_DAYS)
            rotated = 0
            if self._log_dir.exists():
                for log_file in self._log_dir.rglob("*.log"):
                    if log_file.stat().st_mtime < cutoff.timestamp():
                        subprocess.run(["gzip", str(log_file)], check=True, timeout=30)
                        rotated += 1

            # Remove old backups (> 14 days)
            removed_backups = 0
            if self._backup_dir.exists():
                cutoff14 = datetime.now() - timedelta(days=14)
                for f in self._backup_dir.glob("*.sql.gz"):
                    if f.stat().st_mtime < cutoff14.timestamp():
                        f.unlink()
                        removed_backups += 1

            result.remediation_attempted = True
            result.remediation_success = True
            result.remediation_message = f"Rotated {rotated} logs, removed {removed_backups} old backups"
            log.info("health.disk_remediated", rotated=rotated, backups_removed=removed_backups)
        except Exception as exc:
            result.remediation_attempted = True
            result.remediation_success = False
            result.remediation_message = f"Disk remediation failed: {exc}"
            log.error("health.disk_remediation_failed", error=str(exc))

    async def _remediate_memory(self, result: CheckResult) -> None:
        try:
            import psutil
            SAFE_TO_KILL = frozenset()   # nothing auto-killed without explicit allowlist
            runaway = result.metrics.get("runaway_procs", [])
            killed = []
            for proc_info in runaway:
                if proc_info["name"] in SAFE_TO_KILL:
                    try:
                        p = psutil.Process(proc_info["pid"])
                        p.terminate()
                        p.wait(timeout=5)
                        killed.append(proc_info["pid"])
                    except Exception:
                        pass

            result.remediation_attempted = True
            result.remediation_success = bool(killed)
            result.remediation_message = f"Terminated {len(killed)} runaway procs (PIDs: {killed})"
            log.info("health.memory_remediated", killed=killed)
        except ImportError:
            result.remediation_attempted = False

    async def _remediate_supervisor(self, result: CheckResult) -> None:
        failed = result.metrics.get("failed", [])
        restarted = []
        for proc_name in failed:
            try:
                r = subprocess.run(
                    ["supervisorctl", "restart", proc_name],
                    capture_output=True, text=True, timeout=15,
                )
                if r.returncode == 0:
                    restarted.append(proc_name)
                    log.info("health.supervisor_restarted", proc=proc_name)
            except Exception as exc:
                log.error("health.supervisor_restart_failed", proc=proc_name, error=str(exc))

        result.remediation_attempted = True
        result.remediation_success = bool(restarted)
        result.remediation_message = f"Restarted: {restarted}"

    # ==================== Alerting + logging ====================

    async def _handle_result(self, result: CheckResult) -> None:
        """Push critical alerts to notifications queue."""
        if result.status == "critical":
            await self._alert("CRITICAL", result.component, result.message)

    async def _send_alerts(self, results: list[CheckResult]) -> None:
        for result in results:
            if result.status == "critical":
                await self._alert("CRITICAL", result.component, result.message)
            elif result.status == "degraded":
                log.warning("health.degraded", component=result.component, message=result.message)

    async def _alert(self, severity: str, component: str, message: str) -> None:
        """Push to notifications queue — picked up by voice notifier."""
        await self.push_task_async(
            {
                "type": "alert",
                "severity": severity,
                "component": component,
                "message": message,
                "ts": datetime.utcnow().isoformat(),
            },
            "notifications",
        )
        log.warning("health.alert_sent", severity=severity, component=component, message=message)

    async def _log_results(self, results: list[CheckResult]) -> None:
        for result in results:
            await self.log_activity(
                action="health_check",
                status=result.status,
                details=asdict(result),
            )

    # ==================== Helpers ====================

    def _dir_size_mb(self, path: Path) -> float:
        if not path.exists():
            return 0.0
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / (1024 * 1024), 2)


if __name__ == "__main__":
    asyncio.run(HealthWorker().run())
