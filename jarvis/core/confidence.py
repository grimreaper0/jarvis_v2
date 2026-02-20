"""ConfidenceGate v2 — 4-layer weighted confidence scoring (async).

Ported from jarvis_v1 utils/confidence_gate.py with these changes:
- evaluate() is now async (uses async AutoMem v2)
- Same 4-layer weights as production: base=0.25, validation=0.25, historical=0.30, reflexive=0.20
- Same thresholds: >=0.90 execute, 0.60-0.89 delegate, <0.60 clarify
- Tree of Thoughts: stub (not yet implemented in v2), no import error
- All scoring logic is identical to production — it is proven in production

Layer definitions:
  Base (0.25)       — Task clarity: recognized type, substantive description, context present
  Validation (0.25) — Domain checks: API available, within limits, data freshness
  Historical (0.30) — AutoMem hybrid search: golden rules + similar patterns
  Reflexive (0.20)  — Self-assessment: score coherence, consensus bonus/penalty
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import structlog

log = structlog.get_logger()

# Layer weights — identical to jarvis_v1 production
WEIGHT_BASE = 0.25
WEIGHT_VALIDATION = 0.25
WEIGHT_HISTORICAL = 0.30
WEIGHT_REFLEXIVE = 0.20

# Decision thresholds
THRESHOLD_EXECUTE = 0.90
THRESHOLD_DELEGATE = 0.60

# Historical layer constants
GOLDEN_RULE_BOOST = 0.15
HISTORICAL_BASELINE = 0.3

# Recognized task types
KNOWN_TASK_TYPES = {
    "content_post", "content_story", "content_reel", "content_video",
    "trade_stock", "trade_crypto", "hashtag_research", "seo_optimization",
    "trend_analysis", "revenue_analysis", "pattern_extraction",
    "backtest", "portfolio_rebalance",
}


@dataclass
class ConfidenceResult:
    """Complete result of a confidence gate evaluation."""
    base_score: float
    validation_score: float
    historical_score: float
    reflexive_score: float
    final_confidence: float
    decision: str  # "execute" | "delegate" | "clarify"
    golden_rules_applied: list[str] = field(default_factory=list)
    guardrail_flags: list[str] = field(default_factory=list)
    reasoning: str = ""
    audit_id: Optional[str] = None
    tot_validated: bool = False
    tot_metadata: Optional[dict[str, Any]] = None

    # Convenience alias so code expecting final_score also works
    @property
    def final_score(self) -> float:
        return self.final_confidence


class ConfidenceGate:
    """4-layer async confidence gate for autonomous decision-making.

    Usage::

        gate = ConfidenceGate(bot_name="trading_bot", memory=automem)
        result = await gate.evaluate(
            task_type="trade_stock",
            task_description="VWAP mean-reversion entry on NVDA",
            context={"api_available": True, "within_limits": True},
        )
        if result.decision == "execute":
            ...
    """

    def __init__(
        self,
        bot_name: str = "jarvis",
        memory=None,
        automem=None,
        settings=None,
        enable_tot: bool = True,
    ):
        """
        Args:
            bot_name: Bot identifier used in audit logging.
            memory: AutoMem v2 instance. Created lazily if None.
            automem: Alias for memory (jarvis_v1 compat).
            settings: Settings instance. Loaded from config if None.
            enable_tot: Enable Tree of Thoughts for edge cases (stub in v2).
        """
        self.bot_name = bot_name
        self._memory = memory or automem
        self.enable_tot = enable_tot
        from config.settings import get_settings
        self.settings = settings or get_settings()

    @property
    def memory(self):
        if self._memory is None:
            from jarvis.core.memory import AutoMem
            self._memory = AutoMem()
        return self._memory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        task_type: str,
        task_description: str,
        context: dict[str, Any] | None = None,
        validation_checks: list[Callable] | None = None,
        task_embedding: list[float] | None = None,
    ) -> ConfidenceResult:
        """Run the full 4-layer confidence evaluation.

        Args:
            task_type: Recognized type (e.g. "trade_stock", "content_post").
            task_description: Human-readable description of the task.
            context: Domain context (api_available, within_limits, etc.).
            validation_checks: Optional callables returning (bool, str) or bool.
            task_embedding: Pre-computed embedding for historical layer lookup.

        Returns:
            ConfidenceResult with scores, decision, and audit_id.
        """
        context = context or {}

        base_score = self._calculate_base_score(task_type, task_description, context)
        validation_score = self._calculate_validation_score(task_type, context, validation_checks)

        golden_rules_applied: list[str] = []
        historical_score = await self._calculate_historical_score(
            task_type, task_embedding, golden_rules_applied, task_description
        )

        reflexive_score = self._calculate_reflexive_score(
            base_score, validation_score, historical_score
        )

        final = (
            WEIGHT_BASE * base_score
            + WEIGHT_VALIDATION * validation_score
            + WEIGHT_HISTORICAL * historical_score
            + WEIGHT_REFLEXIVE * reflexive_score
        )
        final = max(0.0, min(1.0, final))

        guardrail_flags = await self._check_guardrails(context)
        decision = self._determine_decision(final, guardrail_flags)
        reasoning = self._build_reasoning(
            task_type, base_score, validation_score,
            historical_score, reflexive_score, final,
            decision, golden_rules_applied, guardrail_flags,
        )

        audit_id = await self._log_audit(
            task_type, base_score, validation_score,
            historical_score, reflexive_score, final,
            decision, golden_rules_applied, guardrail_flags,
            reasoning, context,
        )

        result = ConfidenceResult(
            base_score=round(base_score, 4),
            validation_score=round(validation_score, 4),
            historical_score=round(historical_score, 4),
            reflexive_score=round(reflexive_score, 4),
            final_confidence=round(final, 4),
            decision=decision,
            golden_rules_applied=golden_rules_applied,
            guardrail_flags=guardrail_flags,
            reasoning=reasoning,
            audit_id=audit_id,
            tot_validated=False,
            tot_metadata=None,
        )

        # Tree of Thoughts edge-case validation (stub in v2 — no-op)
        if self.enable_tot and 0.75 <= final <= 0.89:
            log.info(
                "confidence.tot_edge_case",
                final=final,
                note="ToT stub — not yet implemented in v2",
            )
            # When ToT is implemented: call validator, possibly flip decision
            result.tot_validated = False

        log.info(
            "confidence.evaluated",
            bot=self.bot_name,
            task=task_type,
            final=result.final_confidence,
            decision=decision,
        )
        return result

    async def record_outcome(
        self,
        audit_id: str,
        outcome: str,
        revenue_impact: float | None = None,
    ) -> None:
        """Close the feedback loop on a confidence decision."""
        await self.memory.update_audit_outcome(audit_id, outcome, revenue_impact)

    # ------------------------------------------------------------------
    # Layer calculations
    # ------------------------------------------------------------------

    def _calculate_base_score(
        self,
        task_type: str,
        task_description: str,
        context: dict[str, Any],
    ) -> float:
        """Base layer: task clarity (0.35 + 0.35 + 0.30 = 1.0 max)."""
        score = 0.0
        if task_type in KNOWN_TASK_TYPES:
            score += 0.35
        if len(task_description.strip()) >= 20:
            score += 0.35
        if context and len(context) > 0:
            score += 0.30
        return min(1.0, score)

    def _calculate_validation_score(
        self,
        task_type: str,
        context: dict[str, Any],
        validation_checks: list[Callable] | None,
    ) -> float:
        """Validation layer: domain-specific checks."""
        if validation_checks:
            passed = 0
            total = len(validation_checks)
            for check in validation_checks:
                try:
                    result = check()
                    if isinstance(result, tuple):
                        success, _ = result
                    else:
                        success = bool(result)
                    if success:
                        passed += 1
                except Exception as exc:
                    log.warning("validation_check.failed", error=str(exc))
            return passed / total if total > 0 else 0.5

        # Default heuristics
        score = 0.5
        if context.get("api_available", False):
            score += 0.2
        if context.get("data_fresh", False):
            score += 0.15
        if context.get("within_limits", True):
            score += 0.15
        else:
            score -= 0.3
        return max(0.0, min(1.0, score))

    async def _calculate_historical_score(
        self,
        task_type: str,
        task_embedding: list[float] | None,
        golden_rules_applied: list[str],
        task_description: str = "",
    ) -> float:
        """Historical layer: AutoMem hybrid search for golden rules and patterns."""
        score = HISTORICAL_BASELINE

        if task_embedding is None:
            return score

        try:
            if task_description:
                matching_rules = await self.memory.hybrid_search_patterns(
                    query=task_description,
                    embedding=task_embedding,
                    pattern_type=task_type,
                    min_confidence=0.7,
                    limit=5,
                    vector_weight=0.7,
                    keyword_weight=0.3,
                )
                matching_rules = [r for r in matching_rules if r.get("is_golden_rule")]
            else:
                matching_rules = await self.memory.get_golden_rules_for_task(
                    task_embedding=task_embedding,
                    task_type=task_type,
                    min_similarity=0.6,
                )

            if matching_rules:
                score = min(1.0, score + GOLDEN_RULE_BOOST * len(matching_rules))
                for rule in matching_rules:
                    golden_rules_applied.append(rule["id"])
                best = matching_rules[0]
                log.info(
                    "historical.golden_rules_matched",
                    count=len(matching_rules),
                    best_hybrid=best.get("hybrid_score"),
                    best_similarity=best.get("similarity"),
                )

            # Regular pattern boost
            if task_description:
                similar = await self.memory.hybrid_search_patterns(
                    query=task_description,
                    embedding=task_embedding,
                    pattern_type=task_type,
                    min_confidence=0.7,
                    limit=5,
                )
            else:
                similar = await self.memory.find_similar_patterns(
                    embedding=task_embedding,
                    pattern_type=task_type,
                    min_confidence=0.7,
                    limit=5,
                )
            if similar:
                pattern_boost = min(0.2, 0.05 * len(similar))
                score = min(1.0, score + pattern_boost)

        except Exception as exc:
            log.warning("historical.score_failed", error=str(exc))

        return min(1.0, score)

    def _calculate_reflexive_score(
        self,
        base_score: float,
        validation_score: float,
        historical_score: float,
    ) -> float:
        """Reflexive layer: score coherence self-assessment."""
        scores = [base_score, validation_score, historical_score]
        spread = max(scores) - min(scores)
        avg = sum(scores) / len(scores)

        if spread <= 0.15:
            return min(1.0, avg + 0.1)   # high agreement bonus
        elif spread <= 0.30:
            return avg                    # moderate — neutral
        else:
            return max(0.0, avg - 0.15)  # high divergence penalty

    # ------------------------------------------------------------------
    # Decision & guardrails
    # ------------------------------------------------------------------

    def _determine_decision(
        self,
        final_confidence: float,
        guardrail_flags: list[str],
    ) -> str:
        """Apply guardrail overrides then threshold decision."""
        has_financial_flag = any(
            flag in ("max_trade_amount", "daily_spend_limit", "max_monthly_api_cost")
            for flag in guardrail_flags
        )
        if has_financial_flag:
            return "delegate"

        execute_threshold = getattr(self.settings, "confidence_execute", THRESHOLD_EXECUTE)
        delegate_threshold = getattr(self.settings, "confidence_delegate", THRESHOLD_DELEGATE)

        if final_confidence >= execute_threshold:
            return "execute"
        elif final_confidence >= delegate_threshold:
            return "delegate"
        else:
            return "clarify"

    async def _check_guardrails(self, context: dict[str, Any]) -> list[str]:
        """Run configured guardrails and return list of triggered flag names."""
        triggered: list[str] = []
        try:
            guardrails = await self.memory.get_active_guardrails(bot_name=self.bot_name)
            for g in guardrails:
                name = g["name"]
                cfg = g.get("config") or {}
                limit = cfg.get("limit")
                if limit is None:
                    continue
                key = name.replace("max_", "").replace("daily_", "")
                val = context.get(key) or context.get(name)
                if val is not None and isinstance(val, (int, float)) and val > limit:
                    triggered.append(name)
        except Exception as exc:
            log.warning("guardrails.check_failed", error=str(exc))
        return triggered

    # ------------------------------------------------------------------
    # Audit & reasoning
    # ------------------------------------------------------------------

    async def _log_audit(
        self,
        task_type: str,
        base_score: float,
        validation_score: float,
        historical_score: float,
        reflexive_score: float,
        final_confidence: float,
        decision: str,
        golden_rules_applied: list[str],
        guardrail_flags: list[str],
        reasoning: str,
        context: dict[str, Any],
    ) -> str | None:
        try:
            return await self.memory.log_confidence_decision(
                bot_name=self.bot_name,
                task_type=task_type,
                base_score=base_score,
                validation_score=validation_score,
                historical_score=historical_score,
                reflexive_score=reflexive_score,
                final_confidence=final_confidence,
                decision=decision,
                golden_rules_applied=golden_rules_applied,
                guardrail_flags=guardrail_flags,
                reasoning=reasoning,
                context=context,
            )
        except Exception as exc:
            log.error("audit.log_failed", error=str(exc))
            return None

    def _build_reasoning(
        self,
        task_type: str,
        base_score: float,
        validation_score: float,
        historical_score: float,
        reflexive_score: float,
        final_confidence: float,
        decision: str,
        golden_rules_applied: list[str],
        guardrail_flags: list[str],
    ) -> str:
        parts = [
            f"Task type: {task_type}",
            (
                f"Scores: base={base_score:.3f}, validation={validation_score:.3f}, "
                f"historical={historical_score:.3f}, reflexive={reflexive_score:.3f}"
            ),
            f"Final confidence: {final_confidence:.3f} -> {decision}",
        ]
        if golden_rules_applied:
            parts.append(f"Golden rules applied: {len(golden_rules_applied)}")
        if guardrail_flags:
            parts.append(f"Guardrails triggered: {guardrail_flags}")
        return " | ".join(parts)
