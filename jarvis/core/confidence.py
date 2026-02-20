"""ConfidenceGate — 4-layer weighted scoring for decision validation."""
import structlog
from dataclasses import dataclass

log = structlog.get_logger()

WEIGHT_BASE = 0.4
WEIGHT_VALIDATION = 0.2
WEIGHT_HISTORICAL = 0.3
WEIGHT_REFLEXIVE = 0.1


@dataclass
class ConfidenceResult:
    base_score: float
    validation_score: float
    historical_score: float
    reflexive_score: float
    final_score: float
    decision: str  # "execute" | "delegate" | "clarify"
    reasoning: str = ""


class ConfidenceGate:
    """4-layer weighted confidence scoring.

    Layers:
      base        (0.4) — LLM self-assessment of the decision
      validation  (0.2) — Data quality and completeness checks
      historical  (0.3) — Pattern match against past outcomes (AutoMem)
      reflexive   (0.1) — Second LLM pass reviewing the first response

    Thresholds (from .env):
      >= CONFIDENCE_EXECUTE  → execute autonomously
      >= CONFIDENCE_DELEGATE → delegate to human / lower-tier bot
      <  CONFIDENCE_DELEGATE → clarify / abort
    """

    def __init__(self, memory=None, settings=None):
        from config.settings import get_settings
        self.settings = settings or get_settings()
        self.memory = memory

    def evaluate(
        self,
        base: float,
        validation: float,
        historical: float,
        reflexive: float,
        context: str = "",
    ) -> ConfidenceResult:
        final = (
            base * WEIGHT_BASE
            + validation * WEIGHT_VALIDATION
            + historical * WEIGHT_HISTORICAL
            + reflexive * WEIGHT_REFLEXIVE
        )

        if final >= self.settings.confidence_execute:
            decision = "execute"
        elif final >= self.settings.confidence_delegate:
            decision = "delegate"
        else:
            decision = "clarify"

        result = ConfidenceResult(
            base_score=base,
            validation_score=validation,
            historical_score=historical,
            reflexive_score=reflexive,
            final_score=round(final, 4),
            decision=decision,
            reasoning=context,
        )
        log.info(
            "confidence.evaluated",
            final=result.final_score,
            decision=decision,
            context=context[:80] if context else "",
        )
        self._audit(result)
        return result

    def _audit(self, result: ConfidenceResult) -> None:
        if self.memory is None:
            return
        try:
            conn = self.memory._get_conn()
            import psycopg2.extras
            from datetime import datetime
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO confidence_audit_log
                        (base_score, validation_score, historical_score, reflexive_score,
                         final_score, decision, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        result.base_score,
                        result.validation_score,
                        result.historical_score,
                        result.reflexive_score,
                        result.final_score,
                        result.decision,
                        datetime.utcnow(),
                    ),
                )
            conn.commit()
        except Exception as exc:
            log.warning("confidence.audit_failed", error=str(exc))
