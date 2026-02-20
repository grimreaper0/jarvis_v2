"""Guardrails — Safety limits for financial, content, and rate decisions."""
import structlog
from dataclasses import dataclass
from enum import Enum

log = structlog.get_logger()


class GuardrailType(str, Enum):
    FINANCIAL = "financial"
    CONTENT = "content"
    RATE_LIMIT = "rate_limit"
    QUALITY = "quality"


@dataclass
class GuardrailResult:
    passed: bool
    rule_name: str
    guardrail_type: GuardrailType
    reason: str = ""


class Guardrails:
    """Configurable safety limits loaded from the guardrails DB table.

    Financial:   max trade size, daily loss circuit breaker
    Content:     min quality score, blocked keywords
    Rate limit:  max posts/hour per platform
    Quality:     min confidence before autonomous action
    """

    DEFAULTS: dict[str, dict] = {
        "max_trade_usd": {"type": GuardrailType.FINANCIAL, "value": 100.0},
        "max_daily_loss_usd": {"type": GuardrailType.FINANCIAL, "value": 300.0},
        "min_content_quality": {"type": GuardrailType.QUALITY, "value": 0.70},
        "max_instagram_posts_per_hour": {"type": GuardrailType.RATE_LIMIT, "value": 3},
        "max_twitter_posts_per_hour": {"type": GuardrailType.RATE_LIMIT, "value": 5},
    }

    def check_trade(self, amount_usd: float, daily_loss_usd: float = 0.0) -> GuardrailResult:
        max_trade = self.DEFAULTS["max_trade_usd"]["value"]
        if amount_usd > max_trade:
            return GuardrailResult(
                passed=False,
                rule_name="max_trade_usd",
                guardrail_type=GuardrailType.FINANCIAL,
                reason=f"Trade ${amount_usd:.2f} exceeds max ${max_trade:.2f}",
            )
        max_loss = self.DEFAULTS["max_daily_loss_usd"]["value"]
        if daily_loss_usd > max_loss:
            return GuardrailResult(
                passed=False,
                rule_name="max_daily_loss_usd",
                guardrail_type=GuardrailType.FINANCIAL,
                reason=f"Daily loss ${daily_loss_usd:.2f} exceeds circuit breaker ${max_loss:.2f}",
            )
        return GuardrailResult(passed=True, rule_name="trade_check", guardrail_type=GuardrailType.FINANCIAL)

    def check_content_quality(self, quality_score: float) -> GuardrailResult:
        min_q = self.DEFAULTS["min_content_quality"]["value"]
        if quality_score < min_q:
            return GuardrailResult(
                passed=False,
                rule_name="min_content_quality",
                guardrail_type=GuardrailType.QUALITY,
                reason=f"Quality {quality_score:.2f} below minimum {min_q:.2f}",
            )
        return GuardrailResult(passed=True, rule_name="quality_check", guardrail_type=GuardrailType.QUALITY)

    def check_rate_limit(self, platform: str, posts_this_hour: int) -> GuardrailResult:
        key = f"max_{platform}_posts_per_hour"
        limit = self.DEFAULTS.get(key, {}).get("value", 10)
        if posts_this_hour >= limit:
            return GuardrailResult(
                passed=False,
                rule_name=key,
                guardrail_type=GuardrailType.RATE_LIMIT,
                reason=f"{platform} rate limit reached ({posts_this_hour}/{limit} per hour)",
            )
        return GuardrailResult(passed=True, rule_name=key, guardrail_type=GuardrailType.RATE_LIMIT)
