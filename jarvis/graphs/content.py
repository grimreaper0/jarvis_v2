"""Content quality gate graph (Tier 2 LangGraph).

score_content → check_guidelines → approve_content / reject_content

Scoring is fully deterministic (no LLM calls needed) for speed and cost.
Platform-specific rules are applied per ContentState['platform'].
"""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from jarvis.core.guardrails import Guardrails
from jarvis.core.state import ContentState

log = structlog.get_logger()

_guardrails = Guardrails()

# Approval threshold matches v1 guardrails min_content_quality = 0.70
QUALITY_THRESHOLD: float = 0.70


# ─────────────────────────── Platform scoring specs ──────────────────────────

def _score_instagram(content: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Instagram scoring rubric:
      Caption length  50-300 chars  → +0.25
      Hashtag count   5-30          → +0.25
      Has CTA                       → +0.25
      Has media_url                 → +0.25
    """
    reasons: list[str] = []
    score = 0.0

    caption: str = content.get("caption", content.get("body", ""))
    caption_len = len(caption.strip())
    if 50 <= caption_len <= 300:
        score += 0.25
    else:
        reasons.append(
            f"caption_length={caption_len} (need 50-300)"
        )

    hashtags: list = content.get("hashtags", [])
    if isinstance(hashtags, str):
        hashtags = hashtags.split()
    hashtag_count = len(hashtags)
    if 5 <= hashtag_count <= 30:
        score += 0.25
    else:
        reasons.append(f"hashtag_count={hashtag_count} (need 5-30)")

    if content.get("cta") or "http" in caption.lower() or "link in bio" in caption.lower():
        score += 0.25
    else:
        reasons.append("missing_cta")

    if content.get("media_url") or content.get("image_url"):
        score += 0.25
    else:
        reasons.append("missing_media_url")

    return round(score, 4), reasons


def _score_youtube(content: dict[str, Any]) -> tuple[float, list[str]]:
    """
    YouTube scoring rubric:
      Title length        30-70 chars   → +0.25
      Description length  100+ chars    → +0.25
      Has keywords list                 → +0.25
      Has thumbnail concept / URL       → +0.25
    """
    reasons: list[str] = []
    score = 0.0

    title: str = content.get("title", "")
    title_len = len(title.strip())
    if 30 <= title_len <= 70:
        score += 0.25
    else:
        reasons.append(f"title_length={title_len} (need 30-70)")

    description: str = content.get("description", content.get("body", ""))
    desc_len = len(description.strip())
    if desc_len >= 100:
        score += 0.25
    else:
        reasons.append(f"description_length={desc_len} (need 100+)")

    keywords = content.get("keywords", content.get("tags", []))
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    if len(keywords) >= 3:
        score += 0.25
    else:
        reasons.append(f"keywords_count={len(keywords)} (need 3+)")

    if content.get("thumbnail_url") or content.get("thumbnail_concept"):
        score += 0.25
    else:
        reasons.append("missing_thumbnail")

    return round(score, 4), reasons


def _score_newsletter(content: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Newsletter scoring rubric:
      Word count          1500-2000     → +0.30
      Has affiliate links               → +0.25
      Has CTA                           → +0.25
      Has subject line                  → +0.20
    """
    reasons: list[str] = []
    score = 0.0

    body: str = content.get("body", content.get("content", ""))
    word_count = len(body.split())
    if 1500 <= word_count <= 2000:
        score += 0.30
    else:
        reasons.append(f"word_count={word_count} (need 1500-2000)")

    affiliate_links = content.get("affiliate_links", [])
    if isinstance(affiliate_links, str):
        affiliate_links = [affiliate_links] if affiliate_links else []
    if len(affiliate_links) >= 1 or "http" in body.lower():
        score += 0.25
    else:
        reasons.append("no_affiliate_links")

    if content.get("cta") or "click here" in body.lower() or "subscribe" in body.lower():
        score += 0.25
    else:
        reasons.append("missing_cta")

    if content.get("subject") or content.get("title"):
        score += 0.20
    else:
        reasons.append("missing_subject_line")

    return round(score, 4), reasons


def _score_tiktok(content: dict[str, Any]) -> tuple[float, list[str]]:
    """
    TikTok scoring rubric:
      Hook present (first 3s description)  → +0.30
      Trending sounds tagged               → +0.25
      Caption length  10-150 chars         → +0.25
      Has video_url or script              → +0.20
    """
    reasons: list[str] = []
    score = 0.0

    hook: str = content.get("hook", "")
    if len(hook.strip()) >= 5:
        score += 0.30
    else:
        reasons.append(f"hook_missing_or_too_short (len={len(hook.strip())})")

    trending_sounds = content.get("trending_sounds", content.get("sounds", []))
    if isinstance(trending_sounds, str):
        trending_sounds = [s.strip() for s in trending_sounds.split(",") if s.strip()]
    if len(trending_sounds) >= 1:
        score += 0.25
    else:
        reasons.append("no_trending_sounds_tagged")

    caption: str = content.get("caption", content.get("body", ""))
    caption_len = len(caption.strip())
    if 10 <= caption_len <= 150:
        score += 0.25
    else:
        reasons.append(f"caption_length={caption_len} (need 10-150)")

    if content.get("video_url") or content.get("script"):
        score += 0.20
    else:
        reasons.append("missing_video_url_or_script")

    return round(score, 4), reasons


def _score_generic(content: dict[str, Any]) -> tuple[float, list[str]]:
    """Generic fallback scorer for unknown platforms."""
    reasons: list[str] = []
    score = 0.0

    title = content.get("title", "")
    body = content.get("body", content.get("content", ""))

    if len(title.strip()) > 10:
        score += 0.25
    else:
        reasons.append("title_too_short")

    if len(body.strip()) > 100:
        score += 0.35
    else:
        reasons.append("body_too_short")

    hashtags = content.get("hashtags", [])
    if len(hashtags) >= 3:
        score += 0.20
    else:
        reasons.append("insufficient_hashtags")

    if content.get("media_url") or content.get("image_url"):
        score += 0.20
    else:
        reasons.append("no_media")

    return round(score, 4), reasons


PLATFORM_SCORERS = {
    "instagram": _score_instagram,
    "youtube": _score_youtube,
    "newsletter": _score_newsletter,
    "tiktok": _score_tiktok,
}


# ─────────────────────────── Node implementations ────────────────────────────


async def score_content(state: ContentState) -> dict[str, Any]:
    """Deterministic quality scoring — no LLM needed."""
    content: dict[str, Any] = state.get("content", {})
    platform: str = state.get("platform", "generic").lower()

    scorer = PLATFORM_SCORERS.get(platform, _score_generic)
    quality_score, deficiencies = scorer(content)

    log.info(
        "content.scored",
        platform=platform,
        score=quality_score,
        deficiencies=deficiencies,
    )
    return {
        "quality_score": quality_score,
        "rejection_reason": "; ".join(deficiencies) if deficiencies else None,
    }


async def check_guidelines(state: ContentState) -> dict[str, Any]:
    """Platform-specific safety and policy checks."""
    content: dict[str, Any] = state.get("content", {})
    platform: str = state.get("platform", "generic").lower()

    # Rate limit check
    posts_this_hour: int = int(content.get("posts_this_hour", 0))
    rate_result = _guardrails.check_rate_limit(
        platform=platform, posts_this_hour=posts_this_hour
    )
    if not rate_result.passed:
        log.warning(
            "content.guidelines.rate_limit",
            platform=platform,
            reason=rate_result.reason,
        )
        return {
            "approved": False,
            "rejection_reason": rate_result.reason,
        }

    # Quality floor check using guardrails
    quality_score: float = state.get("quality_score", 0.0)
    quality_result = _guardrails.check_content_quality(quality_score)
    if not quality_result.passed:
        log.warning(
            "content.guidelines.quality",
            platform=platform,
            score=quality_score,
            reason=quality_result.reason,
        )
        return {
            "approved": False,
            "rejection_reason": (
                f"{state.get('rejection_reason') or ''} | {quality_result.reason}".lstrip(" |")
            ),
        }

    # Blocked keyword sweep (non-exhaustive, extend as needed)
    BLOCKED_KEYWORDS: list[str] = ["guaranteed returns", "get rich quick", "100% profit"]
    body: str = (
        content.get("body", "")
        + " "
        + content.get("caption", "")
        + " "
        + content.get("title", "")
    ).lower()
    for kw in BLOCKED_KEYWORDS:
        if kw in body:
            reason = f"blocked_keyword: '{kw}'"
            log.warning("content.guidelines.blocked_keyword", keyword=kw, platform=platform)
            return {"approved": False, "rejection_reason": reason}

    log.info("content.guidelines.passed", platform=platform, score=quality_score)
    return {"approved": True}


def _route_quality(state: ContentState) -> str:
    return "approve" if state.get("approved") else "reject"


async def approve_content(state: ContentState) -> dict[str, Any]:
    """Approved content: push to publish queue."""
    content: dict[str, Any] = state["content"]
    platform: str = state["platform"]

    payload = json.dumps(
        {
            "id": str(uuid4()),
            "platform": platform,
            "quality_score": state["quality_score"],
            "content": content,
        }
    )

    try:
        import redis.asyncio as aioredis

        from config.settings import get_settings

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        queue_name = f"{platform}_publish"
        await r.rpush(queue_name, payload)
        await r.aclose()
        log.info(
            "content.approve.queued",
            platform=platform,
            queue=queue_name,
            score=state["quality_score"],
        )
    except Exception as exc:
        log.error("content.approve.redis_failed", error=str(exc), platform=platform)

    return {"approved": True}


async def reject_content(state: ContentState) -> dict[str, Any]:
    """Rejected content: push back to regeneration queue with reason."""
    content: dict[str, Any] = state["content"]
    platform: str = state["platform"]
    reason: str = state.get("rejection_reason") or "quality_below_threshold"

    payload = json.dumps(
        {
            "id": str(uuid4()),
            "platform": platform,
            "quality_score": state.get("quality_score", 0.0),
            "rejection_reason": reason,
            "content": content,
        }
    )

    try:
        import redis.asyncio as aioredis

        from config.settings import get_settings

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        queue_name = f"{platform}_regenerate"
        await r.rpush(queue_name, payload)
        await r.aclose()
        log.info(
            "content.reject.queued",
            platform=platform,
            queue=queue_name,
            reason=reason,
            score=state.get("quality_score", 0.0),
        )
    except Exception as exc:
        log.error("content.reject.redis_failed", error=str(exc), platform=platform)

    return {"approved": False, "rejection_reason": reason}


# ─────────────────────────── Graph builder ───────────────────────────────────


def build_content_graph():
    """Build and compile the content quality gate graph."""
    graph = StateGraph(ContentState)

    graph.add_node("score_content", score_content)
    graph.add_node("check_guidelines", check_guidelines)
    graph.add_node("approve_content", approve_content)
    graph.add_node("reject_content", reject_content)

    graph.set_entry_point("score_content")
    graph.add_edge("score_content", "check_guidelines")

    graph.add_conditional_edges(
        "check_guidelines",
        _route_quality,
        {
            "approve": "approve_content",
            "reject": "reject_content",
        },
    )

    graph.add_edge("approve_content", END)
    graph.add_edge("reject_content", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


content_graph = build_content_graph()


# ─────────────────────────── Graph Runner ────────────────────────────────────


class ContentGraphRunner:
    """Consume from platform content queues and run the quality gate graph."""

    # Queues to consume in round-robin; each item must include 'platform'
    QUEUES: list[str] = [
        "instagram_content_check",
        "youtube_content_check",
        "newsletter_content_check",
        "tiktok_content_check",
        "content_check",
    ]

    def __init__(self) -> None:
        self.graph = build_content_graph()
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis

            from config.settings import get_settings

            settings = get_settings()
            self._redis = aioredis.from_url(settings.redis_url)
        return self._redis

    async def run_forever(self) -> None:
        """Consume from content queues and run graph for each item indefinitely."""
        log.info("content_runner.starting", queues=self.QUEUES)
        r = await self._get_redis()

        while True:
            try:
                item = await r.blpop(self.QUEUES, timeout=5)
                if item is None:
                    continue

                queue_name, raw = item
                try:
                    task: dict[str, Any] = json.loads(raw)
                except json.JSONDecodeError as exc:
                    log.error("content_runner.bad_json", error=str(exc))
                    continue

                thread_id: str = task.get("id") or str(uuid4())
                config = {"configurable": {"thread_id": thread_id}}

                platform: str = task.get("platform", "generic")

                initial_state: ContentState = {
                    "messages": [],
                    "confidence": 0.0,
                    "error": None,
                    "content": task.get("content", task),
                    "platform": platform,
                    "quality_score": 0.0,
                    "approved": False,
                    "rejection_reason": None,
                }

                try:
                    result = await self.graph.ainvoke(initial_state, config=config)
                    log.info(
                        "content_runner.completed",
                        thread_id=thread_id,
                        platform=platform,
                        approved=result.get("approved"),
                        score=result.get("quality_score"),
                    )
                except Exception as graph_exc:
                    log.error(
                        "content_runner.graph_error",
                        error=str(graph_exc),
                        thread_id=thread_id,
                        platform=platform,
                    )

            except Exception as loop_exc:
                log.error("content_runner.loop_error", error=str(loop_exc))

import asyncio

if __name__ == "__main__":
    asyncio.run(ContentGraphRunner().run_forever())

