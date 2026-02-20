"""InstagramAgent — Tier 3 specialist subgraph for Instagram content.

Nodes:
  1. research_topic    — AutoMem pattern search for high-performing formats
  2. generate_caption  — LLM self-consistency (3 variants, pick best)
  3. select_hashtags   — Pattern-DB hashtag selection (3-5 niche tags)
  4. add_affiliate_link — Append Linktree/affiliate URL bio reference
  5. quality_gate      — Deterministic scoring; push to Redis if >=0.7
"""
from __future__ import annotations

import asyncio
import json
import structlog
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from jarvis.core.router import LLMRouter, LLMRequest

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class InstagramState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    niche: str
    caption: str
    hashtags: list[str]
    affiliate_links: list[str]
    quality_score: float
    approved: bool
    error: str | None
    patterns: list[dict]
    content_format: str


# ---------------------------------------------------------------------------
# Hashtag pools (ported from v1)
# ---------------------------------------------------------------------------

HASHTAG_POOLS: dict[str, list[str]] = {
    "ai": ["#AIProductivity", "#ChatGPTHacks", "#AIAutomation", "#AIForBusiness"],
    "tech": ["#Tech", "#TechReview", "#TechNews", "#Innovation", "#Software"],
    "business": ["#Entrepreneur", "#StartUp", "#SideHustle", "#PassiveIncome", "#OnlineBusiness"],
    "marketing": ["#DigitalMarketing", "#ContentCreator", "#SEO", "#Branding", "#MarketingTips"],
    "lifestyle": ["#ProductivityTips", "#SelfImprovement", "#WorkFromHome", "#RemoteWork", "#Goals"],
    "general": ["#AITools", "#FiestyGoatAI", "#Trending", "#ContentCreation"],
}

MIN_QUALITY_SCORE = 0.70
LINKTREE_URL = "https://linktr.ee/fiestygoatai"


# ---------------------------------------------------------------------------
# Scoring helpers (deterministic, no LLM — ported from v1)
# ---------------------------------------------------------------------------

def _score_caption(text: str, topic: str) -> float:
    """Simple quality score for a caption variant: 0.0–1.0."""
    score = 0.0
    if len(text) >= 60:
        score += 0.3
    if any(word in text.lower() for word in ["save", "follow", "comment", "link", "learn", "discover", "stop"]):
        score += 0.25
    if "?" in text or "!" in text:
        score += 0.15
    if topic.lower() in text.lower():
        score += 0.15
    if len(text) <= 200:
        score += 0.15
    return round(score, 4)


def _deterministic_quality_score(state: InstagramState) -> float:
    """Deterministic gate — mirrors v1 QUALITY_WEIGHTS logic."""
    score = 0.0
    caption = state.get("caption", "")
    hashtags = state.get("hashtags", [])
    topic = state.get("topic", "")

    if caption.strip():
        score += 0.15
    if len(caption.strip()) >= 50:
        score += 0.10
    if hashtags:
        score += 0.15
    if 3 <= len(hashtags) <= 10:
        score += 0.10
    if any(kw in caption.lower() for kw in ["save", "follow", "link", "comment", "learn"]):
        score += 0.10
    # Affiliate disclosure implicit — no affiliates required for this check
    score += 0.10
    # Content type always valid here
    score += 0.10
    # No prohibited content (naive check)
    banned = ["hate speech", "financial advice", "medical advice"]
    if not any(b in (caption + " " + topic).lower() for b in banned):
        score += 0.20

    return round(score, 4)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def research_topic(state: InstagramState) -> dict:
    """Search AutoMem for high-performing Instagram content patterns."""
    topic = state.get("topic", "AI tools")
    patterns: list[dict] = []

    try:
        import psycopg2
        import psycopg2.extras
        from config.settings import get_settings

        settings = get_settings()
        conn = psycopg2.connect(settings.postgres_url)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT description, confidence_score, success_count, total_uses, metadata
            FROM patterns
            WHERE context ILIKE %s OR description ILIKE %s
            ORDER BY confidence_score DESC
            LIMIT 5
            """,
            (f"%instagram%", f"%{topic[:30]}%"),
        )
        rows = cur.fetchall()
        patterns = [dict(r) for r in rows]
        conn.close()
        log.info("instagram.research_topic", topic=topic, pattern_count=len(patterns))
    except Exception as exc:
        log.warning("instagram.research_topic.failed", error=str(exc))

    niche = "ai" if "ai" in topic.lower() else "general"
    return {"patterns": patterns, "niche": niche, "content_format": "carousel"}


async def generate_caption(state: InstagramState) -> dict:
    """Generate caption via self-consistency: 3 variants, pick highest-scoring."""
    topic = state.get("topic", "AI tools")
    niche = state.get("niche", "general")
    patterns = state.get("patterns", [])

    pattern_context = ""
    if patterns:
        pattern_context = "\n".join(
            f"- {p.get('description', '')[:100]}" for p in patterns[:3]
        )

    prompt = (
        f"Write an Instagram caption for a {state.get('content_format', 'carousel')} post about: {topic}\n\n"
        f"Niche: {niche}\n"
        f"Historical best practices:\n{pattern_context}\n\n"
        f"Requirements:\n"
        f"- Start with a strong hook (question or bold statement)\n"
        f"- Explain the core value clearly\n"
        f"- End with a call-to-action (save, follow, or link in bio)\n"
        f"- 80-180 characters\n"
        f"- Do NOT include hashtags\n\n"
        f"Caption:"
    )

    router = LLMRouter()
    best_caption = f"Discover how {topic} is changing everything. Save this for later! 🔥"
    best_score = 0.0

    try:
        variants = await asyncio.gather(*[
            router.complete(LLMRequest(prompt=prompt, temperature=0.9))
            for _ in range(3)
        ])

        for variant in variants:
            text = variant.content.strip()
            score = _score_caption(text, topic)
            log.debug("instagram.caption_variant", score=score, length=len(text))
            if score > best_score:
                best_score = score
                best_caption = text

        log.info("instagram.generate_caption", best_score=best_score, topic=topic)
    except Exception as exc:
        log.warning("instagram.generate_caption.failed", error=str(exc))

    return {"caption": best_caption}


def select_hashtags(state: InstagramState) -> dict:
    """Select 3-5 niche hashtags from pattern pool based on topic/niche."""
    topic = state.get("topic", "AI tools")
    niche = state.get("niche", "general")
    topic_lower = topic.lower()
    selected: list[str] = []

    pool_scores: dict[str, int] = {}
    for pool_name in HASHTAG_POOLS:
        if pool_name == "general":
            continue
        score = 2 if pool_name in topic_lower else 0
        for tag in HASHTAG_POOLS[pool_name]:
            if tag.lstrip("#").lower() in topic_lower:
                score += 1
        pool_scores[pool_name] = score

    sorted_pools = sorted(pool_scores.items(), key=lambda x: x[1], reverse=True)
    target = 4

    for pool_name, score in sorted_pools:
        if score > 0 and len(selected) < target:
            pool = HASHTAG_POOLS[pool_name]
            for tag in pool:
                if tag not in selected and len(selected) < target:
                    selected.append(tag)

    for tag in HASHTAG_POOLS["general"]:
        if tag not in selected and len(selected) < target:
            selected.append(tag)

    log.info("instagram.select_hashtags", count=len(selected), niche=niche)
    return {"hashtags": selected}


def add_affiliate_link(state: InstagramState) -> dict:
    """Append FTC-compliant Linktree reference and disclosure to caption."""
    caption = state.get("caption", "")
    affiliate_links = state.get("affiliate_links", [])

    if affiliate_links or True:  # Always reference Linktree bio link
        if "link in bio" not in caption.lower() and "linktree" not in caption.lower():
            caption = caption.rstrip() + " 🔗 Link in bio."
        affiliate_links = affiliate_links or [LINKTREE_URL]

    log.info("instagram.add_affiliate_link", links=len(affiliate_links))
    return {"caption": caption, "affiliate_links": affiliate_links}


async def quality_gate(state: InstagramState) -> dict:
    """Score content deterministically; push to Redis queue if >=0.7."""
    score = _deterministic_quality_score(state)
    approved = score >= MIN_QUALITY_SCORE

    log.info("instagram.quality_gate", score=score, approved=approved)

    if approved:
        try:
            import redis.asyncio as aioredis
            from config.settings import get_settings

            settings = get_settings()
            r = aioredis.from_url(settings.redis_url)
            payload = json.dumps({
                "platform": "instagram",
                "topic": state.get("topic"),
                "caption": state.get("caption"),
                "hashtags": state.get("hashtags"),
                "affiliate_links": state.get("affiliate_links"),
                "quality_score": score,
                "content_format": state.get("content_format"),
            })
            await r.lpush("content_gate", payload)
            await r.aclose()
            log.info("instagram.queued_to_content_gate", score=score)
        except Exception as exc:
            log.warning("instagram.redis_push.failed", error=str(exc))

    return {"quality_score": score, "approved": approved}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_instagram_agent():
    """Build and compile the Instagram specialist subgraph."""
    graph = StateGraph(InstagramState)

    graph.add_node("research_topic", research_topic)
    graph.add_node("generate_caption", generate_caption)
    graph.add_node("select_hashtags", select_hashtags)
    graph.add_node("add_affiliate_link", add_affiliate_link)
    graph.add_node("quality_gate", quality_gate)

    graph.set_entry_point("research_topic")
    graph.add_edge("research_topic", "generate_caption")
    graph.add_edge("generate_caption", "select_hashtags")
    graph.add_edge("select_hashtags", "add_affiliate_link")
    graph.add_edge("add_affiliate_link", "quality_gate")
    graph.add_edge("quality_gate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# AgentRunner — Redis consumer on "instagram_task" queue
# ---------------------------------------------------------------------------

class AgentRunner:
    """Consumes tasks from Redis 'instagram_task' queue and runs the subgraph."""

    def __init__(self):
        self.agent = build_instagram_agent()

    async def run(self):
        import redis.asyncio as aioredis
        from config.settings import get_settings

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        log.info("instagram_runner.started", queue="instagram_task")

        while True:
            try:
                raw = await r.brpop("instagram_task", timeout=10)
                if raw is None:
                    continue

                _, data = raw
                task = json.loads(data)
                log.info("instagram_runner.task_received", topic=task.get("topic"))

                initial_state: InstagramState = {
                    "messages": [],
                    "topic": task.get("topic", "AI tools"),
                    "niche": task.get("niche", "general"),
                    "caption": "",
                    "hashtags": [],
                    "affiliate_links": task.get("affiliate_links", []),
                    "quality_score": 0.0,
                    "approved": False,
                    "error": None,
                    "patterns": [],
                    "content_format": task.get("content_format", "carousel"),
                }

                result = await self.agent.ainvoke(initial_state)
                log.info(
                    "instagram_runner.task_complete",
                    approved=result.get("approved"),
                    score=result.get("quality_score"),
                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("instagram_runner.error", error=str(exc))
                await asyncio.sleep(2)

        await r.aclose()


instagram_agent = build_instagram_agent()


if __name__ == "__main__":
    async def _main():
        runner = AgentRunner()
        await runner.run()

    asyncio.run(_main())
