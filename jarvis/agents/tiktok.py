"""TikTokAgent — Tier 3 specialist for short-form viral content.

Nodes:
  1. hook_write    — Write 3 hook variants, pick highest viral-score
  2. sound_select  — Select trending sound category and FYP tags
  3. viral_score   — Deterministic virality scoring (0.0-1.0)
  4. publish_queue — Push to 'tiktok_publish' Redis queue if score > 0.7
"""
from __future__ import annotations

import asyncio
import json
import structlog
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from jarvis.core.router import LLMRouter, LLMRequest

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TikTokState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    hook: str
    hook_variants: list[str]
    script: str
    trending_sound: str
    sound_category: str
    fyp_tags: list[str]
    hashtags: list[str]
    duration_sec: int
    viral_score: float
    approved: bool
    error: str | None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIRAL_SCORE_THRESHOLD = 0.70

TRENDING_SOUND_CATEGORIES = [
    "upbeat_electronic",
    "lo_fi_chill",
    "viral_trending",
    "dramatic_reveal",
    "tutorial_background",
]

FYP_BASE_TAGS = ["#FYP", "#ForYou", "#AITools", "#Tech", "#FiestyGoatAI"]

QUALITY_WEIGHTS = {
    "has_hook": 0.25,
    "has_trending_sound": 0.30,
    "hook_has_emotion": 0.25,
    "duration_optimal": 0.20,
}


def _score_hook_viral(hook: str, topic: str) -> float:
    """Score a TikTok hook for viral potential."""
    score = 0.0
    hook_lower = hook.lower()
    if hook.strip():
        score += 0.20
    emotion_triggers = ["pov:", "wait,", "stop!", "nobody talks about", "this is why",
                        "you won't believe", "the reason", "plot twist", "🤯", "😱", "🔥"]
    if any(t in hook_lower for t in emotion_triggers):
        score += 0.30
    if "?" in hook:
        score += 0.15
    if topic.lower() in hook_lower:
        score += 0.15
    if len(hook) <= 80:
        score += 0.10
    if hook[0].isupper():
        score += 0.10
    return round(score, 4)


def _deterministic_viral_score(state: TikTokState) -> float:
    score = 0.0
    hook = state.get("hook", "")
    sound = state.get("trending_sound", "")
    duration = state.get("duration_sec", 0)

    if hook.strip():
        score += QUALITY_WEIGHTS["has_hook"]
        emotion_triggers = ["pov", "stop", "wait", "🤯", "😱", "nobody", "plot twist"]
        if any(t in hook.lower() for t in emotion_triggers):
            score += QUALITY_WEIGHTS["hook_has_emotion"]

    if sound.strip():
        score += QUALITY_WEIGHTS["has_trending_sound"]

    if 15 <= duration <= 60:
        score += QUALITY_WEIGHTS["duration_optimal"]

    return round(score, 4)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def hook_write(state: TikTokState) -> dict:
    """Generate 3 hook variants via self-consistency, pick highest viral-score."""
    topic = state.get("topic", "AI tools")

    prompt = (
        f"Write a TikTok video hook (opening line) for a video about: {topic}\n\n"
        f"Requirements:\n"
        f"- Maximum 80 characters\n"
        f"- Must trigger immediate emotion: shock, curiosity, or FOMO\n"
        f"- Start with 'POV:', 'STOP:', 'WAIT -', or an emoji\n"
        f"- Do NOT explain — just tease\n"
        f"- Examples: 'POV: You just found out AI can do THIS 🤯', "
        f"'Nobody talks about this {topic} feature...', "
        f"'Stop what you're doing. {topic} just changed everything.'\n\n"
        f"Hook (one line only):"
    )

    router = LLMRouter()
    best_hook = f"POV: {topic} just changed everything and nobody is talking about it 🤯"
    best_score = 0.0
    variants: list[str] = []

    try:
        responses = await asyncio.gather(*[
            router.complete(LLMRequest(prompt=prompt, temperature=0.95))
            for _ in range(3)
        ])

        for resp in responses:
            text = resp.content.strip().split("\n")[0]
            score = _score_hook_viral(text, topic)
            variants.append(text)
            log.debug("tiktok.hook_variant", score=score, hook=text[:60])
            if score > best_score:
                best_score = score
                best_hook = text

        log.info("tiktok.hook_write", best_score=best_score, variant_count=len(variants))
    except Exception as exc:
        log.warning("tiktok.hook_write.failed", error=str(exc))
        variants = [best_hook]

    script = (
        f"HOOK: {best_hook}\n\n"
        f"[Show screen recording / demo of {topic}]\n\n"
        f"VO: 'Here's what you're missing...'\n\n"
        f"[Reveal key feature]\n\n"
        f"CTA: 'Follow for daily AI tool drops. Link in bio for the full guide.'\n"
    )

    return {
        "hook": best_hook,
        "hook_variants": variants,
        "script": script,
        "duration_sec": 30,
    }


def sound_select(state: TikTokState) -> dict:
    """Select trending sound category and populate FYP tags."""
    topic = state.get("topic", "AI tools")
    topic_lower = topic.lower()

    if "tutorial" in topic_lower or "how to" in topic_lower:
        sound_category = "tutorial_background"
        trending_sound = "Lo-fi chill beats (no copyright)"
    elif "reveal" in topic_lower or "secret" in topic_lower or "hidden" in topic_lower:
        sound_category = "dramatic_reveal"
        trending_sound = "Dramatic orchestral sting"
    elif "ai" in topic_lower or "tech" in topic_lower:
        sound_category = "upbeat_electronic"
        trending_sound = "Upbeat synth pop (trending)"
    else:
        sound_category = "viral_trending"
        trending_sound = "Current #1 TikTok trending sound"

    topic_words = topic_lower.split()
    niche_tags: list[str] = []
    if "ai" in topic_words or "artificial" in topic_words:
        niche_tags = ["#ArtificialIntelligence", "#AILife", "#AIHacks"]
    elif "code" in topic_words or "coding" in topic_words:
        niche_tags = ["#CodeTok", "#CodingLife", "#DevTok"]
    else:
        niche_tags = ["#TechTok", "#Innovation2026", "#FutureTech"]

    hashtags = FYP_BASE_TAGS + niche_tags
    log.info("tiktok.sound_select", sound=trending_sound, tags=len(hashtags))
    return {
        "trending_sound": trending_sound,
        "sound_category": sound_category,
        "fyp_tags": FYP_BASE_TAGS,
        "hashtags": hashtags,
    }


def viral_score(state: TikTokState) -> dict:
    """Compute deterministic viral potential score."""
    score = _deterministic_viral_score(state)
    log.info("tiktok.viral_score", score=score, threshold=VIRAL_SCORE_THRESHOLD)
    return {"viral_score": score}


async def publish_queue(state: TikTokState) -> dict:
    """Push to 'tiktok_publish' Redis queue if viral score > 0.7."""
    score = state.get("viral_score", 0.0)
    approved = score >= VIRAL_SCORE_THRESHOLD

    log.info("tiktok.publish_queue", score=score, approved=approved)

    if approved:
        try:
            import redis.asyncio as aioredis
            from config.settings import get_settings

            settings = get_settings()
            r = aioredis.from_url(settings.redis_url)
            payload = json.dumps({
                "platform": "tiktok",
                "topic": state.get("topic"),
                "hook": state.get("hook"),
                "script": state.get("script"),
                "trending_sound": state.get("trending_sound"),
                "sound_category": state.get("sound_category"),
                "hashtags": state.get("hashtags"),
                "fyp_tags": state.get("fyp_tags"),
                "duration_sec": state.get("duration_sec"),
                "viral_score": score,
            })
            await r.lpush("tiktok_publish", payload)
            await r.aclose()
            log.info("tiktok.queued_to_publish", score=score)
        except Exception as exc:
            log.warning("tiktok.redis_push.failed", error=str(exc))
    else:
        log.info("tiktok.below_viral_threshold", score=score, threshold=VIRAL_SCORE_THRESHOLD)

    return {"approved": approved}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_tiktok_agent():
    """Build and compile the TikTok specialist subgraph."""
    graph = StateGraph(TikTokState)

    graph.add_node("hook_write", hook_write)
    graph.add_node("sound_select", sound_select)
    graph.add_node("viral_score", viral_score)
    graph.add_node("publish_queue", publish_queue)

    graph.set_entry_point("hook_write")
    graph.add_edge("hook_write", "sound_select")
    graph.add_edge("sound_select", "viral_score")
    graph.add_edge("viral_score", "publish_queue")
    graph.add_edge("publish_queue", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# AgentRunner — Redis consumer on "tiktok_task" queue
# ---------------------------------------------------------------------------

class AgentRunner:
    def __init__(self):
        self.agent = build_tiktok_agent()

    async def run(self):
        import redis.asyncio as aioredis
        from config.settings import get_settings

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        log.info("tiktok_runner.started", queue="tiktok_task")

        while True:
            try:
                raw = await r.brpop("tiktok_task", timeout=10)
                if raw is None:
                    continue

                _, data = raw
                task = json.loads(data)
                log.info("tiktok_runner.task_received", topic=task.get("topic"))

                initial_state: TikTokState = {
                    "messages": [],
                    "topic": task.get("topic", "AI tools"),
                    "hook": "",
                    "hook_variants": [],
                    "script": "",
                    "trending_sound": "",
                    "sound_category": "",
                    "fyp_tags": [],
                    "hashtags": [],
                    "duration_sec": task.get("duration_sec", 30),
                    "viral_score": 0.0,
                    "approved": False,
                    "error": None,
                }

                result = await self.agent.ainvoke(initial_state)
                log.info(
                    "tiktok_runner.task_complete",
                    approved=result.get("approved"),
                    viral_score=result.get("viral_score"),
                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("tiktok_runner.error", error=str(exc))
                await asyncio.sleep(2)

        await r.aclose()


tiktok_agent = build_tiktok_agent()


if __name__ == "__main__":
    async def _main():
        runner = AgentRunner()
        await runner.run()

    asyncio.run(_main())
