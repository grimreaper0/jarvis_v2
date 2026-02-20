"""TwitterAgent — Tier 3 specialist for thread creation + first-mover advantage.

Nodes:
  1. thread_write      — Write a 5-10 tweet thread with self-consistency (3 variants)
  2. char_validate     — Enforce 280-char limit on each tweet
  3. timing_optimize   — Mark thread for 2-hour first-mover viral window
  4. publish_queue     — Push to 'twitter_publish' Redis queue
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

class TwitterState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    tweets: list[str]
    tweet_variants: list[list[str]]
    thread_score: float
    char_violations: list[int]
    publish_window_hours: int
    first_mover: bool
    timing_strategy: str
    quality_score: float
    approved: bool
    error: str | None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TWEET_CHARS = 280
THREAD_TWEET_MIN = 5
THREAD_TWEET_MAX = 10
MIN_QUALITY_SCORE = 0.65
FIRST_MOVER_WINDOW_HOURS = 2

QUALITY_WEIGHTS = {
    "tweet_count_ok": 0.20,
    "no_char_violations": 0.25,
    "has_hook_tweet": 0.20,
    "has_cta_tweet": 0.20,
    "has_affiliate_mention": 0.15,
}

AFFILIATE_URL = "https://linktr.ee/fiestygoatai"


def _score_thread(tweets: list[str], topic: str) -> float:
    """Score a tweet thread for engagement potential."""
    score = 0.0
    if THREAD_TWEET_MIN <= len(tweets) <= THREAD_TWEET_MAX:
        score += 0.25
    if tweets and len(tweets[0]) <= 250:
        hook_lower = tweets[0].lower()
        if any(w in hook_lower for w in ["thread", "🧵", "things about", "secret", "nobody"]):
            score += 0.25
    if tweets:
        last = tweets[-1].lower()
        if any(w in last for w in ["follow", "like", "retweet", "link", "subscribe", "fiestygoat"]):
            score += 0.25
    if any(len(t) <= MAX_TWEET_CHARS for t in tweets):
        score += 0.25
    return round(score, 4)


def _deterministic_quality(state: TwitterState) -> float:
    score = 0.0
    tweets = state.get("tweets", [])
    violations = state.get("char_violations", [])

    if THREAD_TWEET_MIN <= len(tweets) <= THREAD_TWEET_MAX:
        score += QUALITY_WEIGHTS["tweet_count_ok"]
    if not violations:
        score += QUALITY_WEIGHTS["no_char_violations"]
    if tweets:
        if any(w in tweets[0].lower() for w in ["🧵", "thread", "things", "secret"]):
            score += QUALITY_WEIGHTS["has_hook_tweet"]
        if any(w in tweets[-1].lower() for w in ["follow", "like", "link", "fiestygoat"]):
            score += QUALITY_WEIGHTS["has_cta_tweet"]
    body = " ".join(tweets).lower()
    if "link" in body or "bio" in body or "fiestygoat" in body:
        score += QUALITY_WEIGHTS["has_affiliate_mention"]

    return round(score, 4)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def thread_write(state: TwitterState) -> dict:
    """Write 5-10 tweet thread with self-consistency; pick highest-scoring variant."""
    topic = state.get("topic", "AI tools")

    prompt = (
        f"Write a Twitter/X thread (5-7 tweets) about: {topic}\n\n"
        f"Thread structure:\n"
        f"Tweet 1: Hook tweet with '🧵' — bold claim or surprising fact that stops the scroll\n"
        f"Tweet 2-5: Value tweets (numbered 1/, 2/, etc.) — actionable insights or facts\n"
        f"Tweet 6: Underrated tip or contrarian take\n"
        f"Tweet 7: CTA — 'Follow @FiestyGoatAI for daily AI tools. Full guide → [link in bio]'\n\n"
        f"Rules:\n"
        f"- Each tweet: MAX 260 characters (leave room for URLs)\n"
        f"- Start with a hook that creates curiosity or FOMO\n"
        f"- Be specific with numbers and facts\n"
        f"- No fluff — every tweet delivers value\n"
        f"- Separate tweets with '---' on its own line\n\n"
        f"Thread:"
    )

    router = LLMRouter()

    default_tweets = [
        f"🧵 5 things about {topic} that will change how you work:\n\n(A thread)",
        f"1/ {topic} isn't what you think it is.\n\nMost people use 10% of its capabilities. Here's what the pros do differently...",
        f"2/ The first thing to unlock: automation.\n\n{topic} can handle [specific task] in 30 seconds that used to take hours. Here's how...",
        f"3/ The tool nobody talks about pairing with {topic}: [complementary tool]\n\nTogether they're 10x more powerful. Here's the workflow...",
        f"4/ The ROI is absurd.\n\nUsers report saving 5-15 hours/week with {topic}. At $50/hr that's $1,000-3,000/month in time savings.",
        f"5/ Getting started in under 10 minutes:\n→ Sign up (free plan available)\n→ Connect your workflow\n→ Let it run\n\nFull guide: link in bio",
        f"If you found this useful, follow @FiestyGoatAI for daily AI tools + automation tips.\n\nAnd RT the first tweet to help others discover {topic} 🙏",
    ]
    best_tweets = default_tweets
    best_score = _score_thread(default_tweets, topic)
    all_variants: list[list[str]] = [default_tweets]

    try:
        responses = await asyncio.gather(*[
            router.complete(LLMRequest(prompt=prompt, temperature=0.85, max_tokens=800))
            for _ in range(3)
        ])

        for resp in responses:
            raw = resp.content.strip()
            parts = [p.strip() for p in raw.split("---") if p.strip()]
            if len(parts) < THREAD_TWEET_MIN:
                parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
            parts = [t[:MAX_TWEET_CHARS] for t in parts if t][:THREAD_TWEET_MAX]
            if len(parts) >= THREAD_TWEET_MIN:
                score = _score_thread(parts, topic)
                all_variants.append(parts)
                log.debug("twitter.thread_variant", score=score, tweets=len(parts))
                if score > best_score:
                    best_score = score
                    best_tweets = parts

        log.info("twitter.thread_write", best_score=best_score, tweet_count=len(best_tweets))
    except Exception as exc:
        log.warning("twitter.thread_write.failed", error=str(exc))

    return {
        "tweets": best_tweets,
        "tweet_variants": all_variants,
        "thread_score": best_score,
    }


def char_validate(state: TwitterState) -> dict:
    """Check each tweet for 280-char limit; log violations."""
    tweets = state.get("tweets", [])
    violations = [i for i, t in enumerate(tweets) if len(t) > MAX_TWEET_CHARS]

    if violations:
        log.warning(
            "twitter.char_violations",
            tweet_indices=violations,
            lengths=[len(tweets[i]) for i in violations],
        )
        fixed = []
        for i, t in enumerate(tweets):
            if len(t) > MAX_TWEET_CHARS:
                fixed.append(t[:MAX_TWEET_CHARS - 3] + "...")
            else:
                fixed.append(t)
        tweets = fixed
        violations = []
    else:
        log.info("twitter.char_validate", all_ok=True, tweet_count=len(tweets))

    return {"tweets": tweets, "char_violations": violations}


def timing_optimize(state: TwitterState) -> dict:
    """Mark thread for first-mover 2-hour viral window + optimal posting time."""
    from datetime import datetime

    hour = datetime.utcnow().hour
    if 12 <= hour <= 15:
        timing_strategy = "peak_hours_immediate"
    elif 0 <= hour <= 8:
        timing_strategy = "schedule_9am_et"
    else:
        timing_strategy = "post_immediately_first_mover"

    log.info(
        "twitter.timing_optimize",
        strategy=timing_strategy,
        first_mover=True,
        window_hours=FIRST_MOVER_WINDOW_HOURS,
    )
    return {
        "publish_window_hours": FIRST_MOVER_WINDOW_HOURS,
        "first_mover": True,
        "timing_strategy": timing_strategy,
    }


async def publish_queue(state: TwitterState) -> dict:
    """Score thread and push to 'twitter_publish' Redis queue."""
    score = _deterministic_quality(state)
    approved = score >= MIN_QUALITY_SCORE

    log.info("twitter.publish_queue", score=score, approved=approved, tweets=len(state.get("tweets", [])))

    if approved:
        try:
            import redis.asyncio as aioredis
            from config.settings import get_settings

            settings = get_settings()
            r = aioredis.from_url(settings.redis_url)
            payload = json.dumps({
                "platform": "twitter",
                "topic": state.get("topic"),
                "tweets": state.get("tweets"),
                "thread_score": state.get("thread_score"),
                "timing_strategy": state.get("timing_strategy"),
                "publish_window_hours": state.get("publish_window_hours"),
                "first_mover": state.get("first_mover"),
                "quality_score": score,
            })
            await r.lpush("twitter_publish", payload)
            await r.aclose()
            log.info("twitter.queued_to_publish", tweets=len(state.get("tweets", [])))
        except Exception as exc:
            log.warning("twitter.redis_push.failed", error=str(exc))

    return {"quality_score": score, "approved": approved}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_twitter_agent():
    """Build and compile the Twitter specialist subgraph."""
    graph = StateGraph(TwitterState)

    graph.add_node("thread_write", thread_write)
    graph.add_node("char_validate", char_validate)
    graph.add_node("timing_optimize", timing_optimize)
    graph.add_node("publish_queue", publish_queue)

    graph.set_entry_point("thread_write")
    graph.add_edge("thread_write", "char_validate")
    graph.add_edge("char_validate", "timing_optimize")
    graph.add_edge("timing_optimize", "publish_queue")
    graph.add_edge("publish_queue", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# AgentRunner — Redis consumer on "twitter_task" queue
# ---------------------------------------------------------------------------

class AgentRunner:
    def __init__(self):
        self.agent = build_twitter_agent()

    async def run(self):
        import redis.asyncio as aioredis
        from config.settings import get_settings

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        log.info("twitter_runner.started", queue="twitter_task")

        while True:
            try:
                raw = await r.brpop("twitter_task", timeout=10)
                if raw is None:
                    continue

                _, data = raw
                task = json.loads(data)
                log.info("twitter_runner.task_received", topic=task.get("topic"))

                initial_state: TwitterState = {
                    "messages": [],
                    "topic": task.get("topic", "AI tools"),
                    "tweets": [],
                    "tweet_variants": [],
                    "thread_score": 0.0,
                    "char_violations": [],
                    "publish_window_hours": FIRST_MOVER_WINDOW_HOURS,
                    "first_mover": True,
                    "timing_strategy": "post_immediately_first_mover",
                    "quality_score": 0.0,
                    "approved": False,
                    "error": None,
                }

                result = await self.agent.ainvoke(initial_state)
                log.info(
                    "twitter_runner.task_complete",
                    approved=result.get("approved"),
                    score=result.get("quality_score"),
                    tweets=len(result.get("tweets", [])),
                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("twitter_runner.error", error=str(exc))
                await asyncio.sleep(2)

        await r.aclose()


twitter_agent = build_twitter_agent()


if __name__ == "__main__":
    async def _main():
        runner = AgentRunner()
        await runner.run()

    asyncio.run(_main())
