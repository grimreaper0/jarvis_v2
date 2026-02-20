"""YouTubeAgent — Tier 3 specialist subgraph for YouTube content.

Nodes:
  1. research_niche          — AutoMem search for YouTube SEO patterns
  2. generate_title          — 3 title variants; pick highest CTR score
  3. write_script_outline    — Section-by-section outline (hook, value, CTA)
  4. optimize_seo            — Description, tags, chapters
  5. create_thumbnail_concept — Text description of thumbnail layout
  6. quality_gate            — Push approved content to 'youtube_publish' queue
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

class YouTubeState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    niche: str
    title: str
    script_outline: str
    description: str
    tags: list[str]
    chapters: list[str]
    thumbnail_concept: str
    quality_score: float
    approved: bool
    error: str | None
    patterns: list[dict]
    video_format: str


# ---------------------------------------------------------------------------
# Quality scoring helpers (deterministic — ported from v1)
# ---------------------------------------------------------------------------

QUALITY_WEIGHTS = {
    "has_title": 0.15,
    "title_length_ok": 0.10,
    "has_script_outline": 0.15,
    "has_description": 0.10,
    "has_tags": 0.10,
    "tag_count_in_range": 0.05,
    "has_thumbnail_concept": 0.10,
    "no_prohibited_content": 0.15,
    "has_cta_in_outline": 0.10,
}

MIN_QUALITY_SCORE = 0.70

VIDEO_FORMATS = ["tutorial", "listicle", "explainer", "compilation", "commentary"]

DEFAULT_TAGS = [
    "AI tools", "machine learning", "artificial intelligence", "automation",
    "tech 2026", "FiestyGoatAI", "AI education",
]


def _score_title_ctr(title: str, topic: str) -> float:
    """Score a YouTube title for click-through potential."""
    score = 0.0
    if 40 <= len(title) <= 70:
        score += 0.30
    power_words = ["complete", "guide", "best", "top", "secret", "hidden", "ultimate",
                   "free", "beginners", "advanced", "2026", "how to", "why"]
    if any(w in title.lower() for w in power_words):
        score += 0.25
    if topic.lower() in title.lower():
        score += 0.20
    if title[0].isupper():
        score += 0.10
    numbers = [str(n) for n in range(3, 20)]
    if any(n in title for n in numbers):
        score += 0.15
    return round(score, 4)


def _deterministic_quality(state: YouTubeState) -> float:
    score = 0.0
    title = state.get("title", "")
    outline = state.get("script_outline", "")
    description = state.get("description", "")
    tags = state.get("tags", [])
    thumbnail = state.get("thumbnail_concept", "")
    topic = state.get("topic", "")

    if title.strip():
        score += QUALITY_WEIGHTS["has_title"]
    if 30 <= len(title) <= 100:
        score += QUALITY_WEIGHTS["title_length_ok"]
    if outline.strip():
        score += QUALITY_WEIGHTS["has_script_outline"]
    if description.strip():
        score += QUALITY_WEIGHTS["has_description"]
    if tags:
        score += QUALITY_WEIGHTS["has_tags"]
    if 5 <= len(tags) <= 30:
        score += QUALITY_WEIGHTS["tag_count_in_range"]
    if thumbnail.strip():
        score += QUALITY_WEIGHTS["has_thumbnail_concept"]
    banned = ["hate speech", "financial advice", "medical advice"]
    if not any(b in (title + " " + topic).lower() for b in banned):
        score += QUALITY_WEIGHTS["no_prohibited_content"]
    if "cta" in outline.lower() or "call to action" in outline.lower() or "subscribe" in outline.lower():
        score += QUALITY_WEIGHTS["has_cta_in_outline"]

    return round(score, 4)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def research_niche(state: YouTubeState) -> dict:
    """Search AutoMem for YouTube SEO and format patterns."""
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
            SELECT description, confidence_score, metadata
            FROM patterns
            WHERE context ILIKE %s OR description ILIKE %s
            ORDER BY confidence_score DESC
            LIMIT 5
            """,
            (f"%youtube%", f"%{topic[:30]}%"),
        )
        rows = cur.fetchall()
        patterns = [dict(r) for r in rows]
        conn.close()
        log.info("youtube.research_niche", topic=topic, patterns=len(patterns))
    except Exception as exc:
        log.warning("youtube.research_niche.failed", error=str(exc))

    niche = "ai" if "ai" in topic.lower() else "tech"
    video_format = "tutorial"
    if "list" in topic.lower() or "top" in topic.lower():
        video_format = "listicle"
    elif "explain" in topic.lower() or "what is" in topic.lower():
        video_format = "explainer"

    return {"patterns": patterns, "niche": niche, "video_format": video_format}


async def generate_title(state: YouTubeState) -> dict:
    """Generate 3 title variants; pick highest CTR score."""
    topic = state.get("topic", "AI tools")
    video_format = state.get("video_format", "tutorial")
    patterns = state.get("patterns", [])

    pattern_context = "\n".join(
        f"- {p.get('description', '')[:100]}" for p in patterns[:3]
    ) if patterns else "No historical patterns available."

    prompt = (
        f"Generate a YouTube video title for a {video_format} about: {topic}\n\n"
        f"Historical best-performing patterns:\n{pattern_context}\n\n"
        f"Requirements:\n"
        f"- 40-70 characters\n"
        f"- Include a power word (Complete, Ultimate, Best, Secret, How to, etc.)\n"
        f"- Include year 2026 if relevant\n"
        f"- Front-load the primary keyword\n"
        f"- One title only, no explanation\n\n"
        f"Title:"
    )

    router = LLMRouter()
    best_title = f"{topic.title()} — Complete Guide 2026"
    best_score = 0.0

    try:
        variants = await asyncio.gather(*[
            router.complete(LLMRequest(prompt=prompt, temperature=0.85))
            for _ in range(3)
        ])

        for variant in variants:
            text = variant.content.strip().split("\n")[0]
            score = _score_title_ctr(text, topic)
            log.debug("youtube.title_variant", score=score, title=text[:60])
            if score > best_score:
                best_score = score
                best_title = text

        log.info("youtube.generate_title", best_score=best_score)
    except Exception as exc:
        log.warning("youtube.generate_title.failed", error=str(exc))

    return {"title": best_title}


async def write_script_outline(state: YouTubeState) -> dict:
    """Write section-by-section outline: hook, value sections, CTA."""
    topic = state.get("topic", "AI tools")
    title = state.get("title", topic)
    video_format = state.get("video_format", "tutorial")

    prompt = (
        f"Write a detailed YouTube video script outline for:\n"
        f"Title: {title}\n"
        f"Format: {video_format}\n\n"
        f"Include these sections:\n"
        f"1. HOOK (0:00-0:30): Attention-grabbing opening statement or question\n"
        f"2. INTRO (0:30-1:00): What viewers will learn + credibility\n"
        f"3. MAIN CONTENT (1:00-7:00): 3-5 value-packed sections with timestamps\n"
        f"4. VALUE RECAP (7:00-8:00): Key takeaways summary\n"
        f"5. CTA (8:00-8:30): Subscribe, like, affiliate link mention\n\n"
        f"Format each section with timestamp and 2-3 bullet points of talking points.\n"
        f"Outline:"
    )

    router = LLMRouter()
    outline = (
        f"# {title}\n\n"
        f"## HOOK (0:00-0:30)\n- Bold claim about {topic}\n- Preview what they'll learn\n\n"
        f"## INTRO (0:30-1:00)\n- Channel intro\n- Agenda for the video\n\n"
        f"## MAIN CONTENT (1:00-7:00)\n"
        f"### Part 1: What is {topic}?\n- Definition\n- Why it matters in 2026\n\n"
        f"### Part 2: How it Works\n- Step-by-step breakdown\n- Live demo\n\n"
        f"### Part 3: Top Use Cases\n- Real-world examples\n- Results and metrics\n\n"
        f"## VALUE RECAP (7:00-8:00)\n- Top 3 takeaways\n- What to do next\n\n"
        f"## CTA (8:00-8:30)\n- Subscribe for daily AI tools\n- Link in description\n- Like if helpful"
    )

    try:
        response = await router.complete(LLMRequest(prompt=prompt, max_tokens=600))
        if len(response.content.strip()) > 100:
            outline = response.content.strip()
        log.info("youtube.write_script_outline", length=len(outline))
    except Exception as exc:
        log.warning("youtube.write_script_outline.failed", error=str(exc))

    return {"script_outline": outline}


def optimize_seo(state: YouTubeState) -> dict:
    """Generate meta description, keyword-rich tags, and chapter timestamps."""
    topic = state.get("topic", "AI tools")
    title = state.get("title", topic)
    niche = state.get("niche", "tech")

    meta_description = (
        f"Discover everything about {topic} in this complete 2026 guide. "
        f"Learn how {topic} works, real use cases, and how to get started today. "
        f"Links and resources in description. | FiestyGoat AI"
    )[:500]

    tags = [
        topic,
        f"best {topic}",
        f"{topic} tutorial",
        f"{topic} 2026",
        f"how to use {topic}",
        f"{topic} for beginners",
        f"{topic} guide",
        niche,
        "AI tools",
        "FiestyGoatAI",
        "artificial intelligence",
        "machine learning",
        "automation",
        "tech tutorial",
        "AI education",
    ]

    chapters = [
        "0:00 Introduction",
        "0:30 What You'll Learn",
        "1:00 Main Content",
        "4:00 Advanced Tips",
        "7:00 Key Takeaways",
        "8:00 Final Thoughts & CTA",
    ]

    log.info("youtube.optimize_seo", tags=len(tags))
    return {"description": meta_description, "tags": tags, "chapters": chapters}


def create_thumbnail_concept(state: YouTubeState) -> dict:
    """Describe thumbnail layout for the design team / AI image gen."""
    topic = state.get("topic", "AI tools")
    title = state.get("title", topic)

    short_title = title[:25] + ("..." if len(title) > 25 else "")
    concept = (
        f"THUMBNAIL CONCEPT:\n"
        f"- Background: Bold gradient (deep blue → purple)\n"
        f"- Left half: Large bold white text — '{short_title}'\n"
        f"- Right half: Shocked face emoji or surprised person\n"
        f"- Accent: Orange/yellow lightning bolt or arrow\n"
        f"- Bottom left: FiestyGoat AI logo (small)\n"
        f"- Font: Impact or Bebas Neue, uppercase\n"
        f"- Power word badge (red banner): 'FREE' or '2026' or 'UPDATED'\n"
        f"- Dimensions: 1280x720px"
    )

    log.info("youtube.create_thumbnail_concept")
    return {"thumbnail_concept": concept}


async def quality_gate(state: YouTubeState) -> dict:
    """Score content; push to 'youtube_publish' Redis queue if approved."""
    score = _deterministic_quality(state)
    approved = score >= MIN_QUALITY_SCORE

    log.info("youtube.quality_gate", score=score, approved=approved)

    if approved:
        try:
            import redis.asyncio as aioredis
            from config.settings import get_settings

            settings = get_settings()
            r = aioredis.from_url(settings.redis_url)
            payload = json.dumps({
                "platform": "youtube",
                "topic": state.get("topic"),
                "title": state.get("title"),
                "script_outline": state.get("script_outline"),
                "description": state.get("description"),
                "tags": state.get("tags"),
                "chapters": state.get("chapters"),
                "thumbnail_concept": state.get("thumbnail_concept"),
                "quality_score": score,
                "video_format": state.get("video_format"),
            })
            await r.lpush("youtube_publish", payload)
            await r.aclose()
            log.info("youtube.queued_to_publish", score=score)
        except Exception as exc:
            log.warning("youtube.redis_push.failed", error=str(exc))

    return {"quality_score": score, "approved": approved}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_youtube_agent():
    """Build and compile the YouTube specialist subgraph."""
    graph = StateGraph(YouTubeState)

    graph.add_node("research_niche", research_niche)
    graph.add_node("generate_title", generate_title)
    graph.add_node("write_script_outline", write_script_outline)
    graph.add_node("optimize_seo", optimize_seo)
    graph.add_node("create_thumbnail_concept", create_thumbnail_concept)
    graph.add_node("quality_gate", quality_gate)

    graph.set_entry_point("research_niche")
    graph.add_edge("research_niche", "generate_title")
    graph.add_edge("generate_title", "write_script_outline")
    graph.add_edge("write_script_outline", "optimize_seo")
    graph.add_edge("optimize_seo", "create_thumbnail_concept")
    graph.add_edge("create_thumbnail_concept", "quality_gate")
    graph.add_edge("quality_gate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# AgentRunner — Redis consumer on "youtube_task" queue
# ---------------------------------------------------------------------------

class AgentRunner:
    def __init__(self):
        self.agent = build_youtube_agent()

    async def run(self):
        import redis.asyncio as aioredis
        from config.settings import get_settings

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        log.info("youtube_runner.started", queue="youtube_task")

        while True:
            try:
                raw = await r.brpop("youtube_task", timeout=10)
                if raw is None:
                    continue

                _, data = raw
                task = json.loads(data)
                log.info("youtube_runner.task_received", topic=task.get("topic"))

                initial_state: YouTubeState = {
                    "messages": [],
                    "topic": task.get("topic", "AI tools"),
                    "niche": task.get("niche", "tech"),
                    "title": "",
                    "script_outline": "",
                    "description": "",
                    "tags": [],
                    "chapters": [],
                    "thumbnail_concept": "",
                    "quality_score": 0.0,
                    "approved": False,
                    "error": None,
                    "patterns": [],
                    "video_format": task.get("video_format", "tutorial"),
                }

                result = await self.agent.ainvoke(initial_state)
                log.info(
                    "youtube_runner.task_complete",
                    approved=result.get("approved"),
                    score=result.get("quality_score"),
                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("youtube_runner.error", error=str(exc))
                await asyncio.sleep(2)

        await r.aclose()


youtube_agent = build_youtube_agent()


if __name__ == "__main__":
    async def _main():
        runner = AgentRunner()
        await runner.run()

    asyncio.run(_main())
