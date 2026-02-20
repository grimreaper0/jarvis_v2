"""SEOBlogAgent — Tier 3 specialist for long-form AI tool content.

Nodes:
  1. keyword_research  — Seed + LSI keyword discovery from topic
  2. outline          — Section-by-section article outline
  3. write_1500_words  — LLM drafts full 1500-2500 word article
  4. on_page_seo       — Meta title, meta description, internal link anchors
  5. schema_markup     — Article + FAQ JSON-LD schema
  6. publish_queue     — Push to 'seo_publish' Redis queue

Target: Evergreen Google traffic + affiliate revenue via WordPress.
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

class SEOBlogState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    primary_keyword: str
    secondary_keywords: list[str]
    lsi_keywords: list[str]
    outline: list[dict]
    body: str
    word_count: int
    meta_title: str
    meta_description: str
    slug: str
    internal_links: list[dict]
    schema: dict[str, Any]
    quality_score: float
    approved: bool
    error: str | None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_WORD_MIN = 1500
TARGET_WORD_MAX = 2500
MIN_QUALITY_SCORE = 0.70

QUALITY_WEIGHTS = {
    "word_count_in_range": 0.20,
    "has_primary_keyword_in_title": 0.15,
    "has_meta_description": 0.10,
    "has_sections": 0.15,
    "has_schema": 0.15,
    "has_internal_links": 0.10,
    "has_cta": 0.15,
}

AFFILIATE_URL = "https://linktr.ee/fiestygoatai"

INTERNAL_LINK_ANCHORS = [
    {"anchor": "best AI tools", "url": "/best-ai-tools-2026"},
    {"anchor": "AI automation guide", "url": "/ai-automation-beginners-guide"},
    {"anchor": "FiestyGoat AI Newsletter", "url": "/newsletter"},
]


def _slugify(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:60]


def _deterministic_quality(state: SEOBlogState) -> float:
    score = 0.0
    body = state.get("body", "")
    word_count = state.get("word_count", 0)
    meta_title = state.get("meta_title", "")
    meta_desc = state.get("meta_description", "")
    pk = state.get("primary_keyword", "").lower()
    outline = state.get("outline", [])
    schema = state.get("schema", {})
    internal_links = state.get("internal_links", [])

    if TARGET_WORD_MIN <= word_count <= TARGET_WORD_MAX:
        score += QUALITY_WEIGHTS["word_count_in_range"]
    if pk and pk in meta_title.lower():
        score += QUALITY_WEIGHTS["has_primary_keyword_in_title"]
    if meta_desc.strip():
        score += QUALITY_WEIGHTS["has_meta_description"]
    if len(outline) >= 4:
        score += QUALITY_WEIGHTS["has_sections"]
    if schema:
        score += QUALITY_WEIGHTS["has_schema"]
    if internal_links:
        score += QUALITY_WEIGHTS["has_internal_links"]
    if "affiliate" in body.lower() or "check it out" in body.lower() or "link" in body.lower():
        score += QUALITY_WEIGHTS["has_cta"]

    return round(score, 4)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def keyword_research(state: SEOBlogState) -> dict:
    """Generate primary + LSI keywords from topic using deterministic expansion."""
    topic = state.get("topic", "AI tools")
    topic_lower = topic.lower()
    current_year = "2026"

    primary = topic
    secondary = [
        f"best {topic}",
        f"{topic} review",
        f"how to use {topic}",
        f"{topic} tutorial",
        f"{topic} {current_year}",
    ]
    lsi = [
        f"{topic} pricing",
        f"{topic} alternatives",
        f"{topic} vs competitors",
        f"{topic} features",
        f"{topic} for beginners",
        f"is {topic} worth it",
        f"{topic} free plan",
        f"AI tool {topic.split()[-1] if ' ' in topic else topic}",
    ]

    slug = _slugify(f"best-{topic}-{current_year}-complete-guide")

    log.info("seo_blog.keyword_research", primary=primary, secondary_count=len(secondary))
    return {
        "primary_keyword": primary,
        "secondary_keywords": secondary,
        "lsi_keywords": lsi,
        "slug": slug,
    }


async def outline(state: SEOBlogState) -> dict:
    """Generate section-by-section article outline."""
    topic = state.get("topic", "AI tools")
    primary_keyword = state.get("primary_keyword", topic)
    secondary = state.get("secondary_keywords", [])

    prompt = (
        f"Create a detailed SEO blog post outline for: '{primary_keyword}'\n\n"
        f"Secondary keywords to include: {', '.join(secondary[:5])}\n\n"
        f"Outline structure (7-9 sections):\n"
        f"1. Introduction (150 words) — hook + what reader will learn\n"
        f"2. What Is [Topic]? (200 words) — definition + why it matters\n"
        f"3. Key Features (300 words) — top 5-7 features with subheadings\n"
        f"4. How to Get Started (250 words) — step-by-step guide\n"
        f"5. Pricing Plans (150 words) — free vs paid tiers\n"
        f"6. Pros and Cons (200 words) — balanced assessment\n"
        f"7. Use Cases (200 words) — 3 real-world examples\n"
        f"8. [Topic] Alternatives (150 words) — 3 alternatives with affiliate links\n"
        f"9. Conclusion + CTA (100 words) — summary + affiliate CTA\n\n"
        f"Return as a JSON array of objects with 'heading' and 'target_words' keys.\n"
        f"Example: [{{\"heading\": \"Introduction\", \"target_words\": 150}}, ...]\n\n"
        f"JSON:"
    )

    default_outline = [
        {"heading": "Introduction", "target_words": 150},
        {"heading": f"What Is {topic}?", "target_words": 200},
        {"heading": "Key Features", "target_words": 300},
        {"heading": "How to Get Started (Step-by-Step)", "target_words": 250},
        {"heading": "Pricing Plans", "target_words": 150},
        {"heading": "Pros and Cons", "target_words": 200},
        {"heading": "Real-World Use Cases", "target_words": 200},
        {"heading": f"{topic} Alternatives", "target_words": 150},
        {"heading": "Conclusion", "target_words": 100},
    ]

    try:
        router = LLMRouter()
        response = await router.complete(LLMRequest(prompt=prompt, max_tokens=600, temperature=0.5))
        text = response.content.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            if isinstance(parsed, list) and len(parsed) >= 4:
                default_outline = parsed
        log.info("seo_blog.outline", sections=len(default_outline))
    except Exception as exc:
        log.warning("seo_blog.outline.failed", error=str(exc))

    return {"outline": default_outline}


async def write_1500_words(state: SEOBlogState) -> dict:
    """Write the full 1500-2500 word article using LLM."""
    topic = state.get("topic", "AI tools")
    primary_keyword = state.get("primary_keyword", topic)
    secondary = state.get("secondary_keywords", [])
    lsi = state.get("lsi_keywords", [])
    article_outline = state.get("outline", [])
    year = "2026"

    outline_text = "\n".join(
        f"{i+1}. {sec['heading']} (~{sec['target_words']} words)"
        for i, sec in enumerate(article_outline)
    )

    prompt = (
        f"Write a complete, SEO-optimized blog post about: '{primary_keyword}'\n\n"
        f"Target: 1,500-2,000 words\n"
        f"Primary keyword: {primary_keyword}\n"
        f"Secondary keywords: {', '.join(secondary[:4])}\n"
        f"LSI keywords to sprinkle in: {', '.join(lsi[:6])}\n\n"
        f"Outline:\n{outline_text}\n\n"
        f"Formatting requirements:\n"
        f"- Use ## for H2 headings, ### for H3 subheadings\n"
        f"- Include a comparison table where relevant\n"
        f"- Add bullet points for features and pros/cons\n"
        f"- Include affiliate CTA: 'Try {topic} for free → [affiliate link]'\n"
        f"- Natural keyword density: 1-2% for primary keyword\n"
        f"- Include '{year}' in the introduction\n"
        f"- Conversational but authoritative tone\n"
        f"- End with strong affiliate CTA pointing to FiestyGoat AI Linktree\n\n"
        f"Article:"
    )

    body = (
        f"# Best {primary_keyword.title()} in {year}: Complete Guide\n\n"
        f"## Introduction\n"
        f"Looking for the best {primary_keyword} in {year}? You're in the right place. "
        f"In this guide, we'll cover everything you need to know — from key features to pricing — "
        f"so you can make an informed decision.\n\n"
    )

    for section in article_outline[1:]:
        h = section.get("heading", "Section")
        words = section.get("target_words", 150)
        body += (
            f"## {h}\n"
            f"[{words}-word section about {h.lower()} covering {primary_keyword}...]\n\n"
        )

    body += (
        f"## Conclusion\n"
        f"Ready to get started with {primary_keyword}? "
        f"Try it free through our affiliate link and let us know what you think.\n\n"
        f"**[Try {topic} Now — Free Plan Available]({AFFILIATE_URL})**\n\n"
        f"---\n*FiestyGoat AI — AI Education for Everyone*\n"
    )

    try:
        router = LLMRouter()
        response = await router.complete(LLMRequest(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.7,
        ))
        if len(response.content.strip()) > 500:
            body = response.content.strip()
        log.info("seo_blog.write_1500_words", length=len(body))
    except Exception as exc:
        log.warning("seo_blog.write_1500_words.failed", error=str(exc))

    word_count = len(body.split())
    return {"body": body, "word_count": word_count}


def on_page_seo(state: SEOBlogState) -> dict:
    """Generate meta title (50-60 chars), meta description (150-160 chars), internal links."""
    topic = state.get("topic", "AI tools")
    primary_keyword = state.get("primary_keyword", topic)
    year = "2026"

    raw_title = f"Best {primary_keyword.title()} in {year}: Complete Guide"
    meta_title = raw_title[:60]

    meta_description = (
        f"Discover the best {primary_keyword} in {year}. "
        f"Expert review covering features, pricing, pros & cons, and step-by-step setup. "
        f"Updated {year}."
    )[:160]

    internal_links = [
        link for link in INTERNAL_LINK_ANCHORS
        if any(word in state.get("body", "").lower() for word in link["anchor"].split()[:2])
    ] or INTERNAL_LINK_ANCHORS[:2]

    log.info("seo_blog.on_page_seo", title_len=len(meta_title), desc_len=len(meta_description))
    return {
        "meta_title": meta_title,
        "meta_description": meta_description,
        "internal_links": internal_links,
    }


def schema_markup(state: SEOBlogState) -> dict:
    """Generate Article + FAQPage JSON-LD schema."""
    primary_keyword = state.get("primary_keyword", "AI tools")
    meta_title = state.get("meta_title", primary_keyword)
    meta_description = state.get("meta_description", "")
    slug = state.get("slug", "")

    schema = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "Article",
                "headline": meta_title,
                "description": meta_description,
                "author": {
                    "@type": "Organization",
                    "name": "FiestyGoat AI LLC",
                    "url": "https://fiestygoat.ai",
                },
                "publisher": {
                    "@type": "Organization",
                    "name": "FiestyGoat AI LLC",
                    "logo": {
                        "@type": "ImageObject",
                        "url": "https://fiestygoat.ai/logo.png",
                    },
                },
                "dateModified": "2026-01-01",
                "url": f"https://fiestygoat.ai/{slug}",
            },
            {
                "@type": "FAQPage",
                "mainEntity": [
                    {
                        "@type": "Question",
                        "name": f"What is {primary_keyword}?",
                        "acceptedAnswer": {
                            "@type": "Answer",
                            "text": f"{primary_keyword} is an AI-powered tool designed to help users automate and optimize their workflows.",
                        },
                    },
                    {
                        "@type": "Question",
                        "name": f"Is {primary_keyword} free?",
                        "acceptedAnswer": {
                            "@type": "Answer",
                            "text": f"{primary_keyword} offers a free plan with limited features. Premium plans start at varying price points.",
                        },
                    },
                ],
            },
        ],
    }

    log.info("seo_blog.schema_markup")
    return {"schema": schema}


async def publish_queue(state: SEOBlogState) -> dict:
    """Score content and push to 'seo_publish' Redis queue if approved."""
    score = _deterministic_quality(state)
    approved = score >= MIN_QUALITY_SCORE

    log.info("seo_blog.publish_queue", score=score, approved=approved, words=state.get("word_count"))

    if approved:
        try:
            import redis.asyncio as aioredis
            from config.settings import get_settings

            settings = get_settings()
            r = aioredis.from_url(settings.redis_url)
            payload = json.dumps({
                "platform": "wordpress",
                "topic": state.get("topic"),
                "primary_keyword": state.get("primary_keyword"),
                "secondary_keywords": state.get("secondary_keywords"),
                "meta_title": state.get("meta_title"),
                "meta_description": state.get("meta_description"),
                "slug": state.get("slug"),
                "body": state.get("body"),
                "word_count": state.get("word_count"),
                "schema": state.get("schema"),
                "internal_links": state.get("internal_links"),
                "quality_score": score,
            })
            await r.lpush("seo_publish", payload)
            await r.aclose()
            log.info("seo_blog.queued_to_publish", slug=state.get("slug"))
        except Exception as exc:
            log.warning("seo_blog.redis_push.failed", error=str(exc))

    return {"quality_score": score, "approved": approved}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_seo_blog_agent():
    """Build and compile the SEO Blog specialist subgraph."""
    graph = StateGraph(SEOBlogState)

    graph.add_node("keyword_research", keyword_research)
    graph.add_node("outline", outline)
    graph.add_node("write_1500_words", write_1500_words)
    graph.add_node("on_page_seo", on_page_seo)
    graph.add_node("schema_markup", schema_markup)
    graph.add_node("publish_queue", publish_queue)

    graph.set_entry_point("keyword_research")
    graph.add_edge("keyword_research", "outline")
    graph.add_edge("outline", "write_1500_words")
    graph.add_edge("write_1500_words", "on_page_seo")
    graph.add_edge("on_page_seo", "schema_markup")
    graph.add_edge("schema_markup", "publish_queue")
    graph.add_edge("publish_queue", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# AgentRunner — Redis consumer on "seo_task" queue
# ---------------------------------------------------------------------------

class AgentRunner:
    def __init__(self):
        self.agent = build_seo_blog_agent()

    async def run(self):
        import redis.asyncio as aioredis
        from config.settings import get_settings

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        log.info("seo_runner.started", queue="seo_task")

        while True:
            try:
                raw = await r.brpop("seo_task", timeout=10)
                if raw is None:
                    continue

                _, data = raw
                task = json.loads(data)
                log.info("seo_runner.task_received", topic=task.get("topic"))

                initial_state: SEOBlogState = {
                    "messages": [],
                    "topic": task.get("topic", "AI tools"),
                    "primary_keyword": "",
                    "secondary_keywords": [],
                    "lsi_keywords": [],
                    "outline": [],
                    "body": "",
                    "word_count": 0,
                    "meta_title": "",
                    "meta_description": "",
                    "slug": "",
                    "internal_links": [],
                    "schema": {},
                    "quality_score": 0.0,
                    "approved": False,
                    "error": None,
                }

                result = await self.agent.ainvoke(initial_state)
                log.info(
                    "seo_runner.task_complete",
                    approved=result.get("approved"),
                    score=result.get("quality_score"),
                    words=result.get("word_count"),
                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("seo_runner.error", error=str(exc))
                await asyncio.sleep(2)

        await r.aclose()


seo_blog_agent = build_seo_blog_agent()


if __name__ == "__main__":
    async def _main():
        runner = AgentRunner()
        await runner.run()

    asyncio.run(_main())
