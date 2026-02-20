"""NewsletterAgent — Tier 3 specialist subgraph for weekly AI tools roundup.

Nodes:
  1. curate_tools        — Search AutoMem for top AI tools discovered this week
  2. write_roundup       — 1500-2000 word weekly AI tools roundup via LLM
  3. add_affiliate_links — Insert tracked affiliate URLs
  4. validate_compliance — CAN-SPAM check (unsubscribe link, physical address)
  5. schedule_send       — Push to 'newsletter_queue' with next Thursday send time
"""
from __future__ import annotations

import asyncio
import json
import structlog
from datetime import datetime, timedelta
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from jarvis.core.router import LLMRouter, LLMRequest

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class NewsletterState(TypedDict):
    messages: Annotated[list, add_messages]
    week_label: str
    tools: list[dict]
    body: str
    word_count: int
    affiliate_links: list[dict]
    compliant: bool
    compliance_issues: list[str]
    quality_score: float
    approved: bool
    scheduled_at: str
    error: str | None


# ---------------------------------------------------------------------------
# Quality thresholds (ported from v1)
# ---------------------------------------------------------------------------

MIN_WORD_COUNT = 1500
MAX_WORD_COUNT = 2000
MIN_QUALITY_SCORE = 0.70

SPAM_TRIGGERS = [
    "free money", "act now", "limited time", "click here",
    "guarantee", "risk-free", "no cost", "earn $$$",
    "make money fast", "buy now", "order now",
]

QUALITY_WEIGHTS = {
    "word_count_in_range": 0.20,
    "has_all_required_sections": 0.20,
    "tool_count_ok": 0.10,
    "has_unsubscribe_link": 0.20,
    "has_physical_address": 0.15,
    "no_spam_triggers": 0.10,
    "has_affiliate_disclosure": 0.05,
}

REQUIRED_SECTIONS = ["intro", "tool", "call to action", "unsubscribe"]

PHYSICAL_ADDRESS = "FiestyGoat AI LLC, California, USA"
UNSUBSCRIBE_PLACEHOLDER = "{{unsubscribe_url}}"

LINKTREE_URL = "https://linktr.ee/fiestygoatai"

DEFAULT_AFFILIATE_LINKS = [
    {"name": "Top AI Tools Hub", "url": LINKTREE_URL, "tracked": True},
]


def _next_thursday() -> str:
    """Return ISO timestamp of the next Thursday at 9:00 AM PT."""
    now = datetime.utcnow()
    days_ahead = (3 - now.weekday()) % 7 or 7
    next_thu = now + timedelta(days=days_ahead)
    return next_thu.replace(hour=17, minute=0, second=0, microsecond=0).isoformat()


def _deterministic_quality(state: NewsletterState) -> float:
    score = 0.0
    body = state.get("body", "")
    word_count = state.get("word_count", 0)
    tools = state.get("tools", [])
    body_lower = body.lower()

    if MIN_WORD_COUNT <= word_count <= MAX_WORD_COUNT:
        score += QUALITY_WEIGHTS["word_count_in_range"]
    if all(sec in body_lower for sec in REQUIRED_SECTIONS):
        score += QUALITY_WEIGHTS["has_all_required_sections"]
    if 3 <= len(tools) <= 10:
        score += QUALITY_WEIGHTS["tool_count_ok"]
    if "unsubscribe" in body_lower:
        score += QUALITY_WEIGHTS["has_unsubscribe_link"]
    if "fiestygoat" in body_lower or "california" in body_lower:
        score += QUALITY_WEIGHTS["has_physical_address"]
    if not any(t in body_lower for t in SPAM_TRIGGERS):
        score += QUALITY_WEIGHTS["no_spam_triggers"]
    if "affiliate" in body_lower or "#ad" in body_lower:
        score += QUALITY_WEIGHTS["has_affiliate_disclosure"]

    return round(score, 4)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def curate_tools(state: NewsletterState) -> dict:
    """Search AutoMem + ChromaDB for top AI tools discovered this week."""
    tools: list[dict] = []
    week_label = state.get("week_label", datetime.utcnow().strftime("Week of %B %d, %Y"))

    try:
        import psycopg2
        import psycopg2.extras
        from config.settings import get_settings

        settings = get_settings()
        conn = psycopg2.connect(settings.postgres_url)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT title, summary, context
            FROM conversations
            WHERE outcome IN ('article_stored', 'discovery', 'learning_stored')
              AND created_at > NOW() - INTERVAL '7 days'
            ORDER BY created_at DESC
            LIMIT 10
            """,
        )
        rows = cur.fetchall()
        conn.close()

        for row in rows:
            ctx = row.get("context") or {}
            tools.append({
                "name": (ctx.get("tool_name") or row.get("title") or "AI Tool")[:80],
                "description": (row.get("summary") or "")[:300],
                "affiliate_link": LINKTREE_URL,
            })
        log.info("newsletter.curate_tools", tools_found=len(tools))
    except Exception as exc:
        log.warning("newsletter.curate_tools.failed", error=str(exc))

    if not tools:
        tools = [
            {"name": "Claude Code", "description": "Anthropic's AI coding assistant — transforms how developers write production code.", "affiliate_link": LINKTREE_URL},
            {"name": "Cursor AI", "description": "AI-first code editor with natural language editing and multi-file context.", "affiliate_link": LINKTREE_URL},
            {"name": "Perplexity AI", "description": "AI-powered search engine that cites sources in real time.", "affiliate_link": LINKTREE_URL},
            {"name": "Kling AI", "description": "State-of-the-art AI video generation from text prompts.", "affiliate_link": LINKTREE_URL},
            {"name": "Manus AI", "description": "Fully autonomous AI agent that browses, codes, and acts on the web.", "affiliate_link": LINKTREE_URL},
        ]

    return {"tools": tools, "week_label": week_label}


async def write_roundup(state: NewsletterState) -> dict:
    """Generate a 1500-2000 word AI tools roundup using LLM."""
    tools = state.get("tools", [])
    week_label = state.get("week_label", "This Week")

    tools_list = "\n".join(
        f"- {t['name']}: {t.get('description', '')[:150]}"
        for t in tools[:8]
    )

    prompt = (
        f"Write a weekly AI tools newsletter for FiestyGoat AI — {week_label}.\n\n"
        f"Tools to feature:\n{tools_list}\n\n"
        f"Structure:\n"
        f"## Intro\nEngage readers, tease the week's top finds (2-3 sentences)\n\n"
        f"## Tool Summaries\nFor each tool: name, what it does, best use case, pricing, affiliate link reference\n\n"
        f"## Exclusive Insights\nUnderrated feature or workflow combining 2+ tools (150-200 words)\n\n"
        f"## Call to Action\nUpgrade to premium for deeper analysis, links, and early access ($15/mo)\n\n"
        f"Requirements:\n"
        f"- 1500-2000 words total\n"
        f"- Professional but conversational tone\n"
        f"- Each tool section: 200-300 words\n"
        f"- Include 'Affiliate Disclosure: This newsletter contains affiliate links.'\n"
        f"- Do NOT include unsubscribe link (added in compliance step)\n\n"
        f"Newsletter:"
    )

    router = LLMRouter()

    body = (
        f"# This Week in AI Tools — {week_label}\n\n"
        f"## Intro\n"
        f"Welcome back to FiestyGoat AI's weekly roundup of the most powerful AI tools discovered this week. "
        f"We've tested these so you don't have to — let's dive in.\n\n"
        f"*Affiliate Disclosure: This newsletter contains affiliate links. "
        f"We may earn a commission at no extra cost to you.*\n\n"
        f"## Tool Summaries\n\n"
    )

    for tool in tools[:8]:
        body += (
            f"### {tool['name']}\n"
            f"{tool.get('description', 'A powerful new AI tool that...')}\n\n"
            f"**Best for:** Developers and productivity enthusiasts\n"
            f"**Try it:** [Check it out here]({tool.get('affiliate_link', LINKTREE_URL)})\n\n"
        )

    body += (
        f"## Exclusive Insights\n"
        f"This week's power combo: Use Perplexity AI to research your topic, "
        f"then Claude Code to implement it — cutting research-to-deployment time by 60%.\n\n"
        f"## Call to Action\n"
        f"Ready to go deeper? Upgrade to FiestyGoat AI Premium for weekly deep-dives, "
        f"early access to new tools, and 1-on-1 AI strategy sessions.\n"
        f"[Upgrade to Premium — $15/month]({LINKTREE_URL})\n\n"
    )

    try:
        response = await router.complete(LLMRequest(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.7,
        ))
        if len(response.content.strip()) > 500:
            body = response.content.strip()
        log.info("newsletter.write_roundup", length=len(body))
    except Exception as exc:
        log.warning("newsletter.write_roundup.failed", error=str(exc))

    word_count = len(body.split())
    return {"body": body, "word_count": word_count}


def add_affiliate_links(state: NewsletterState) -> dict:
    """Insert tracked affiliate URLs and append link block."""
    body = state.get("body", "")
    tools = state.get("tools", [])
    affiliate_links = state.get("affiliate_links", []) or DEFAULT_AFFILIATE_LINKS

    link_block = "\n\n---\n**Quick Links:**\n"
    for al in affiliate_links:
        link_block += f"- [{al['name']}]({al['url']})\n"

    body += link_block
    log.info("newsletter.add_affiliate_links", links=len(affiliate_links))
    return {"body": body, "affiliate_links": affiliate_links}


def validate_compliance(state: NewsletterState) -> dict:
    """CAN-SPAM compliance: unsubscribe link + physical address."""
    body = state.get("body", "")
    issues: list[str] = []

    if "unsubscribe" not in body.lower():
        body += (
            f"\n\n---\n"
            f"[Unsubscribe]({UNSUBSCRIBE_PLACEHOLDER}) | "
            f"{PHYSICAL_ADDRESS}\n"
            f"You're receiving this because you subscribed to FiestyGoat AI Newsletter."
        )
        log.info("newsletter.compliance.unsubscribe_added")
    else:
        if PHYSICAL_ADDRESS.lower() not in body.lower() and "california" not in body.lower():
            body += f"\n\n{PHYSICAL_ADDRESS}"
            issues.append("physical_address_appended")

    spam_found = [t for t in SPAM_TRIGGERS if t in body.lower()]
    if spam_found:
        issues.extend([f"spam_trigger:{t}" for t in spam_found])
        log.warning("newsletter.compliance.spam_triggers", triggers=spam_found)

    compliant = len(spam_found) == 0
    word_count = len(body.split())

    log.info("newsletter.validate_compliance", compliant=compliant, issues=issues, words=word_count)
    return {
        "body": body,
        "word_count": word_count,
        "compliant": compliant,
        "compliance_issues": issues,
    }


async def schedule_send(state: NewsletterState) -> dict:
    """Push newsletter to 'newsletter_queue' Redis key with next Thursday send time."""
    score = _deterministic_quality(state)
    approved = score >= MIN_QUALITY_SCORE and state.get("compliant", False)
    scheduled_at = _next_thursday()

    log.info("newsletter.schedule_send", score=score, approved=approved, send_at=scheduled_at)

    if approved:
        try:
            import redis.asyncio as aioredis
            from config.settings import get_settings

            settings = get_settings()
            r = aioredis.from_url(settings.redis_url)
            payload = json.dumps({
                "platform": "newsletter",
                "week_label": state.get("week_label"),
                "body": state.get("body"),
                "word_count": state.get("word_count"),
                "tools_featured": len(state.get("tools", [])),
                "affiliate_links": state.get("affiliate_links"),
                "quality_score": score,
                "scheduled_at": scheduled_at,
                "compliant": state.get("compliant"),
            })
            await r.lpush("newsletter_queue", payload)
            await r.aclose()
            log.info("newsletter.queued", scheduled_at=scheduled_at)
        except Exception as exc:
            log.warning("newsletter.redis_push.failed", error=str(exc))

    return {
        "quality_score": score,
        "approved": approved,
        "scheduled_at": scheduled_at,
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_newsletter_agent():
    """Build and compile the Newsletter specialist subgraph."""
    graph = StateGraph(NewsletterState)

    graph.add_node("curate_tools", curate_tools)
    graph.add_node("write_roundup", write_roundup)
    graph.add_node("add_affiliate_links", add_affiliate_links)
    graph.add_node("validate_compliance", validate_compliance)
    graph.add_node("schedule_send", schedule_send)

    graph.set_entry_point("curate_tools")
    graph.add_edge("curate_tools", "write_roundup")
    graph.add_edge("write_roundup", "add_affiliate_links")
    graph.add_edge("add_affiliate_links", "validate_compliance")
    graph.add_edge("validate_compliance", "schedule_send")
    graph.add_edge("schedule_send", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# AgentRunner — Redis consumer on "newsletter_task" queue
# ---------------------------------------------------------------------------

class AgentRunner:
    def __init__(self):
        self.agent = build_newsletter_agent()

    async def run(self):
        import redis.asyncio as aioredis
        from config.settings import get_settings

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        log.info("newsletter_runner.started", queue="newsletter_task")

        while True:
            try:
                raw = await r.brpop("newsletter_task", timeout=10)
                if raw is None:
                    continue

                _, data = raw
                task = json.loads(data)
                log.info("newsletter_runner.task_received", week=task.get("week_label"))

                initial_state: NewsletterState = {
                    "messages": [],
                    "week_label": task.get("week_label", datetime.utcnow().strftime("Week of %B %d, %Y")),
                    "tools": task.get("tools", []),
                    "body": "",
                    "word_count": 0,
                    "affiliate_links": task.get("affiliate_links", []),
                    "compliant": False,
                    "compliance_issues": [],
                    "quality_score": 0.0,
                    "approved": False,
                    "scheduled_at": "",
                    "error": None,
                }

                result = await self.agent.ainvoke(initial_state)
                log.info(
                    "newsletter_runner.task_complete",
                    approved=result.get("approved"),
                    score=result.get("quality_score"),
                    words=result.get("word_count"),
                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("newsletter_runner.error", error=str(exc))
                await asyncio.sleep(2)

        await r.aclose()


newsletter_agent = build_newsletter_agent()


if __name__ == "__main__":
    async def _main():
        runner = AgentRunner()
        await runner.run()

    asyncio.run(_main())
