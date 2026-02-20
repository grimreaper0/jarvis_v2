"""LearningWorker — 24/7 RSS + arXiv scraping, AutoMem/pgvector storage.

Queue: learning_scrape
Idle:  scrapes all 28 RSS feeds every hour
Storage: PostgreSQL (AutoMem conversations table with pgvector embeddings)
         ChromaDB is NOT used in v2 — pgvector only.
"""
import asyncio
import hashlib
import json
import structlog
from datetime import datetime

from jarvis.workers.base import ContinuousWorker

log = structlog.get_logger()

# Full feed list ported from jarvis_v1 LearningBot.FEEDS
RSS_FEEDS: list[dict] = [
    # AI Automation & Revenue Opportunities
    {"name": "ai_automation_trends",    "url": "https://www.marktechpost.com/feed/",                     "category": "revenue"},
    {"name": "automation_anywhere",     "url": "https://www.automationanywhere.com/rpa/blog/rss.xml",    "category": "revenue"},
    # Affiliate Marketing & E-commerce
    {"name": "affiliate_marketing_blog","url": "https://www.smartpassiveincome.com/feed/",               "category": "revenue"},
    {"name": "niche_pursuits",          "url": "https://www.nichepursuits.com/feed/",                    "category": "revenue"},
    {"name": "authority_hacker",        "url": "https://www.authorityhacker.com/feed/",                  "category": "revenue"},
    {"name": "ecommerce_fuel",          "url": "https://www.ecommercefuel.com/feed/",                    "category": "revenue"},
    # Content Creation & Social Media Revenue
    {"name": "social_media_examiner",   "url": "https://www.socialmediaexaminer.com/feed/",              "category": "revenue"},
    {"name": "vidiq_blog",              "url": "https://vidiq.com/blog/feed/",                           "category": "revenue"},
    {"name": "later_blog",              "url": "https://later.com/blog/feed/",                           "category": "revenue"},
    # YouTube Creation & AdSense Revenue
    {"name": "think_media",             "url": "https://www.thinkific.com/blog/feed/",                   "category": "revenue"},
    {"name": "tubefilter",              "url": "https://www.tubefilter.com/feed/",                       "category": "revenue"},
    {"name": "creator_academy",         "url": "https://blog.hootsuite.com/feed/",                       "category": "revenue"},
    {"name": "backlinko_blog",          "url": "https://backlinko.com/feed",                             "category": "revenue"},
    # Freelancing & Services
    {"name": "freelance_to_freedom",    "url": "https://freelancetofreedomproject.com/feed/",            "category": "revenue"},
    {"name": "freelancing_hacks",       "url": "https://www.freelancinghacks.com/feed/",                 "category": "revenue"},
    # AI Research — arXiv RSS
    {"name": "arxiv_ai",                "url": "http://export.arxiv.org/rss/cs.AI",                      "category": "ai"},
    {"name": "arxiv_ml",                "url": "http://export.arxiv.org/rss/cs.LG",                      "category": "ai"},
    {"name": "arxiv_multiagent",        "url": "http://export.arxiv.org/rss/cs.MA",                      "category": "ai"},
    # AI News & Research Labs
    {"name": "openai_blog",             "url": "https://openai.com/blog/rss/",                           "category": "ai"},
    {"name": "deepmind_blog",           "url": "https://deepmind.google/blog/rss.xml",                   "category": "ai"},
    {"name": "anthropic_research",      "url": "https://www.anthropic.com/news/rss",                     "category": "ai"},
    {"name": "meta_ai",                 "url": "https://ai.meta.com/blog/rss/",                          "category": "ai"},
    # AI Community & Trends
    {"name": "reddit_localllama",       "url": "https://www.reddit.com/r/LocalLLaMA/.rss",               "category": "ai"},
    {"name": "reddit_ml",               "url": "https://www.reddit.com/r/MachineLearning/.rss",          "category": "ai"},
    {"name": "reddit_ai",               "url": "https://www.reddit.com/r/artificial/.rss",               "category": "ai"},
    {"name": "reddit_singularity",      "url": "https://www.reddit.com/r/singularity/.rss",              "category": "ai"},
    # AI Industry News
    {"name": "huggingface_blog",        "url": "https://huggingface.co/blog/feed.xml",                   "category": "ai"},
    {"name": "ai_weekly",               "url": "https://aiweekly.co/rss",                                "category": "ai"},
]


def _content_hash(title: str, link: str) -> str:
    """MD5 dedup key — matches jarvis_v1 store_articles() logic."""
    return hashlib.md5(f"{title}{link}".encode()).hexdigest()


class LearningWorker(ContinuousWorker):
    """Continuously scrapes RSS feeds and arXiv, stores articles in AutoMem (pgvector).

    Queue:  learning_scrape  — accepts {type: scrape_feed | store_article | extract_patterns}
    Idle:   scrapes all 28 RSS feeds, then sleeps until next cycle
    Output: pushes to 'pattern_discovery' queue after storing each article
    """

    queue_name = "learning_scrape"
    worker_name = "learning_worker"

    async def process_task(self, task: dict) -> None:
        task_type = task.get("type")
        if task_type == "scrape_feed":
            await self._scrape_feed(task["url"], task.get("name", "unknown"), task.get("category", "general"))
        elif task_type == "store_article":
            await self._store_article(task["article"])
        elif task_type == "extract_patterns":
            await self._extract_patterns(task.get("articles", []))
        else:
            log.warning("learning.unknown_task", task_type=task_type)

    async def idle_loop(self) -> None:
        log.info("learning.idle_scrape_start", feed_count=len(RSS_FEEDS))
        success = 0
        fail = 0
        for feed in RSS_FEEDS:
            try:
                await self._scrape_feed(feed["url"], feed["name"], feed["category"])
                success += 1
                await asyncio.sleep(2)   # polite crawl delay
            except Exception as exc:
                fail += 1
                log.warning("learning.feed_error", feed=feed["name"], error=str(exc))

        log.info("learning.idle_scrape_done", success=success, fail=fail)
        await self.log_activity(
            action="idle_scrape_cycle",
            status="completed",
            details={"feeds_ok": success, "feeds_failed": fail, "total": len(RSS_FEEDS)},
        )

    # ==================== Feed scraping ====================

    async def _scrape_feed(self, url: str, name: str, category: str) -> None:
        import httpx
        import feedparser

        headers = {"User-Agent": "jarvis_v2/0.1 (+https://github.com/jarvis)"}
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        feed = feedparser.parse(resp.text)
        articles = []
        for entry in feed.entries[:20]:     # up to 20 per feed, same as v1
            title = entry.get("title", "No title")
            link = entry.get("link", "")
            summary = entry.get("summary", entry.get("description", ""))[:800]
            published = entry.get("published", "")

            articles.append({
                "title": title,
                "link": link,
                "summary": summary,
                "published": published,
                "source": name,
                "category": category,
                "scraped_at": datetime.utcnow().isoformat(),
            })

        log.info("learning.feed_scraped", feed=name, articles=len(articles))

        for article in articles:
            await self.push_task_async(
                {"type": "store_article", "article": article},
                "learning_scrape",
            )

    # ==================== Article storage (pgvector) ====================

    async def _store_article(self, article: dict) -> None:
        """Store article in AutoMem conversations table with pgvector embedding.

        Deduplicates by content_hash stored in the context JSON column.
        ChromaDB is NOT used — v2 uses pgvector exclusively.
        """
        title = article.get("title", "")
        link = article.get("link", "")
        source = article.get("source", "unknown")
        summary = article.get("summary", "")

        content_hash = _content_hash(title, link)
        article_text = f"{title}\n\n{summary}".strip()

        # Check for duplicate — query conversations for this hash in context
        try:
            async with await self._get_db() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT id FROM conversations
                        WHERE context->>'article_hash' = %s
                        LIMIT 1
                        """,
                        (content_hash,),
                    )
                    row = await cur.fetchone()
                    if row:
                        log.debug("learning.article_exists", hash=content_hash, title=title[:50])
                        return
        except Exception as exc:
            log.warning("learning.dedup_check_failed", error=str(exc))

        # Generate embedding via Ollama
        embedding = await self._get_embedding(article_text)

        # Store in conversations table
        try:
            async with await self._get_db() as conn:
                async with conn.cursor() as cur:
                    context = {
                        "source": source,
                        "url": link,
                        "title": title,
                        "published": article.get("published", ""),
                        "article_hash": content_hash,
                        "category": article.get("category", "general"),
                    }
                    if embedding:
                        await cur.execute(
                            """
                            INSERT INTO conversations
                                (session_id, summary, embedding, context, outcome, created_at)
                            VALUES (%s, %s, %s::vector, %s, %s, %s)
                            """,
                            (
                                f"learning-{datetime.utcnow().strftime('%Y-%m-%d')}",
                                f"[{source}] {title}",
                                json.dumps(embedding),
                                json.dumps(context),
                                "learning",
                                datetime.utcnow(),
                            ),
                        )
                    else:
                        # Store without embedding if Ollama unavailable
                        await cur.execute(
                            """
                            INSERT INTO conversations
                                (session_id, summary, context, outcome, created_at)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (
                                f"learning-{datetime.utcnow().strftime('%Y-%m-%d')}",
                                f"[{source}] {title}",
                                json.dumps(context),
                                "learning",
                                datetime.utcnow(),
                            ),
                        )
                    await conn.commit()

            log.info("learning.article_stored", hash=content_hash, title=title[:60], source=source)

            # Push to pattern_discovery for research workers to pick up
            await self.push_task_to_graph(
                {
                    "type": "pattern_discovery",
                    "article": {
                        "title": title,
                        "summary": summary[:500],
                        "source": source,
                        "url": link,
                        "category": article.get("category", "general"),
                    },
                },
                "pattern_discovery",
            )

            await self.log_activity(
                action="article_stored",
                status="success",
                details={"title": title[:80], "source": source, "hash": content_hash},
            )

        except Exception as exc:
            log.error("learning.article_store_failed", title=title[:60], error=str(exc))
            await self.log_activity(
                action="article_stored",
                status="error",
                details={"title": title[:80], "error": str(exc)},
            )

    # ==================== Pattern extraction ====================

    async def _extract_patterns(self, articles: list) -> None:
        """Basic pattern extraction — groups articles by source/category for analysis."""
        if not articles:
            return

        by_category: dict[str, int] = {}
        for a in articles:
            cat = a.get("category", "general")
            by_category[cat] = by_category.get(cat, 0) + 1

        log.info("learning.patterns_extracted", count=len(articles), by_category=by_category)
        await self.log_activity(
            action="pattern_extraction",
            status="completed",
            details={"article_count": len(articles), "by_category": by_category},
        )

    # ==================== Embedding helper ====================

    async def _get_embedding(self, text: str) -> list[float] | None:
        """Call Ollama nomic-embed-text synchronously in a thread pool."""
        try:
            import httpx
            payload = {"model": "nomic-embed-text", "prompt": text[:2000]}
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.settings.ollama_base_url}/api/embeddings",
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json().get("embedding")
        except Exception as exc:
            log.warning("learning.embedding_failed", error=str(exc))
            return None


if __name__ == "__main__":
    asyncio.run(LearningWorker().run())
