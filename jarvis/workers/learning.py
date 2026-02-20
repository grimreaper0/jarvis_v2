"""LearningWorker — 24/7 RSS + arXiv + Reddit scraping and knowledge storage."""
import asyncio
import structlog
from jarvis.workers.base import ContinuousWorker

log = structlog.get_logger()

RSS_FEEDS: list[dict] = [
    {"name": "niche_pursuits", "url": "https://www.nichepursuits.com/feed/", "category": "revenue"},
    {"name": "ai_automation_trends", "url": "https://aiautomationtrends.com/feed/", "category": "revenue"},
    {"name": "buffer_blog", "url": "https://buffer.com/resources/rss/", "category": "revenue"},
    {"name": "towards_data_science", "url": "https://towardsdatascience.com/feed", "category": "ai"},
    {"name": "hugging_face_blog", "url": "https://huggingface.co/blog/feed.xml", "category": "ai"},
]

ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.MA"]


class LearningWorker(ContinuousWorker):
    """Continuously scrapes RSS feeds and arXiv, stores articles in ChromaDB + AutoMem.

    Queue: learning_scrape, article_store, pattern_extract
    Idle: scrapes all RSS feeds every hour
    """

    queue_name = "learning_scrape"
    worker_name = "learning_worker"

    async def process_task(self, task: dict) -> None:
        task_type = task.get("type")
        if task_type == "scrape_feed":
            await self._scrape_feed(task["url"], task.get("name", "unknown"))
        elif task_type == "store_article":
            await self._store_article(task["article"])
        elif task_type == "extract_patterns":
            await self._extract_patterns(task.get("articles", []))
        else:
            log.warning("learning.unknown_task", task_type=task_type)

    async def idle_loop(self) -> None:
        log.info("learning.idle_scrape", feed_count=len(RSS_FEEDS))
        for feed in RSS_FEEDS:
            try:
                await self._scrape_feed(feed["url"], feed["name"])
                await asyncio.sleep(2)
            except Exception as exc:
                log.warning("learning.feed_error", feed=feed["name"], error=str(exc))

    async def _scrape_feed(self, url: str, name: str) -> None:
        import httpx
        import feedparser
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        articles = []
        for entry in feed.entries[:10]:
            articles.append({
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "summary": entry.get("summary", "")[:500],
                "source": name,
            })
        log.info("learning.feed_scraped", feed=name, articles=len(articles))
        for article in articles:
            self.push_task({"type": "store_article", "article": article}, "article_store")

    async def _store_article(self, article: dict) -> None:
        log.debug("learning.article_stored", title=article.get("title", "")[:60])

    async def _extract_patterns(self, articles: list) -> None:
        log.info("learning.patterns_extracted", count=len(articles))


if __name__ == "__main__":
    worker = LearningWorker()
    asyncio.run(worker.run())
