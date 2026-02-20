"""ResearchWorker — 24/7 AI breakthrough detection from 19 deep sources."""
import asyncio
import structlog
from jarvis.workers.base import ContinuousWorker

log = structlog.get_logger()

DARK_HOLE_SOURCES: list[dict] = [
    {"name": "arxiv_cs_ai", "type": "arxiv", "category": "cs.AI"},
    {"name": "arxiv_cs_lg", "type": "arxiv", "category": "cs.LG"},
    {"name": "arxiv_cs_ma", "type": "arxiv", "category": "cs.MA"},
    {"name": "reddit_localllama", "type": "reddit", "subreddit": "LocalLLaMA"},
    {"name": "reddit_ml", "type": "reddit", "subreddit": "MachineLearning"},
    {"name": "reddit_ai", "type": "reddit", "subreddit": "ArtificialIntelligence"},
    {"name": "reddit_singularity", "type": "reddit", "subreddit": "singularity"},
    {"name": "openai_blog", "type": "rss", "url": "https://openai.com/blog/rss.xml"},
    {"name": "deepmind_blog", "type": "rss", "url": "https://deepmind.google/blog/rss.xml"},
    {"name": "anthropic_news", "type": "rss", "url": "https://www.anthropic.com/news/rss.xml"},
    {"name": "meta_ai_blog", "type": "rss", "url": "https://ai.meta.com/blog/rss/"},
]

QUEUES = ["research_request", "viral_analysis", "competitive_intelligence", "pattern_discovery", "discovery_scrape"]


class ResearchWorker(ContinuousWorker):
    """Continuously monitors AI research sources for breakthrough discoveries.

    Queue: research_request, viral_analysis, competitive_intelligence,
           pattern_discovery, discovery_scrape
    Idle:  scrapes all 19 sources every hour
    """

    queue_name = "research_request"
    worker_name = "research_worker"

    async def process_task(self, task: dict) -> None:
        task_type = task.get("type")
        if task_type == "deep_dive":
            await self._deep_dive(task["discovery"])
        elif task_type == "viral_analysis":
            await self._viral_analysis(task["item"])
        elif task_type == "competitive_intelligence":
            await self._competitive_intel(task["source"])
        else:
            log.warning("research.unknown_task", task_type=task_type)

    async def idle_loop(self) -> None:
        log.info("research.idle_scrape", sources=len(DARK_HOLE_SOURCES))
        for source in DARK_HOLE_SOURCES:
            try:
                await self._scrape_source(source)
                await asyncio.sleep(3)
            except Exception as exc:
                log.warning("research.source_error", source=source["name"], error=str(exc))

    async def _scrape_source(self, source: dict) -> None:
        source_type = source.get("type")
        if source_type == "rss":
            await self._scrape_rss(source)
        elif source_type == "arxiv":
            await self._scrape_arxiv(source)
        elif source_type == "reddit":
            await self._scrape_reddit(source)

    async def _scrape_rss(self, source: dict) -> None:
        import httpx
        import feedparser
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(source["url"])
            resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        log.info("research.rss_scraped", source=source["name"], entries=len(feed.entries))

    async def _scrape_arxiv(self, source: dict) -> None:
        category = source["category"]
        url = f"https://export.arxiv.org/rss/{category}"
        import httpx
        import feedparser
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        log.info("research.arxiv_scraped", category=category, entries=len(feed.entries))

    async def _scrape_reddit(self, source: dict) -> None:
        subreddit = source["subreddit"]
        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=25"
        import httpx
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "jarvis_v2/0.1"}) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        data = resp.json()
        posts = data.get("data", {}).get("children", [])
        log.info("research.reddit_scraped", subreddit=subreddit, posts=len(posts))

    async def _deep_dive(self, discovery: dict) -> None:
        log.info("research.deep_dive", title=discovery.get("title", "")[:60])

    async def _viral_analysis(self, item: dict) -> None:
        log.info("research.viral_analysis", item=str(item)[:60])

    async def _competitive_intel(self, source: dict) -> None:
        log.info("research.competitive_intel", source=str(source)[:60])


if __name__ == "__main__":
    worker = ResearchWorker()
    asyncio.run(worker.run())
