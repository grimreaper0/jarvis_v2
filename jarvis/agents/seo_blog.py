"""SEOBlogAgent — Long-form AI tool content (1,500-2,500 words), evergreen Google traffic."""
import structlog
from langgraph.graph import StateGraph, END
from jarvis.core.state import ContentState

log = structlog.get_logger()

TARGET_WORD_MIN = 1500
TARGET_WORD_MAX = 2500


def research_keywords(state: ContentState) -> ContentState:
    content = state["content"]
    topic = content.get("topic", "AI tools")
    keywords = [topic, f"best {topic} 2026", f"{topic} review", f"how to use {topic}"]
    return {**state, "content": {**content, "keywords": keywords, "primary_keyword": keywords[0]}}


def write_article(state: ContentState) -> ContentState:
    content = state["content"]
    pk = content.get("primary_keyword", "AI tools")
    article = (
        f"# Best {pk.title()} in 2026: Complete Guide\n\n"
        f"## Introduction\n[1,500-2,500 word article about {pk}...]\n\n"
        f"## What Is {pk.title()}?\n\n"
        f"## Top Features\n\n"
        f"## How to Get Started\n\n"
        f"## Pricing\n\n"
        f"## Pros and Cons\n\n"
        f"## Conclusion\n\n"
        f"[Affiliate CTA]\n"
    )
    return {**state, "content": {**content, "body": article}}


def add_schema_markup(state: ContentState) -> ContentState:
    schema = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": state["content"].get("title", ""),
        "author": {"@type": "Organization", "name": "FiestyGoat AI LLC"},
    }
    return {**state, "content": {**state["content"], "schema": schema}}


def optimize_on_page_seo(state: ContentState) -> ContentState:
    content = state["content"]
    pk = content.get("primary_keyword", "")
    meta_description = f"Discover the best {pk} in 2026. Expert review, pricing, and step-by-step guide."[:160]
    return {**state, "content": {**content, "meta_description": meta_description}}


def queue_for_publish(state: ContentState) -> ContentState:
    log.info("seo_blog.queued", platform="wordpress", score=state.get("quality_score", 0))
    return state


def build_seo_blog_agent() -> StateGraph:
    graph = StateGraph(ContentState)

    graph.add_node("research_keywords", research_keywords)
    graph.add_node("write_article", write_article)
    graph.add_node("add_schema_markup", add_schema_markup)
    graph.add_node("optimize_on_page_seo", optimize_on_page_seo)
    graph.add_node("queue_for_publish", queue_for_publish)

    graph.set_entry_point("research_keywords")
    graph.add_edge("research_keywords", "write_article")
    graph.add_edge("write_article", "add_schema_markup")
    graph.add_edge("add_schema_markup", "optimize_on_page_seo")
    graph.add_edge("optimize_on_page_seo", "queue_for_publish")
    graph.add_edge("queue_for_publish", END)

    return graph.compile()


seo_blog_agent = build_seo_blog_agent()
