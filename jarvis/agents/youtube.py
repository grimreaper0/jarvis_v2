"""YouTubeAgent — Tier 3 specialist subgraph for YouTube content."""
import structlog
from langgraph.graph import StateGraph, END
from jarvis.core.state import ContentState

log = structlog.get_logger()


def generate_script_outline(state: ContentState) -> ContentState:
    content = state["content"]
    log.info("youtube.script_generate", topic=content.get("topic", ""))
    outline = f"[Script outline for: {content.get('topic', 'AI tools')}]\n\nIntro → Demo → Value → CTA"
    return {**state, "content": {**content, "body": outline}}


def optimize_seo(state: ContentState) -> ContentState:
    content = state["content"]
    seo_title = f"{content.get('topic', 'AI Tool')} — Complete Guide 2026"
    tags = ["AI tools", "machine learning", "automation", "FiestyGoatAI"]
    return {**state, "content": {**content, "title": seo_title, "tags": tags}}


def generate_thumbnail_concept(state: ContentState) -> ContentState:
    content = state["content"]
    thumbnail = f"Bold text: '{content.get('topic', 'AI')}' | Red background | Face of shock"
    return {**state, "content": {**content, "thumbnail_concept": thumbnail}}


def queue_for_publish(state: ContentState) -> ContentState:
    log.info("youtube.queued", platform="youtube", score=state.get("quality_score", 0))
    return state


def build_youtube_agent() -> StateGraph:
    graph = StateGraph(ContentState)

    graph.add_node("generate_script_outline", generate_script_outline)
    graph.add_node("optimize_seo", optimize_seo)
    graph.add_node("generate_thumbnail_concept", generate_thumbnail_concept)
    graph.add_node("queue_for_publish", queue_for_publish)

    graph.set_entry_point("generate_script_outline")
    graph.add_edge("generate_script_outline", "optimize_seo")
    graph.add_edge("optimize_seo", "generate_thumbnail_concept")
    graph.add_edge("generate_thumbnail_concept", "queue_for_publish")
    graph.add_edge("queue_for_publish", END)

    return graph.compile()


youtube_agent = build_youtube_agent()
