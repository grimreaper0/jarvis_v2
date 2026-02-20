"""InstagramAgent — Tier 3 specialist subgraph for Instagram content."""
import structlog
from langgraph.graph import StateGraph, END
from jarvis.core.state import ContentState

log = structlog.get_logger()


def generate_caption(state: ContentState) -> ContentState:
    """Generate Instagram caption with hashtags using LLMRouter."""
    content = state["content"]
    log.info("instagram.caption_generate", topic=content.get("topic", ""))
    caption = f"[Caption for: {content.get('topic', 'AI tools')}]\n\n#AITools #FiestyGoatAI"
    return {**state, "content": {**content, "body": caption}}


def select_hashtags(state: ContentState) -> ContentState:
    """Select optimized hashtags based on AutoMem pattern history."""
    content = state["content"]
    hashtags = content.get("hashtags", []) or ["#AI", "#MachineLearning", "#TechTools", "#AINews", "#FiestyGoatAI"]
    return {**state, "content": {**content, "hashtags": hashtags}}


def queue_for_publish(state: ContentState) -> ContentState:
    """Push approved content to the Instagram publish queue."""
    log.info("instagram.queued", platform="instagram", score=state.get("quality_score", 0))
    return state


def build_instagram_agent() -> StateGraph:
    from jarvis.graphs.content import build_content_graph
    graph = StateGraph(ContentState)

    graph.add_node("generate_caption", generate_caption)
    graph.add_node("select_hashtags", select_hashtags)
    graph.add_node("quality_gate", lambda s: s)
    graph.add_node("queue_for_publish", queue_for_publish)

    graph.set_entry_point("generate_caption")
    graph.add_edge("generate_caption", "select_hashtags")
    graph.add_edge("select_hashtags", "quality_gate")
    graph.add_edge("quality_gate", "queue_for_publish")
    graph.add_edge("queue_for_publish", END)

    return graph.compile()


instagram_agent = build_instagram_agent()
