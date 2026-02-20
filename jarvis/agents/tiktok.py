"""TikTokAgent — Short-form viral content + Creator Fund/Shop/affiliates."""
import structlog
from langgraph.graph import StateGraph, END
from jarvis.core.state import ContentState

log = structlog.get_logger()

VIRAL_SCORE_THRESHOLD = 0.65


def generate_hook(state: ContentState) -> ContentState:
    content = state["content"]
    topic = content.get("topic", "AI")
    hook = f"POV: You just found out {topic} can do THIS... 🤯"
    return {**state, "content": {**content, "hook": hook}}


def score_viral_potential(state: ContentState) -> ContentState:
    content = state["content"]
    score = 0.0
    if content.get("trending_sound"):
        score += 0.3
    if content.get("hook") and "?" in content["hook"] or "🤯" in content.get("hook", ""):
        score += 0.25
    if content.get("duration_sec", 0) <= 30:
        score += 0.25
    if content.get("fyp_tags"):
        score += 0.2
    return {**state, "content": {**content, "viral_score": round(score, 4)}}


def add_fyp_tags(state: ContentState) -> ContentState:
    content = state["content"]
    tags = ["#FYP", "#ForYou", "#AITools", "#Tech", "#FiestyGoatAI"]
    return {**state, "content": {**content, "fyp_tags": tags, "hashtags": tags}}


def queue_for_publish(state: ContentState) -> ContentState:
    viral_score = state["content"].get("viral_score", 0)
    if viral_score < VIRAL_SCORE_THRESHOLD:
        log.info("tiktok.below_viral_threshold", score=viral_score)
    else:
        log.info("tiktok.queued", platform="tiktok", viral_score=viral_score)
    return state


def build_tiktok_agent() -> StateGraph:
    graph = StateGraph(ContentState)

    graph.add_node("generate_hook", generate_hook)
    graph.add_node("add_fyp_tags", add_fyp_tags)
    graph.add_node("score_viral_potential", score_viral_potential)
    graph.add_node("queue_for_publish", queue_for_publish)

    graph.set_entry_point("generate_hook")
    graph.add_edge("generate_hook", "add_fyp_tags")
    graph.add_edge("add_fyp_tags", "score_viral_potential")
    graph.add_edge("score_viral_potential", "queue_for_publish")
    graph.add_edge("queue_for_publish", END)

    return graph.compile()


tiktok_agent = build_tiktok_agent()
