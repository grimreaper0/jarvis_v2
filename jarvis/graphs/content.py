"""Content quality gate graph (Tier 2 LangGraph)."""
import structlog
from langgraph.graph import StateGraph, END
from jarvis.core.state import ContentState
from jarvis.core.guardrails import Guardrails

log = structlog.get_logger()
guardrails = Guardrails()


def score_content(state: ContentState) -> ContentState:
    """Score content quality deterministically (no LLM needed for speed)."""
    content = state["content"]
    score = 0.0

    if content.get("title") and len(content["title"]) > 10:
        score += 0.2
    if content.get("body") and len(content["body"]) > 100:
        score += 0.3
    if content.get("hashtags") and len(content["hashtags"]) >= 3:
        score += 0.2
    if content.get("media_url"):
        score += 0.3

    return {**state, "quality_score": round(score, 4)}


def gate_quality(state: ContentState) -> ContentState:
    """Apply quality guardrail — approve or reject."""
    result = guardrails.check_content_quality(state["quality_score"])
    if not result.passed:
        return {**state, "approved": False, "rejection_reason": result.reason}
    return {**state, "approved": True}


def route_quality(state: ContentState) -> str:
    return "approve" if state.get("approved") else "reject"


def approve_content(state: ContentState) -> ContentState:
    log.info("content.approved", platform=state["platform"], score=state["quality_score"])
    return state


def reject_content(state: ContentState) -> ContentState:
    log.info("content.rejected", platform=state["platform"], reason=state.get("rejection_reason"))
    return state


def build_content_graph() -> StateGraph:
    graph = StateGraph(ContentState)

    graph.add_node("score_content", score_content)
    graph.add_node("gate_quality", gate_quality)
    graph.add_node("approve_content", approve_content)
    graph.add_node("reject_content", reject_content)

    graph.set_entry_point("score_content")
    graph.add_edge("score_content", "gate_quality")
    graph.add_conditional_edges("gate_quality", route_quality, {
        "approve": "approve_content",
        "reject": "reject_content",
    })
    graph.add_edge("approve_content", END)
    graph.add_edge("reject_content", END)

    return graph.compile()


content_graph = build_content_graph()
