"""Revenue opportunity evaluation graph (Tier 2 LangGraph)."""
import structlog
from langgraph.graph import StateGraph, END
from jarvis.core.state import RevenueState

log = structlog.get_logger()


def evaluate_opportunity(state: RevenueState) -> RevenueState:
    """Score the revenue opportunity using the confidence gate."""
    opp = state["opportunity"]
    score = opp.get("estimated_value", 0) / 1000.0
    score = min(max(score, 0.0), 1.0)
    return {**state, "evaluation_score": round(score, 4)}


def decide_action(state: RevenueState) -> RevenueState:
    """Route to execute, delegate, or skip based on evaluation score."""
    score = state["evaluation_score"]
    if score >= 0.9:
        action = "execute"
    elif score >= 0.6:
        action = "delegate"
    else:
        action = "skip"
    log.info("revenue.decision", score=score, action=action)
    return {**state, "action": action}


def route_action(state: RevenueState) -> str:
    return state["action"]


def execute_opportunity(state: RevenueState) -> RevenueState:
    log.info("revenue.executing", opportunity=state["opportunity"].get("name", "unknown"))
    return state


def delegate_opportunity(state: RevenueState) -> RevenueState:
    log.info("revenue.delegating", opportunity=state["opportunity"].get("name", "unknown"))
    return {**state, "delegated_to": "human_review"}


def skip_opportunity(state: RevenueState) -> RevenueState:
    log.info("revenue.skipping", opportunity=state["opportunity"].get("name", "unknown"))
    return state


def build_revenue_graph() -> StateGraph:
    graph = StateGraph(RevenueState)

    graph.add_node("evaluate", evaluate_opportunity)
    graph.add_node("decide", decide_action)
    graph.add_node("execute", execute_opportunity)
    graph.add_node("delegate", delegate_opportunity)
    graph.add_node("skip", skip_opportunity)

    graph.set_entry_point("evaluate")
    graph.add_edge("evaluate", "decide")
    graph.add_conditional_edges("decide", route_action, {
        "execute": "execute",
        "delegate": "delegate",
        "skip": "skip",
    })
    graph.add_edge("execute", END)
    graph.add_edge("delegate", END)
    graph.add_edge("skip", END)

    return graph.compile()


revenue_graph = build_revenue_graph()
