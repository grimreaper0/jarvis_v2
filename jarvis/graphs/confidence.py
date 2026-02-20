"""Confidence evaluation graph (Tier 2 LangGraph)."""
import structlog
from langgraph.graph import StateGraph, END
from jarvis.core.state import ConfidenceState
from jarvis.core.confidence import ConfidenceGate

log = structlog.get_logger()
gate = ConfidenceGate()


def compute_base(state: ConfidenceState) -> ConfidenceState:
    """Layer 1: base confidence from LLM self-assessment (passed in state)."""
    return {**state, "base_score": state.get("base_score", 0.5)}


def compute_validation(state: ConfidenceState) -> ConfidenceState:
    """Layer 2: validation score from data quality checks."""
    return {**state, "validation_score": state.get("validation_score", 0.5)}


def compute_historical(state: ConfidenceState) -> ConfidenceState:
    """Layer 3: historical score from AutoMem pattern matching."""
    return {**state, "historical_score": state.get("historical_score", 0.5)}


def compute_reflexive(state: ConfidenceState) -> ConfidenceState:
    """Layer 4: reflexive score from second LLM pass."""
    return {**state, "reflexive_score": state.get("reflexive_score", 0.5)}


def finalize(state: ConfidenceState) -> ConfidenceState:
    """Compute weighted final score and determine decision."""
    result = gate.evaluate(
        base=state["base_score"],
        validation=state["validation_score"],
        historical=state["historical_score"],
        reflexive=state["reflexive_score"],
    )
    return {
        **state,
        "final_score": result.final_score,
        "decision": result.decision,
    }


def build_confidence_graph() -> StateGraph:
    graph = StateGraph(ConfidenceState)

    graph.add_node("base", compute_base)
    graph.add_node("validation", compute_validation)
    graph.add_node("historical", compute_historical)
    graph.add_node("reflexive", compute_reflexive)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("base")
    graph.add_edge("base", "validation")
    graph.add_edge("validation", "historical")
    graph.add_edge("historical", "reflexive")
    graph.add_edge("reflexive", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


confidence_graph = build_confidence_graph()
