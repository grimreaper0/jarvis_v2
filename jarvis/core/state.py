"""TypedDict state models for LangGraph graphs."""
from typing import Annotated, Any
from typing_extensions import TypedDict, NotRequired
from langgraph.graph.message import add_messages


class BaseState(TypedDict):
    """Shared base fields across all graphs."""
    messages: Annotated[list, add_messages]
    confidence: float
    error: str | None


class RevenueState(BaseState):
    """State for the revenue opportunity graph."""
    opportunity: dict[str, Any]
    evaluation_score: float
    action: str  # "execute" | "delegate" | "skip"
    delegated_to: str | None


class TradingState(BaseState):
    """State for the trading signal graph."""
    symbol: str
    signal: dict[str, Any]
    risk_approved: bool
    order_id: str | None
    position_size: float


class ContentState(BaseState):
    """State for the content quality gate graph."""
    content: dict[str, Any]
    platform: str
    quality_score: float
    approved: bool
    rejection_reason: str | None


class ConfidenceState(BaseState):
    """State for the 4-layer confidence evaluation graph."""
    base_score: float
    validation_score: float
    historical_score: float
    reflexive_score: float
    final_score: float
    decision: str  # "execute" | "delegate" | "clarify"
    user_decision: NotRequired[str | None]  # set after interrupt() resume in clarify_node
