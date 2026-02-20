"""Unit tests for core state models."""
import pytest
from jarvis.core.state import (
    BaseState,
    RevenueState,
    TradingState,
    ContentState,
    ConfidenceState,
)


def test_revenue_state_fields():
    state: RevenueState = {
        "messages": [],
        "confidence": 0.85,
        "error": None,
        "opportunity": {"name": "instagram_post", "estimated_value": 50.0},
        "evaluation_score": 0.0,
        "action": "skip",
        "delegated_to": None,
    }
    assert state["action"] == "skip"
    assert state["confidence"] == 0.85
    assert state["opportunity"]["name"] == "instagram_post"


def test_trading_state_fields():
    state: TradingState = {
        "messages": [],
        "confidence": 0.75,
        "error": None,
        "symbol": "AAPL",
        "signal": {"side": "buy", "price": 185.0},
        "risk_approved": False,
        "order_id": None,
        "position_size": 0.0,
    }
    assert state["symbol"] == "AAPL"
    assert state["risk_approved"] is False


def test_content_state_fields():
    state: ContentState = {
        "messages": [],
        "confidence": 0.9,
        "error": None,
        "content": {"caption": "test", "hashtags": ["#ai", "#tech"]},
        "platform": "instagram",
        "quality_score": 0.0,
        "approved": False,
        "rejection_reason": None,
    }
    assert state["platform"] == "instagram"
    assert state["approved"] is False


def test_confidence_state_fields():
    state: ConfidenceState = {
        "messages": [],
        "confidence": 0.0,
        "error": None,
        "base_score": 0.8,
        "validation_score": 0.9,
        "historical_score": 0.7,
        "reflexive_score": 0.85,
        "final_score": 0.0,
        "decision": "execute",
    }
    assert state["decision"] == "execute"
    assert state["base_score"] == 0.8
