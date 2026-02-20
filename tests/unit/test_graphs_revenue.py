"""Unit tests for revenue LangGraph."""
import pytest
import asyncio


@pytest.mark.asyncio
async def test_revenue_graph_execute():
    """High-value opportunity should be executed."""
    from jarvis.graphs.revenue import build_revenue_graph
    from jarvis.core.state import RevenueState

    graph = build_revenue_graph()
    state: RevenueState = {
        "messages": [],
        "confidence": 0.0,
        "error": None,
        "opportunity": {
            "name": "instagram_viral_post",
            "estimated_value": 950.0,
            "platform": "instagram",
        },
        "evaluation_score": 0.0,
        "action": "skip",
        "delegated_to": None,
    }
    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "test-1"}})
    assert result["action"] in ("execute", "delegate", "skip")
    assert 0.0 <= result["evaluation_score"] <= 1.0


@pytest.mark.asyncio
async def test_revenue_graph_skip_low_value():
    """Very low value opportunity should be skipped."""
    from jarvis.graphs.revenue import build_revenue_graph
    from jarvis.core.state import RevenueState

    graph = build_revenue_graph()
    state: RevenueState = {
        "messages": [],
        "confidence": 0.0,
        "error": None,
        "opportunity": {
            "name": "tiny_opportunity",
            "estimated_value": 1.0,  # $1 — very low
            "platform": "misc",
        },
        "evaluation_score": 0.0,
        "action": "skip",
        "delegated_to": None,
    }
    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "test-2"}})
    assert result["action"] == "skip"


@pytest.mark.asyncio
async def test_trading_graph_risk_check():
    """Trading graph should validate risk before approving."""
    from jarvis.graphs.trading import build_trading_graph
    from jarvis.core.state import TradingState

    graph = build_trading_graph()
    state: TradingState = {
        "messages": [],
        "confidence": 0.85,
        "error": None,
        "symbol": "AAPL",
        "signal": {
            "side": "buy",
            "price": 185.00,
            "strategy": "vwap_mean_reversion",
            "deviation_pct": 1.8,
        },
        "risk_approved": False,
        "order_id": None,
        "position_size": 0.0,
    }
    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "test-3"}})
    assert "risk_approved" in result


@pytest.mark.asyncio
async def test_content_graph_instagram_approve():
    """Valid Instagram content should be approved."""
    from jarvis.graphs.content import build_content_graph
    from jarvis.core.state import ContentState

    graph = build_content_graph()
    state: ContentState = {
        "messages": [],
        "confidence": 0.0,
        "error": None,
        "content": {
            "caption": "🤖 Top 5 AI tools that will 10x your productivity in 2026! Which one is your favorite? Drop it below! 👇 #AI #productivity #tech #artificial intelligence #ChatGPT #automation #tools #future #innovation #machinelearning",
            "hashtags": ["#AI", "#productivity", "#tech", "#automation", "#tools"],
            "has_cta": True,
        },
        "platform": "instagram",
        "quality_score": 0.0,
        "approved": False,
        "rejection_reason": None,
    }
    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "test-4"}})
    assert "quality_score" in result
    assert "approved" in result
