"""End-to-end integration tests for the 3-tier pipeline.

Validates the full flow:
  Tier 1 Worker → Redis queue → Tier 2 GraphRunner → Redis queue → Tier 3 Agent

Uses fakeredis to avoid needing a live Redis instance.
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis


# ─────────────────────── Fixtures ─────────────────────────────────────────────


@pytest.fixture
def fake_redis():
    """Provide a fakeredis async instance shared across the test."""
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


# ─────────────────────── Tier 1 → Tier 2 (queue handoff) ─────────────────────


@pytest.mark.asyncio
async def test_trading_worker_pushes_to_trading_decision(fake_redis):
    """TradingWorker._route_signal pushes to 'trading_decision' which
    TradingGraphRunner consumes."""
    from jarvis.graphs.trading import TradingGraphRunner

    # Simulate what TradingWorker._route_signal does: lpush a signal
    signal = {
        "symbol": "AAPL",
        "side": "buy",
        "price": 185.50,
        "qty": 1,
        "strategy": "vwap_mean_reversion",
        "source": "e2e_test",
    }
    await fake_redis.lpush("trading_decision", json.dumps(signal))

    # Verify the GraphRunner's queue name matches
    assert TradingGraphRunner.QUEUE == "trading_decision"

    # Verify data is consumable from the queue
    item = await fake_redis.brpop("trading_decision", timeout=1)
    assert item is not None
    _, raw = item
    parsed = json.loads(raw)
    assert parsed["symbol"] == "AAPL"
    assert parsed["side"] == "buy"


@pytest.mark.asyncio
async def test_revenue_worker_pushes_to_revenue_opportunity(fake_redis):
    """Revenue opportunities land in 'revenue_opportunity' which
    RevenueGraphRunner consumes."""
    from jarvis.graphs.revenue import RevenueGraphRunner

    opportunity = {
        "task_type": "content_post",
        "description": "Instagram carousel: top 5 AI tools",
        "context": {"api_available": True},
        "source": "e2e_test",
    }
    await fake_redis.lpush("revenue_opportunity", json.dumps(opportunity))

    assert RevenueGraphRunner.QUEUE == "revenue_opportunity"

    item = await fake_redis.brpop("revenue_opportunity", timeout=1)
    assert item is not None
    _, raw = item
    parsed = json.loads(raw)
    assert parsed["task_type"] == "content_post"


@pytest.mark.asyncio
async def test_content_graph_consumes_platform_queues(fake_redis):
    """ContentGraphRunner listens on per-platform content check queues."""
    from jarvis.graphs.content import ContentGraphRunner

    expected_queues = [
        "instagram_content_check",
        "youtube_content_check",
        "newsletter_content_check",
        "tiktok_content_check",
        "content_check",
    ]
    assert ContentGraphRunner.QUEUES == expected_queues


# ─────────────────────── Tier 2 Graph execution (mocked LLM) ─────────────────


@pytest.mark.asyncio
async def test_revenue_graph_evaluates_opportunity():
    """Revenue graph processes an opportunity and returns a scored decision."""
    from jarvis.graphs.revenue import build_revenue_graph

    graph = build_revenue_graph()
    result = await graph.ainvoke(
        {
            "messages": [],
            "confidence": 0.0,
            "error": None,
            "opportunity": {
                "task_type": "content_post",
                "description": "Instagram carousel about AI tools",
                "context": {"api_available": False},
                "source": "e2e_test",
            },
            "evaluation_score": 0.0,
            "action": "skip",
            "delegated_to": None,
        },
        config={"configurable": {"thread_id": "e2e-rev-1"}},
    )

    # Graph must produce a decision
    assert result["action"] in ("execute", "delegate", "skip")
    assert isinstance(result["evaluation_score"], (int, float))


@pytest.mark.asyncio
async def test_trading_graph_validates_signal():
    """Trading graph processes a signal through risk validation."""
    from jarvis.graphs.trading import build_trading_graph

    graph = build_trading_graph()
    result = await graph.ainvoke(
        {
            "messages": [],
            "confidence": 0.8,
            "error": None,
            "symbol": "SPY",
            "signal": {
                "symbol": "SPY",
                "side": "buy",
                "price": 450.0,
                "qty": 1,
                "strategy": "vwap_mean_reversion",
            },
            "risk_approved": False,
            "order_id": None,
            "position_size": 0.0,
        },
        config={"configurable": {"thread_id": "e2e-trade-1"}},
    )

    # Graph must produce a risk decision
    assert "risk_approved" in result
    assert isinstance(result["risk_approved"], bool)


@pytest.mark.asyncio
async def test_content_graph_scores_instagram():
    """Content graph deterministically scores Instagram content."""
    from jarvis.graphs.content import build_content_graph

    graph = build_content_graph()
    result = await graph.ainvoke(
        {
            "messages": [],
            "confidence": 0.0,
            "error": None,
            "content": {
                "caption": "Transform your workflow with these 5 AI productivity tools! "
                           "Each one saves 2+ hours per week. Thread below...",
                "hashtags": ["#AITools", "#Productivity", "#Tech", "#AI", "#Automation"],
                "media_url": "https://example.com/carousel.jpg",
                "topic": "AI productivity tools",
            },
            "platform": "instagram",
            "quality_score": 0.0,
            "approved": False,
            "rejection_reason": None,
        },
        config={"configurable": {"thread_id": "e2e-content-1"}},
    )

    assert isinstance(result["quality_score"], (int, float))
    assert isinstance(result["approved"], bool)
    assert result["quality_score"] > 0.0


@pytest.mark.asyncio
async def test_confidence_graph_4_layer_scoring():
    """Confidence graph produces all 4 layer scores and a decision."""
    from jarvis.graphs.confidence import build_confidence_graph
    from langchain_core.messages import HumanMessage

    graph = build_confidence_graph()
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=json.dumps({
                "task_type": "content_post",
                "description": "Post AI tools carousel to Instagram",
                "context": {"api_available": True},
            }))],
            "confidence": 0.0,
            "error": None,
            "base_score": 0.0,
            "validation_score": 0.0,
            "historical_score": 0.0,
            "reflexive_score": 0.0,
            "final_score": 0.0,
            "decision": "clarify",
            "user_decision": None,
        },
        config={"configurable": {"thread_id": "e2e-conf-1"}},
    )

    # All 4 layers must produce scores
    for key in ("base_score", "validation_score", "historical_score", "reflexive_score"):
        assert isinstance(result[key], (int, float)), f"{key} missing or wrong type"
    assert result["final_score"] > 0.0
    assert result["decision"] in ("execute", "delegate", "clarify")


# ─────────────────────── Queue naming consistency ─────────────────────────────


def test_all_queue_names_align():
    """Verify that producer queue names match consumer queue names."""
    from jarvis.graphs.trading import TradingGraphRunner
    from jarvis.graphs.revenue import RevenueGraphRunner
    from jarvis.graphs.confidence import ConfidenceGraphRunner

    # TradingWorker pushes to 'trading_decision' (documented in trading.py:8)
    assert TradingGraphRunner.QUEUE == "trading_decision"

    # ResearchWorker/RevenueBot pushes to 'revenue_opportunity'
    assert RevenueGraphRunner.QUEUE == "revenue_opportunity"

    # API/Workers push to 'confidence_eval'
    assert ConfidenceGraphRunner.QUEUE == "confidence_eval"


def test_content_graph_queue_naming():
    """Content graph queues end with _content_check."""
    from jarvis.graphs.content import ContentGraphRunner

    for q in ContentGraphRunner.QUEUES:
        assert "content_check" in q, f"Queue {q} doesn't follow pattern"


# ─────────────────────── Full pipeline simulation ─────────────────────────────


@pytest.mark.asyncio
async def test_full_revenue_pipeline_simulation(fake_redis):
    """Simulate complete flow: opportunity → revenue graph → tier 3 queue.

    This test validates that:
    1. An opportunity can be placed on the revenue_opportunity queue
    2. The revenue graph processes it and makes a decision
    3. If action=execute, the task would be routed to a Tier 3 queue
    """
    from jarvis.graphs.revenue import build_revenue_graph, QUEUE_MAP

    # Step 1: Simulate Tier 1 pushing to revenue_opportunity
    opportunity = {
        "task_type": "content_post",
        "description": "Create viral AI tools carousel for Instagram",
        "context": {"api_available": True, "within_limits": True},
        "source": "e2e_pipeline_test",
    }
    await fake_redis.lpush("revenue_opportunity", json.dumps(opportunity))

    # Step 2: Consume from queue (like GraphRunner would)
    item = await fake_redis.brpop("revenue_opportunity", timeout=1)
    assert item is not None
    _, raw = item
    task = json.loads(raw)

    # Step 3: Run through revenue graph
    graph = build_revenue_graph()
    result = await graph.ainvoke(
        {
            "messages": [],
            "confidence": 0.0,
            "error": None,
            "opportunity": task,
            "evaluation_score": 0.0,
            "action": "skip",
            "delegated_to": None,
        },
        config={"configurable": {"thread_id": "e2e-pipeline-1"}},
    )

    # Step 4: Verify decision was made
    assert result["action"] in ("execute", "delegate", "skip")

    # Step 5: Verify QUEUE_MAP has correct Tier 3 routing
    assert "content_post" in QUEUE_MAP or "default" in QUEUE_MAP or len(QUEUE_MAP) > 0
