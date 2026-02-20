"""Quick smoke tests for all four Tier 2 LangGraph graphs."""
import asyncio
import json
import sys
from langchain_core.messages import HumanMessage


async def main() -> None:
    errors: list[str] = []

    # ── Revenue graph ──────────────────────────────────────────────────────
    try:
        from jarvis.graphs.revenue import build_revenue_graph
        g = build_revenue_graph()

        result = await g.ainvoke(
            {
                "messages": [],
                "confidence": 0.0,
                "error": None,
                "opportunity": {
                    "task_type": "content_post",
                    "description": "Create an Instagram carousel about the top 5 AI productivity tools",
                    "context": {"api_available": False, "within_limits": True},
                    "source": "smoke_test",
                },
                "evaluation_score": 0.0,
                "action": "skip",
                "delegated_to": None,
            },
            config={"configurable": {"thread_id": "smoke-rev-1"}},
        )
        print(f"[revenue] action={result['action']}  score={result['evaluation_score']}")

        # Invalid (missing fields) → must skip with error
        result2 = await g.ainvoke(
            {
                "messages": [],
                "confidence": 0.0,
                "error": None,
                "opportunity": {"source": "smoke_test"},
                "evaluation_score": 0.0,
                "action": "skip",
                "delegated_to": None,
            },
            config={"configurable": {"thread_id": "smoke-rev-2"}},
        )
        assert result2["action"] == "skip", f"Expected skip, got {result2['action']}"
        assert result2["error"], "Expected error message for invalid opportunity"
        print(f"[revenue/invalid] action={result2['action']}  error={result2['error'][:60]}")

    except Exception as exc:
        errors.append(f"revenue: {exc}")
        print(f"[revenue] ERROR: {exc}", file=sys.stderr)

    # ── Trading graph ──────────────────────────────────────────────────────
    try:
        from jarvis.graphs.trading import build_trading_graph
        tg = build_trading_graph()

        # Valid signal within limits
        result3 = await tg.ainvoke(
            {
                "messages": [],
                "confidence": 0.65,
                "error": None,
                "symbol": "AAPL",
                "signal": {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "price": 50.0,
                    "confidence": 0.65,
                    "strategy": "VWAP",
                    "context": {"consecutive_losses": 0, "daily_loss_usd": 0.0},
                },
                "risk_approved": False,
                "order_id": None,
                "position_size": 0.0,
            },
            config={"configurable": {"thread_id": "smoke-trade-1"}},
        )
        print(
            f"[trading] risk_approved={result3['risk_approved']}  "
            f"order_id={result3['order_id']}  "
            f"position_size={result3['position_size']}"
        )

        # Over-size signal → must reject
        result4 = await tg.ainvoke(
            {
                "messages": [],
                "confidence": 0.9,
                "error": None,
                "symbol": "TSLA",
                "signal": {
                    "symbol": "TSLA",
                    "action": "BUY",
                    "price": 250.0,
                    "confidence": 0.9,
                    "strategy": "VWAP",
                    "context": {},
                },
                "risk_approved": False,
                "order_id": None,
                "position_size": 0.0,
            },
            config={"configurable": {"thread_id": "smoke-trade-2"}},
        )
        assert not result4["risk_approved"], "Over-size trade should be rejected"
        print(
            f"[trading/oversize] risk_approved={result4['risk_approved']}  "
            f"order_id={result4['order_id']}"
        )

        # Circuit breaker (3 consecutive losses)
        result5 = await tg.ainvoke(
            {
                "messages": [],
                "confidence": 0.8,
                "error": None,
                "symbol": "NVDA",
                "signal": {
                    "symbol": "NVDA",
                    "action": "BUY",
                    "price": 40.0,
                    "confidence": 0.8,
                    "strategy": "VWAP",
                    "context": {"consecutive_losses": 3, "daily_loss_usd": 0.0},
                },
                "risk_approved": False,
                "order_id": None,
                "position_size": 0.0,
            },
            config={"configurable": {"thread_id": "smoke-trade-3"}},
        )
        assert not result5["risk_approved"], "Circuit breaker should reject trade"
        print(
            f"[trading/circuit_breaker] risk_approved={result5['risk_approved']}  "
            f"error={result5.get('error', '')[:60]}"
        )

    except Exception as exc:
        errors.append(f"trading: {exc}")
        print(f"[trading] ERROR: {exc}", file=sys.stderr)

    # ── Content graph ──────────────────────────────────────────────────────
    try:
        from jarvis.graphs.content import build_content_graph
        cg = build_content_graph()

        # Well-formed Instagram content → approved
        result6 = await cg.ainvoke(
            {
                "messages": [],
                "confidence": 0.0,
                "error": None,
                "content": {
                    "caption": (
                        "Check out these 5 amazing AI tools that will 10x your productivity! "
                        "Link in bio for the full breakdown."
                    ),
                    "hashtags": ["#ai", "#productivity", "#tools", "#tech", "#automation", "#llm"],
                    "media_url": "https://example.com/carousel.png",
                    "cta": "link in bio",
                },
                "platform": "instagram",
                "quality_score": 0.0,
                "approved": False,
                "rejection_reason": None,
            },
            config={"configurable": {"thread_id": "smoke-content-1"}},
        )
        print(
            f"[content/instagram] approved={result6['approved']}  "
            f"score={result6['quality_score']}"
        )

        # Stub TikTok content → check it runs without crash
        result7 = await cg.ainvoke(
            {
                "messages": [],
                "confidence": 0.0,
                "error": None,
                "content": {
                    "hook": "Wait, this AI can do WHAT?",
                    "caption": "Mind-blowing AI tool you need to see",
                    "trending_sounds": ["original sound - trendy"],
                    "script": "Script content here",
                },
                "platform": "tiktok",
                "quality_score": 0.0,
                "approved": False,
                "rejection_reason": None,
            },
            config={"configurable": {"thread_id": "smoke-content-2"}},
        )
        print(
            f"[content/tiktok] approved={result7['approved']}  "
            f"score={result7['quality_score']}"
        )

        # Bad content → rejected
        result8 = await cg.ainvoke(
            {
                "messages": [],
                "confidence": 0.0,
                "error": None,
                "content": {"caption": "hi", "hashtags": []},
                "platform": "instagram",
                "quality_score": 0.0,
                "approved": False,
                "rejection_reason": None,
            },
            config={"configurable": {"thread_id": "smoke-content-3"}},
        )
        assert not result8["approved"], "Thin content should be rejected"
        print(
            f"[content/instagram_bad] approved={result8['approved']}  "
            f"score={result8['quality_score']}"
        )

    except Exception as exc:
        errors.append(f"content: {exc}")
        print(f"[content] ERROR: {exc}", file=sys.stderr)

    # ── Confidence graph ───────────────────────────────────────────────────
    try:
        from jarvis.graphs.confidence import build_confidence_graph
        cfg = build_confidence_graph()

        task_msg = HumanMessage(
            content=json.dumps(
                {
                    "task_type": "content_post",
                    "description": (
                        "Create Instagram carousel about top AI productivity tools for 2025"
                    ),
                    "context": {
                        "api_available": True,
                        "data_fresh": True,
                        "within_limits": True,
                    },
                }
            )
        )
        result9 = await cfg.ainvoke(
            {
                "messages": [task_msg],
                "confidence": 0.0,
                "error": None,
                "base_score": 0.0,
                "validation_score": 0.0,
                "historical_score": 0.0,
                "reflexive_score": 0.0,
                "final_score": 0.0,
                "decision": "clarify",
            },
            config={"configurable": {"thread_id": "smoke-conf-1"}},
        )
        print(
            f"[confidence] base={result9['base_score']}  "
            f"validation={result9['validation_score']}  "
            f"historical={result9['historical_score']}  "
            f"reflexive={result9['reflexive_score']}  "
            f"final={result9['final_score']}  "
            f"decision={result9['decision']}"
        )

    except Exception as exc:
        errors.append(f"confidence: {exc}")
        print(f"[confidence] ERROR: {exc}", file=sys.stderr)

    # ── Summary ────────────────────────────────────────────────────────────
    if errors:
        print(f"\nFAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nAll smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
