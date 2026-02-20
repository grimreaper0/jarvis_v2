"""FastAPI server — REST interface for jarvis_v2 graphs and workers."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import structlog

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("jarvis_v2.api_startup")
    yield
    log.info("jarvis_v2.api_shutdown")


app = FastAPI(
    title="jarvis_v2 API",
    description="FiestyGoat AI — LangGraph-native autonomous revenue system",
    version="0.1.0",
    lifespan=lifespan,
)


class OpportunityRequest(BaseModel):
    name: str
    estimated_value: float
    platform: str
    description: str = ""


class ContentRequest(BaseModel):
    topic: str
    platform: str
    hashtags: list[str] = []
    media_url: str = ""


class TradingSignalRequest(BaseModel):
    symbol: str
    side: str
    price: float
    qty: int
    strategy: str = "vwap_mean_reversion"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "jarvis_v2", "version": "0.1.0"}


@app.post("/revenue/evaluate")
async def evaluate_revenue(request: OpportunityRequest) -> dict:
    from jarvis.graphs.revenue import revenue_graph
    from jarvis.core.state import RevenueState

    initial_state: RevenueState = {
        "messages": [],
        "confidence": 0.0,
        "error": None,
        "opportunity": request.model_dump(),
        "evaluation_score": 0.0,
        "action": "skip",
        "delegated_to": None,
    }
    result = await revenue_graph.ainvoke(initial_state)
    return {"action": result["action"], "score": result["evaluation_score"]}


@app.post("/content/gate")
async def content_gate(request: ContentRequest) -> dict:
    from jarvis.graphs.content import content_graph
    from jarvis.core.state import ContentState

    initial_state: ContentState = {
        "messages": [],
        "confidence": 0.0,
        "error": None,
        "content": request.model_dump(),
        "platform": request.platform,
        "quality_score": 0.0,
        "approved": False,
        "rejection_reason": None,
    }
    result = await content_graph.ainvoke(initial_state)
    return {"approved": result["approved"], "quality_score": result["quality_score"]}


@app.post("/trading/signal")
async def submit_trading_signal(request: TradingSignalRequest) -> dict:
    from jarvis.graphs.trading import trading_graph
    from jarvis.core.state import TradingState

    initial_state: TradingState = {
        "messages": [],
        "confidence": 0.8,
        "error": None,
        "symbol": request.symbol,
        "signal": request.model_dump(),
        "risk_approved": False,
        "order_id": None,
        "position_size": 0.0,
    }
    result = await trading_graph.ainvoke(initial_state)
    return {
        "risk_approved": result["risk_approved"],
        "order_id": result.get("order_id"),
        "error": result.get("error"),
    }


@app.get("/workers/status")
async def worker_status() -> dict:
    import redis as redis_lib
    from config.settings import get_settings
    settings = get_settings()
    queues = ["learning_scrape", "research_request", "trading_signal", "health_check", "notifications"]
    status = {}
    try:
        r = redis_lib.from_url(settings.redis_url, decode_responses=True)
        for q in queues:
            status[q] = r.llen(q)
    except Exception as exc:
        log.warning("api.redis_unavailable", error=str(exc))
        status = {q: -1 for q in queues}
    return {"queues": status}
