"""FastAPI server — REST interface for jarvis_v2 graphs and workers."""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any
import structlog

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("jarvis_v2.api_startup")
    yield
    log.info("jarvis_v2.api_shutdown")


app = FastAPI(
    title="BILLY API",
    description="FiestyGoat AI — LangGraph-native autonomous revenue system",
    version="0.1.0",
    lifespan=lifespan,
)

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


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


class ConfidenceEvalRequest(BaseModel):
    task_type: str
    description: str = ""
    context: dict[str, Any] = {}
    thread_id: str | None = None


class ConfidenceResumeRequest(BaseModel):
    thread_id: str
    user_choice: str  # "execute" | "delegate" | "skip" | "abort"


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


@app.post("/confidence/evaluate")
async def evaluate_confidence(request: ConfidenceEvalRequest) -> dict:
    """Run 4-layer confidence scoring.

    If confidence < 0.60, graph hits interrupt() and returns:
        {"status": "waiting_for_input", "thread_id": "...", "interrupt": {...}}

    Resume via POST /confidence/resume with same thread_id.
    """
    from jarvis.graphs.confidence import build_confidence_graph
    from jarvis.core.state import ConfidenceState
    from langchain_core.messages import HumanMessage
    import json
    from uuid import uuid4

    graph = build_confidence_graph()
    thread_id = request.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    task_message = HumanMessage(content=json.dumps({
        "task_type": request.task_type,
        "description": request.description,
        "context": request.context,
    }))

    initial_state: ConfidenceState = {
        "messages": [task_message],
        "confidence": 0.0,
        "error": None,
        "base_score": 0.0,
        "validation_score": 0.0,
        "historical_score": 0.0,
        "reflexive_score": 0.0,
        "final_score": 0.0,
        "decision": "clarify",
        "user_decision": None,
    }

    result = await graph.ainvoke(initial_state, config=config)

    # Check if graph was interrupted (confidence < 0.60 → clarify_node called interrupt())
    snapshot = graph.get_state(config)
    if snapshot.next:
        interrupt_val: dict = {}
        for task_info in snapshot.tasks:
            if task_info.interrupts:
                interrupt_val = task_info.interrupts[0].value
                break
        log.info("confidence.api.interrupted", thread_id=thread_id)
        return {
            "status": "waiting_for_input",
            "thread_id": thread_id,
            "interrupt": interrupt_val,
        }

    log.info(
        "confidence.api.complete",
        thread_id=thread_id,
        decision=result.get("decision"),
        final_score=result.get("final_score"),
    )
    return {
        "status": "complete",
        "thread_id": thread_id,
        "decision": result.get("decision"),
        "final_score": result.get("final_score"),
        "base_score": result.get("base_score"),
        "validation_score": result.get("validation_score"),
        "historical_score": result.get("historical_score"),
        "reflexive_score": result.get("reflexive_score"),
        "user_decision": result.get("user_decision"),
    }


@app.post("/confidence/resume")
async def resume_confidence(request: ConfidenceResumeRequest) -> dict:
    """Resume a confidence evaluation that was interrupted.

    Pass the thread_id from /confidence/evaluate and the operator's choice.
    The graph resumes from the clarify_node and routes to END with the chosen decision.
    """
    from jarvis.graphs.confidence import build_confidence_graph
    from langgraph.types import Command

    graph = build_confidence_graph()
    config = {"configurable": {"thread_id": request.thread_id}}

    # Check the thread exists and is actually waiting
    snapshot = graph.get_state(config)
    if not snapshot.next:
        raise HTTPException(
            status_code=404,
            detail=f"No interrupted graph found for thread_id={request.thread_id!r}. "
                   "Either already completed or thread_id is wrong.",
        )

    result = await graph.ainvoke(Command(resume=request.user_choice), config=config)
    log.info(
        "confidence.api.resumed",
        thread_id=request.thread_id,
        user_choice=request.user_choice,
        decision=result.get("decision"),
    )
    return {
        "status": "complete",
        "thread_id": request.thread_id,
        "decision": result.get("decision"),
        "user_decision": result.get("user_decision"),
        "final_score": result.get("final_score"),
    }


@app.get("/api/keys/status")
async def api_keys_status() -> dict:
    """Check which cloud LLM API keys are configured (no values exposed)."""
    import keyring
    keys = {}
    for name in ("groq", "openrouter", "grok", "deepseek"):
        key_name = {"groq": "groq_api_key", "openrouter": "openrouter_api_key",
                    "grok": "xai_api_key", "deepseek": "deepseek_api_key"}[name]
        val = None
        for svc in ("jarvis_v2", "personal-agent-hub", "personal_agent_hub"):
            val = keyring.get_password(svc, key_name)
            if val:
                break
        keys[name] = bool(val)
    return keys


@app.get("/admin", response_class=HTMLResponse)
async def admin_ui() -> HTMLResponse:
    """LangGraph Admin UI — read-only system overview."""
    from jarvis.api.admin import get_admin_html
    return HTMLResponse(content=get_admin_html())


@app.get("/admin/data")
async def admin_data() -> dict:
    """Admin config as JSON — all thresholds, LLM models, guardrail values."""
    from config.settings import get_settings
    from jarvis.api.admin import GRAPHS, WORKERS, AGENTS, GUARDRAILS, LLMS, PRIME_DIRECTIVES, COMMANDMENTS
    settings = get_settings()
    return {
        "settings": {
            "confidence_execute": settings.confidence_execute,
            "confidence_delegate": settings.confidence_delegate,
            "ollama_model_fast": settings.ollama_model_fast,
            "ollama_model_reason": settings.ollama_model_reason,
            "api_port": settings.api_port,
            "app_version": settings.app_version,
        },
        "graphs": [{"name": g["name"], "queue": g["queue"], "file": g["file"]} for g in GRAPHS],
        "workers": [{"name": w["name"], "queue": w["queue"]} for w in WORKERS],
        "agents": [{"name": a["name"], "platform": a["platform"], "queue": a["queue"]} for a in AGENTS],
        "guardrails": GUARDRAILS,
        "llms": [{"name": l["name"], "provider": l["provider"], "cost": l["cost"]} for l in LLMS],
        "prime_directives": [{"number": p["number"], "name": p["name"]} for p in PRIME_DIRECTIVES],
    }
