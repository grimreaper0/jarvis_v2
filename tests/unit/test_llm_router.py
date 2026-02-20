"""Unit tests for LLMRouter v3 — task-type routing, backend selection, fallback logic."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from jarvis.core.router import LLMRouter, LLMRequest, LLMResponse, LLMBackend, TASK_ROUTING


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def router():
    r = LLMRouter()
    # Clear backend cache so each test controls availability
    r._backends = None
    return r


def _backends(
    ollama_models: set[str] | None = None,
    vllm_models: set[str] | None = None,
    groq: bool = False,
    deepseek: bool = False,
    openrouter: bool = False,
    claude: bool = False,
    grok: bool = False,
) -> dict:
    return {
        "ollama": ollama_models or set(),
        "vllm": vllm_models or set(),
        "groq": groq,
        "deepseek": deepseek,
        "openrouter": openrouter,
        "claude": claude,
        "grok": grok,
    }


# ---------------------------------------------------------------------------
# TASK_ROUTING sanity checks
# ---------------------------------------------------------------------------

def test_all_task_types_have_routing():
    required = {"simple", "reasoning", "coding", "long_ctx", "creative", "trading", "analysis", "content"}
    missing = required - set(TASK_ROUTING.keys())
    assert not missing, f"Missing routing for task types: {missing}"


def test_routing_entries_are_tuples_of_two():
    for task_type, entries in TASK_ROUTING.items():
        assert entries, f"No routing entries for {task_type}"
        for entry in entries:
            assert len(entry) == 2, f"Bad entry {entry} in {task_type} routing"
            backend, model = entry
            assert isinstance(backend, str)
            assert isinstance(model, str)


def test_reasoning_routing_prefers_deepseek():
    entries = TASK_ROUTING["reasoning"]
    backends = [b for b, _ in entries]
    # First local entry should be ollama with deepseek-r1
    assert entries[0][0] == "ollama"
    assert "deepseek" in entries[0][1]


def test_coding_routing_prefers_coder_model():
    entries = TASK_ROUTING["coding"]
    assert entries[0][0] == "ollama"
    assert "coder" in entries[0][1]


def test_trading_routing_uses_reasoning_models():
    entries = TASK_ROUTING["trading"]
    models = [m for _, m in entries]
    # Should include DeepSeek R1 variants
    assert any("deepseek" in m.lower() or "r1" in m.lower() for m in models)


# ---------------------------------------------------------------------------
# _is_available tests
# ---------------------------------------------------------------------------

def test_ollama_available_exact_match(router):
    b = _backends(ollama_models={"qwen2.5:0.5b", "deepseek-r1:7b"})
    assert router._is_available("ollama", "qwen2.5:0.5b", b)
    assert router._is_available("ollama", "deepseek-r1:7b", b)
    assert not router._is_available("ollama", "mistral:7b", b)


def test_ollama_available_prefix_match(router):
    b = _backends(ollama_models={"qwen2.5:0.5b-fp16"})
    assert router._is_available("ollama", "qwen2.5:0.5b", b)


def test_vllm_available_case_insensitive(router):
    b = _backends(vllm_models={"Qwen/Qwen3-30B-A3B", "deepseek-r1-14b"})
    assert router._is_available("vllm", "Qwen/Qwen3-30B-A3B", b)
    assert router._is_available("vllm", "deepseek-r1-14b", b)
    assert not router._is_available("vllm", "mistral-7b", b)


def test_vllm_unavailable_when_no_models(router):
    b = _backends(vllm_models=set())
    assert not router._is_available("vllm", "Qwen/Qwen3-30B-A3B", b)


def test_cloud_backend_requires_key(router):
    b_no_key = _backends(groq=False)
    b_with_key = _backends(groq=True)
    assert not router._is_available("groq", "llama-3.3-70b-versatile", b_no_key)
    assert router._is_available("groq", "llama-3.3-70b-versatile", b_with_key)


# ---------------------------------------------------------------------------
# _select_route tests
# ---------------------------------------------------------------------------

def test_simple_task_picks_qwen_when_available(router):
    b = _backends(ollama_models={"qwen2.5:0.5b"})
    backend, model = router._select_route("simple", False, None, b)
    assert backend == "ollama"
    assert "qwen2.5" in model


def test_reasoning_falls_through_to_groq(router):
    b = _backends(groq=True)  # Ollama has no deepseek-r1
    backend, model = router._select_route("reasoning", True, None, b)
    assert backend == "groq"
    assert "deepseek" in model or "r1" in model


def test_vllm_prioritized_over_cloud_for_analysis(router):
    b = _backends(
        vllm_models={"Qwen/Qwen3-30B-A3B"},
        groq=True,
    )
    backend, model = router._select_route("analysis", False, None, b)
    assert backend == "vllm"


def test_explicit_backend_override_ollama(router):
    b = _backends(ollama_models={"qwen2.5:0.5b", "deepseek-r1:7b"})
    backend, model = router._select_route("simple", True, LLMBackend.OLLAMA, b)
    assert backend == "ollama"
    assert model == router.REASONING_MODEL  # reasoning=True → deepseek


def test_explicit_backend_override_vllm(router):
    b = _backends(vllm_models={"Qwen/Qwen3-30B-A3B"})
    backend, model = router._select_route("analysis", False, LLMBackend.VLLM, b)
    assert backend == "vllm"
    assert "Qwen3" in model


def test_no_backend_raises_runtime_error(router):
    b = _backends()  # nothing available
    with pytest.raises(RuntimeError, match="No LLM backend available"):
        router._select_route("simple", False, None, b)


def test_last_resort_picks_any_ollama(router):
    b = _backends(ollama_models={"jarvis:latest"})  # no matching routing model
    backend, model = router._select_route("simple", False, None, b)
    assert backend == "ollama"
    assert model == "jarvis:latest"


# ---------------------------------------------------------------------------
# LLMRequest / LLMResponse model validation
# ---------------------------------------------------------------------------

def test_llm_request_defaults():
    req = LLMRequest(prompt="Hello")
    assert req.task_type == "simple"
    assert req.reasoning is False
    assert req.backend is None
    assert req.max_tokens == 2048


def test_llm_request_reasoning_flag():
    req = LLMRequest(prompt="Analyze this", reasoning=True, task_type="trading")
    assert req.reasoning is True
    assert req.task_type == "trading"


def test_llm_response_defaults():
    resp = LLMResponse(content="Hello", backend=LLMBackend.OLLAMA, model="qwen2.5:0.5b")
    assert resp.latency_ms == 0.0
    assert resp.reasoning is None
    assert resp.tokens_used == 0


# ---------------------------------------------------------------------------
# Backend enum coverage
# ---------------------------------------------------------------------------

def test_all_backends_in_enum():
    backends = {b.value for b in LLMBackend}
    required = {"ollama", "vllm", "groq", "deepseek", "openrouter", "claude", "grok"}
    assert required.issubset(backends)


# ---------------------------------------------------------------------------
# Integration: complete() routing (mocked backends)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_routes_to_ollama(router):
    router._backends = _backends(ollama_models={"qwen2.5:0.5b"})
    router._backends_checked_at = 1e12  # prevent re-detection

    mock_resp = LLMResponse(content="test", backend=LLMBackend.OLLAMA, model="qwen2.5:0.5b")
    router._ollama_complete = AsyncMock(return_value=mock_resp)

    resp = await router.complete(LLMRequest(prompt="Hello", task_type="simple"))
    assert resp.backend == LLMBackend.OLLAMA
    router._ollama_complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_complete_routes_to_groq_when_ollama_empty(router):
    router._backends = _backends(groq=True)
    router._backends_checked_at = 1e12

    mock_resp = LLMResponse(content="test", backend=LLMBackend.GROQ, model="llama-3.3-70b-versatile")
    router._openai_compat_complete = AsyncMock(return_value=mock_resp)

    resp = await router.complete(LLMRequest(prompt="Hello", task_type="simple"))
    assert resp.backend == LLMBackend.GROQ


@pytest.mark.asyncio
async def test_complete_reasoning_sets_reasoning_fields(router):
    router._backends = _backends(ollama_models={"deepseek-r1:7b"})
    router._backends_checked_at = 1e12

    raw = LLMResponse(
        content="<think>My reasoning here</think>\nMy conclusion",
        backend=LLMBackend.OLLAMA,
        model="deepseek-r1:7b",
    )
    router._ollama_complete = AsyncMock(return_value=raw)

    resp = await router.complete(LLMRequest(prompt="Reason about X", task_type="reasoning", reasoning=True))
    assert resp.reasoning == "My reasoning here"
    assert resp.conclusion == "My conclusion"


@pytest.mark.asyncio
async def test_generate_returns_string(router):
    router._backends = _backends(ollama_models={"qwen2.5:0.5b"})
    router._backends_checked_at = 1e12

    mock_resp = LLMResponse(content="Generated text", backend=LLMBackend.OLLAMA, model="qwen2.5:0.5b")
    router._ollama_complete = AsyncMock(return_value=mock_resp)

    result = await router.generate("Hello")
    assert result == "Generated text"
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_reason_returns_dict_with_keys(router):
    router._backends = _backends(ollama_models={"deepseek-r1:7b"})
    router._backends_checked_at = 1e12

    raw = LLMResponse(
        content="<think>Step 1\nStep 2</think>\nFinal answer",
        backend=LLMBackend.OLLAMA,
        model="deepseek-r1:7b",
    )
    router._ollama_complete = AsyncMock(return_value=raw)

    result = await router.reason("Analyze this")
    assert "reasoning" in result
    assert "conclusion" in result
    assert "model" in result
    assert "backend" in result
    assert result["reasoning_available"] is True
    assert result["reasoning_steps"] >= 1
