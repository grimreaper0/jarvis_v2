"""Unit tests for LLMRouter v4 — vLLM-first, no Ollama, task-type routing for BILLY."""
import pytest
from unittest.mock import AsyncMock, patch
from jarvis.core.router import LLMRouter, LLMRequest, LLMResponse, LLMBackend, TASK_ROUTING
from jarvis.core.mandate import OPERATING_MANDATE, MANDATE_SYSTEM_PROMPT


@pytest.fixture
def router():
    r = LLMRouter()
    r._backends = None
    return r


def _backends(vllm_local=None, vllm=None, groq=False, deepseek=False,
              openrouter=False, grok=False, claude=False):
    return {
        "vllm_local": vllm_local or set(),
        "vllm": vllm or set(),
        "groq": groq, "deepseek": deepseek,
        "openrouter": openrouter, "grok": grok, "claude": claude,
    }


# ── TASK_ROUTING sanity ─────────────────────────────────────────────────────

def test_all_task_types_have_routing():
    required = {"simple", "reasoning", "coding", "long_ctx", "creative", "trading", "analysis", "content"}
    assert not (required - set(TASK_ROUTING.keys()))


def test_routing_entries_are_tuples_of_two():
    for task_type, entries in TASK_ROUTING.items():
        assert entries
        for backend, model in entries:
            assert isinstance(backend, str) and isinstance(model, str)


def test_no_ollama_in_routing():
    for task_type, entries in TASK_ROUTING.items():
        for backend, model in entries:
            assert backend != "ollama", f"Ollama found in {task_type}: {backend}/{model}"


def test_vllm_local_is_first_for_local_tasks():
    for task in ["simple", "reasoning", "coding", "trading", "analysis", "content"]:
        assert TASK_ROUTING[task][0][0] == "vllm_local", f"Expected vllm_local first for {task}"


def test_reasoning_uses_deepseek_r1():
    models = [m for _, m in TASK_ROUTING["reasoning"]]
    assert any("deepseek" in m.lower() or "r1" in m.lower() for m in models)


def test_long_ctx_has_qwen35_via_openrouter():
    entries = TASK_ROUTING["long_ctx"]
    assert any(b == "openrouter" and "qwen3.5" in m.lower() for b, m in entries)


# ── LLMBackend enum ─────────────────────────────────────────────────────────

def test_no_ollama_backend_in_enum():
    assert "ollama" not in {b.value for b in LLMBackend}


def test_required_backends_present():
    backends = {b.value for b in LLMBackend}
    assert {"vllm_local", "vllm", "groq", "deepseek", "openrouter", "grok", "claude"}.issubset(backends)


# ── _is_available ───────────────────────────────────────────────────────────

def test_vllm_local_exact_match(router):
    b = _backends(vllm_local={"Qwen/Qwen3-4B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"})
    assert router._is_available("vllm_local", "Qwen/Qwen3-4B", b)
    assert not router._is_available("vllm_local", "mistralai/Mistral-7B", b)


def test_vllm_local_substring_match(router):
    b = _backends(vllm_local={"Qwen/Qwen3-4B-Instruct"})
    assert router._is_available("vllm_local", "Qwen/Qwen3-4B", b)


def test_vllm_empty_returns_false(router):
    b = _backends(vllm_local=set())
    assert not router._is_available("vllm_local", "Qwen/Qwen3-4B", b)


def test_cloud_backend_requires_key(router):
    assert not router._is_available("groq", "model", _backends(groq=False))
    assert router._is_available("groq", "model", _backends(groq=True))


# ── _select_route ───────────────────────────────────────────────────────────

def test_simple_picks_vllm_local(router):
    b = _backends(vllm_local={"Qwen/Qwen2.5-0.5B-Instruct"})
    backend, model = router._select_route("simple", False, None, b)
    assert backend == "vllm_local"


def test_reasoning_falls_to_groq_when_no_vllm(router):
    b = _backends(groq=True)
    backend, model = router._select_route("reasoning", True, None, b)
    assert backend == "groq"
    assert "r1" in model.lower() or "deepseek" in model.lower()


def test_vllm_remote_beats_groq_for_analysis(router):
    b = _backends(vllm={"Qwen/Qwen3-30B-A3B"}, groq=True)
    backend, model = router._select_route("analysis", False, None, b)
    assert backend == "vllm"


def test_reasoning_flag_overrides_task_type(router):
    b = _backends(vllm_local={"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"})
    backend, model = router._select_route("content", True, None, b)
    assert backend == "vllm_local"
    assert "r1" in model.lower() or "deepseek" in model.lower()


def test_explicit_vllm_local_override(router):
    b = _backends(vllm_local={"Qwen/Qwen3-4B"})
    backend, _ = router._select_route("simple", False, LLMBackend.VLLM_LOCAL, b)
    assert backend == "vllm_local"


def test_no_backend_raises_error(router):
    with pytest.raises(RuntimeError, match="No LLM backend available"):
        router._select_route("simple", False, None, _backends())


def test_last_resort_picks_any_vllm_local(router):
    b = _backends(vllm_local={"custom/MyModel"})
    backend, model = router._select_route("simple", False, None, b)
    assert backend == "vllm_local"
    assert model == "custom/MyModel"


# ── Request/Response models ─────────────────────────────────────────────────

def test_llm_request_defaults():
    req = LLMRequest(prompt="Hello")
    assert req.task_type == "simple"
    assert req.reasoning is False
    assert req.backend is None


def test_llm_response_defaults():
    resp = LLMResponse(content="Hi", backend=LLMBackend.VLLM_LOCAL, model="Qwen/Qwen3-4B")
    assert resp.latency_ms == 0.0
    assert resp.reasoning is None


# ── Integration (mocked) ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_complete_routes_to_vllm_local(router):
    router._backends = _backends(vllm_local={"Qwen/Qwen2.5-0.5B-Instruct"})
    router._backends_checked_at = 1e12
    mock_resp = LLMResponse(content="ok", backend=LLMBackend.VLLM_LOCAL, model="Qwen/Qwen2.5-0.5B-Instruct")
    router._openai_compat_complete = AsyncMock(return_value=mock_resp)
    resp = await router.complete(LLMRequest(prompt="Hi", task_type="simple"))
    assert resp.backend == LLMBackend.VLLM_LOCAL


@pytest.mark.asyncio
async def test_complete_falls_back_to_groq(router):
    router._backends = _backends(groq=True)
    router._backends_checked_at = 1e12
    mock_resp = LLMResponse(content="ok", backend=LLMBackend.GROQ, model="llama-3.3-70b-versatile")
    router._openai_compat_complete = AsyncMock(return_value=mock_resp)
    resp = await router.complete(LLMRequest(prompt="Hi", task_type="simple"))
    assert resp.backend == LLMBackend.GROQ


@pytest.mark.asyncio
async def test_reasoning_tags_parsed(router):
    router._backends = _backends(vllm_local={"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"})
    router._backends_checked_at = 1e12
    raw = LLMResponse(
        content="<think>Step 1\nStep 2</think>\nFinal answer",
        backend=LLMBackend.VLLM_LOCAL,
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    router._openai_compat_complete = AsyncMock(return_value=raw)
    resp = await router.complete(LLMRequest(prompt="Reason", task_type="reasoning", reasoning=True))
    assert resp.reasoning == "Step 1\nStep 2"
    assert resp.conclusion == "Final answer"


@pytest.mark.asyncio
async def test_generate_returns_string(router):
    router._backends = _backends(vllm_local={"Qwen/Qwen2.5-0.5B-Instruct"})
    router._backends_checked_at = 1e12
    router._openai_compat_complete = AsyncMock(return_value=LLMResponse(
        content="Generated", backend=LLMBackend.VLLM_LOCAL, model="Qwen/Qwen2.5-0.5B-Instruct"
    ))
    result = await router.generate("Hello")
    assert result == "Generated"


@pytest.mark.asyncio
async def test_reason_returns_structured_dict(router):
    router._backends = _backends(vllm_local={"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"})
    router._backends_checked_at = 1e12
    router._openai_compat_complete = AsyncMock(return_value=LLMResponse(
        content="<think>Step 1\nStep 2</think>\nConclusion",
        backend=LLMBackend.VLLM_LOCAL,
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ))
    result = await router.reason("Analyze this")
    assert result["reasoning_available"] is True
    assert result["conclusion"] == "Conclusion"
    assert "backend" in result


@pytest.mark.asyncio
async def test_vllm_remote_wins_over_groq(router):
    router._backends = _backends(vllm={"Qwen/Qwen3-30B-A3B"}, groq=True)
    router._backends_checked_at = 1e12
    mock_resp = LLMResponse(content="ok", backend=LLMBackend.VLLM, model="Qwen/Qwen3-30B-A3B")
    router._openai_compat_complete = AsyncMock(return_value=mock_resp)
    resp = await router.complete(LLMRequest(prompt="Analyze", task_type="analysis"))
    assert resp.backend == LLMBackend.VLLM


# ── Operating Mandate (Prime Directive 4) ────────────────────────────────────

def test_operating_mandate_is_non_empty():
    """The verbatim Operating Mandate must exist and contain the 90% confidence requirement."""
    assert OPERATING_MANDATE
    assert "90%" in OPERATING_MANDATE
    assert "never skip" in OPERATING_MANDATE


def test_mandate_system_prompt_contains_all_six_principles():
    """MANDATE_SYSTEM_PROMPT must encode all 6 operating principles."""
    assert "CONFIDENCE" in MANDATE_SYSTEM_PROMPT
    assert "QUALITY" in MANDATE_SYSTEM_PROMPT
    assert "THINK HARD" in MANDATE_SYSTEM_PROMPT
    assert "PARALLELIZE" in MANDATE_SYSTEM_PROMPT
    assert "FIX OR ASK" in MANDATE_SYSTEM_PROMPT
    assert "CONTEXT" in MANDATE_SYSTEM_PROMPT


def test_mandate_system_prompt_identifies_billy():
    """MANDATE_SYSTEM_PROMPT must identify BILLY and FiestyGoat AI LLC."""
    assert "BILLY" in MANDATE_SYSTEM_PROMPT
    assert "FiestyGoat AI LLC" in MANDATE_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_mandate_injected_without_task_system(router):
    """When request.system is empty, MANDATE_SYSTEM_PROMPT is the sole system message."""
    captured: list[dict] = []

    async def fake_post(self_client, url, *, json, headers, **kwargs):
        captured.append(json)
        class R:
            def raise_for_status(self): pass
            def json(self):
                return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 5}}
        return R()

    import httpx
    with patch.object(httpx.AsyncClient, "post", fake_post):
        router._backends = _backends(vllm_local={"Qwen/Qwen3-4B"})
        router._backends_checked_at = 1e12
        await router.complete(LLMRequest(prompt="Do something", task_type="simple"))

    msgs = captured[0]["messages"]
    system_msgs = [m for m in msgs if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == MANDATE_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_mandate_prepended_to_task_system(router):
    """When request.system is set, mandate is first and task context is appended after."""
    captured: list[dict] = []

    async def fake_post(self_client, url, *, json, headers, **kwargs):
        captured.append(json)
        class R:
            def raise_for_status(self): pass
            def json(self):
                return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 5}}
        return R()

    import httpx
    with patch.object(httpx.AsyncClient, "post", fake_post):
        router._backends = _backends(vllm_local={"Qwen/Qwen3-4B"})
        router._backends_checked_at = 1e12
        await router.complete(LLMRequest(
            prompt="Analyze market",
            system="You are a trading analyst.",
            task_type="trading",
        ))

    msgs = captured[0]["messages"]
    system_msgs = [m for m in msgs if m["role"] == "system"]
    assert len(system_msgs) == 1
    content = system_msgs[0]["content"]
    # Mandate comes first
    assert content.startswith(MANDATE_SYSTEM_PROMPT)
    # Task context is appended — cannot replace the mandate
    assert "You are a trading analyst." in content
    mandate_pos = content.find(MANDATE_SYSTEM_PROMPT)
    task_pos = content.find("You are a trading analyst.")
    assert mandate_pos < task_pos, "Mandate must appear before task-specific system prompt"


def test_mandate_cannot_be_overridden():
    """MANDATE_SYSTEM_PROMPT starts with BILLY identification — always the anchor."""
    assert MANDATE_SYSTEM_PROMPT.startswith("You are BILLY")
