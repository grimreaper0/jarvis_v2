"""LLMRouter v2 — Ollama-first with Claude API / Grok HTTP fallback (async).

Ported from jarvis_v1 utils/llm_router.py with these improvements:
- Fully async (httpx for Ollama and Grok, anthropic SDK for Claude)
- Backends: Ollama (primary, $0) → Claude API (secondary) → Grok HTTP (tertiary)
- as_langchain_llm() returns a LangChain-compatible LLM for use in LangGraph nodes
- Free-first principle preserved: never pay when Ollama works
- Token tracking logged to AutoMem audit
- Models: qwen2.5:0.5b (fast), deepseek-r1:7b (reasoning)
"""

import re
import time
from enum import Enum
from typing import Any, Optional

import httpx
import structlog
from pydantic import BaseModel

log = structlog.get_logger()


class LLMBackend(str, Enum):
    OLLAMA = "ollama"
    CLAUDE = "claude"
    GROK = "grok"


class LLMRequest(BaseModel):
    prompt: str
    system: str = ""
    model: str | None = None
    backend: LLMBackend = LLMBackend.OLLAMA
    reasoning: bool = False
    max_tokens: int = 2048
    temperature: float = 0.7
    task_type: str = "simple"


class LLMResponse(BaseModel):
    content: str
    backend: LLMBackend
    model: str
    tokens_used: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    reasoning: Optional[str] = None
    conclusion: Optional[str] = None


class LLMRouter:
    """Routes LLM calls to the appropriate backend.

    Priority: Ollama (local, $0) → Claude API → Grok HTTP
    Reasoning tasks always route to deepseek-r1:7b via Ollama.

    Usage::

        router = LLMRouter()
        resp = await router.complete(LLMRequest(prompt="Summarize this..."))
        print(resp.content)

        # Reasoning with DeepSeek R1
        resp = await router.complete(LLMRequest(
            prompt="Analyze this trading opportunity...",
            reasoning=True,
        ))
        print(resp.reasoning)   # chain-of-thought
        print(resp.conclusion)  # final answer
    """

    # Model names — match jarvis_v1 production
    REASONING_MODEL = "deepseek-r1:7b"
    STANDARD_MODEL = "qwen2.5:0.5b"

    def __init__(self, settings=None, memory=None):
        from config.settings import get_settings
        self.settings = settings or get_settings()
        self._memory = memory
        self._backends: dict[str, bool] | None = None
        self._backends_checked_at: float = 0.0

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    async def _detect_backends(self) -> dict[str, bool]:
        """Probe available backends. Results cached for 5 minutes."""
        now = time.monotonic()
        if self._backends is not None and now - self._backends_checked_at < 300:
            return self._backends

        backends: dict[str, bool] = {
            "ollama": False,
            "ollama_reasoning": False,
            "claude": False,
            "grok": False,
        }

        # Ollama probe
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(f"{self.settings.ollama_base_url}/api/tags")
                if resp.status_code == 200:
                    models = [m.get("name", "") for m in resp.json().get("models", [])]
                    backends["ollama"] = any(self.STANDARD_MODEL in n for n in models)
                    backends["ollama_reasoning"] = any(
                        self.REASONING_MODEL in n or "deepseek-r1" in n for n in models
                    )
        except Exception as exc:
            log.warning("router.ollama_probe_failed", error=str(exc))

        # Claude probe (key presence check only)
        try:
            claude_key = self._get_keychain_key("anthropic_api_key")
            backends["claude"] = bool(claude_key)
        except Exception:
            pass

        # Grok probe (key presence check only)
        try:
            grok_key = self._get_keychain_key("xai_api_key")
            backends["grok"] = bool(grok_key)
        except Exception:
            pass

        log.info("router.backends_detected", **backends)
        self._backends = backends
        self._backends_checked_at = now
        return backends

    def _get_keychain_key(self, key_name: str) -> str | None:
        """Attempt to load a key from macOS Keychain via keyring."""
        try:
            import keyring
            val = keyring.get_password("personal_agent_hub", key_name)
            return val if val else None
        except Exception:
            return None

    def _select_backend(self, prefer: list[str], backends: dict[str, bool]) -> str:
        for b in prefer:
            if backends.get(b):
                return b
        raise RuntimeError(
            f"No available LLM backend from preference list {prefer}. "
            "Ensure Ollama is running or API keys are configured."
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Route the request to the best available backend and return a response."""
        backends = await self._detect_backends()

        if request.reasoning:
            preferred = ["ollama_reasoning", "grok", "claude"]
        else:
            preferred = ["ollama", "grok", "claude"]

        # Allow explicit backend override
        if request.backend == LLMBackend.CLAUDE:
            preferred = ["claude"]
        elif request.backend == LLMBackend.GROK:
            preferred = ["grok"]

        backend_name = self._select_backend(preferred, backends)

        log.info(
            "router.dispatch",
            backend=backend_name,
            reasoning=request.reasoning,
            task_type=request.task_type,
            prompt_len=len(request.prompt),
        )

        t0 = time.monotonic()

        if backend_name in ("ollama", "ollama_reasoning"):
            model = request.model or (
                self.REASONING_MODEL if request.reasoning else self.STANDARD_MODEL
            )
            resp = await self._ollama_complete(request, model)
        elif backend_name == "claude":
            resp = await self._claude_complete(request)
        elif backend_name == "grok":
            resp = await self._grok_complete(request)
        else:
            raise RuntimeError(f"Unknown backend: {backend_name}")

        resp.latency_ms = round((time.monotonic() - t0) * 1000, 1)

        # Parse DeepSeek R1 reasoning tags
        if request.reasoning and resp.backend == LLMBackend.OLLAMA:
            resp = _parse_reasoning(resp)

        await self._log_usage(resp, request.task_type)
        return resp

    # Convenience alias matching jarvis_v1 interface
    async def generate(
        self,
        prompt: str,
        task_type: str = "simple",
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate text. Returns the content string directly."""
        resp = await self.complete(LLMRequest(
            prompt=prompt,
            task_type=task_type,
            max_tokens=max_tokens,
            temperature=temperature,
        ))
        return resp.content

    async def reason(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Use the reasoning model (DeepSeek R1) for complex analysis.

        Returns dict with keys: reasoning, conclusion, raw_response, model.
        """
        full_prompt = prompt
        if context:
            ctx_lines = "\n".join(f"- {k}: {v}" for k, v in context.items())
            full_prompt = f"{prompt}\n\nContext:\n{ctx_lines}"

        resp = await self.complete(LLMRequest(
            prompt=full_prompt,
            reasoning=True,
            max_tokens=max_tokens,
            temperature=temperature,
        ))
        return {
            "reasoning": resp.reasoning,
            "conclusion": resp.conclusion or resp.content,
            "raw_response": resp.content,
            "model": resp.model,
            "reasoning_available": resp.reasoning is not None,
            "reasoning_steps": len(resp.reasoning.split("\n")) if resp.reasoning else 0,
        }

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    async def _ollama_complete(self, request: LLMRequest, model: str) -> LLMResponse:
        payload = {
            "model": model,
            "prompt": request.prompt,
            "system": request.system,
            "stream": False,
            "options": {
                "num_ctx": 2048,
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
                "keep_alive": -1,
            },
        }
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{self.settings.ollama_base_url}/api/generate", json=payload
            )
            resp.raise_for_status()
            data = resp.json()

        return LLMResponse(
            content=data["response"],
            backend=LLMBackend.OLLAMA,
            model=model,
            tokens_used=data.get("eval_count", 0),
        )

    async def _claude_complete(self, request: LLMRequest) -> LLMResponse:
        """Call Anthropic Claude API (claude-sonnet-4-6 by default)."""
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

        api_key = self._get_keychain_key("anthropic_api_key")
        if not api_key:
            raise RuntimeError("anthropic_api_key not found in Keychain")

        client = anthropic.AsyncAnthropic(api_key=api_key)
        model = request.model or "claude-sonnet-4-6"

        messages = [{"role": "user", "content": request.prompt}]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }
        if request.system:
            kwargs["system"] = request.system

        resp = await client.messages.create(**kwargs)
        content = resp.content[0].text
        tokens = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)

        log.info("claude.response", input_tokens=resp.usage.input_tokens,
                 output_tokens=resp.usage.output_tokens)
        return LLMResponse(
            content=content,
            backend=LLMBackend.CLAUDE,
            model=model,
            tokens_used=tokens,
        )

    async def _grok_complete(self, request: LLMRequest) -> LLMResponse:
        """Call xAI Grok via OpenAI-compatible REST endpoint."""
        api_key = self._get_keychain_key("xai_api_key")
        if not api_key:
            raise RuntimeError("xai_api_key not found in Keychain")

        model = request.model or "grok-4-fast-reasoning"
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        log.info("grok.response", tokens=tokens, model=model)

        return LLMResponse(
            content=content,
            backend=LLMBackend.GROK,
            model=model,
            tokens_used=tokens,
        )

    # ------------------------------------------------------------------
    # LangChain integration
    # ------------------------------------------------------------------

    def as_langchain_llm(self, model: str | None = None, reasoning: bool = False):
        """Return a LangChain-compatible LLM object backed by this router.

        The returned object can be used directly in LangGraph nodes as a
        LangChain BaseLLM. It wraps the async router in a sync-compatible
        interface via OllamaLLM from langchain-ollama.

        For LangGraph async nodes, use the router directly via `await router.complete(...)`.
        """
        try:
            from langchain_ollama import OllamaLLM
            selected_model = model or (
                self.REASONING_MODEL if reasoning else self.STANDARD_MODEL
            )
            return OllamaLLM(
                base_url=self.settings.ollama_base_url,
                model=selected_model,
            )
        except ImportError:
            raise RuntimeError(
                "langchain-ollama not installed. Run: pip install langchain-ollama"
            )

    # ------------------------------------------------------------------
    # Token/cost tracking
    # ------------------------------------------------------------------

    async def _log_usage(self, resp: LLMResponse, task_type: str) -> None:
        """Log token usage to structlog (AutoMem logging is a future enhancement)."""
        log.info(
            "router.usage",
            backend=resp.backend,
            model=resp.model,
            tokens=resp.tokens_used,
            latency_ms=resp.latency_ms,
            task_type=task_type,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_reasoning(resp: LLMResponse) -> LLMResponse:
    """Parse DeepSeek R1 <think>...</think> tags from response content."""
    match = re.search(r"<think>(.*?)</think>", resp.content, re.DOTALL | re.IGNORECASE)
    if match:
        resp.reasoning = match.group(1).strip()
        resp.conclusion = resp.content[match.end():].strip()
    else:
        resp.reasoning = None
        resp.conclusion = resp.content
    return resp
