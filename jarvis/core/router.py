"""LLMRouter v4 — vLLM-first, task-type-aware multi-backend routing (async).

BILLY (Billy Goat) — AI system by FiestyGoat AI LLC

Priority: vLLM Local (Mac Studio) → vLLM Remote (g5.xlarge) → Free Cloud → Paid Dev-Only

All backends use the OpenAI-compatible API (/v1/chat/completions).
Ollama is NOT used. All models served via vLLM.

--- What is HuggingFace vs LangGraph? ---
LangGraph  = orchestration engine. Builds the graphs, manages state, routes tasks,
             handles interrupt(). It's the workflow brain of BILLY.
HuggingFace = model registry (like an app store for model weights). When vLLM starts,
              it downloads model files from HuggingFace using model IDs like
              "Qwen/Qwen3-4B". That's all it does here.
vLLM       = inference server. Downloads model from HuggingFace, serves it via
             OpenAI-compatible API. LangGraph calls vLLM to generate text.
--- End explanation ---

Backends
--------
- vllm_local  : mlx_lm.server on Mac Studio M2 Max (localhost:8001) — primary, always-on, $0
                Start: venv/bin/python3.13 -m mlx_lm server --model mlx-community/Qwen3-4B-4bit --port 8001 --host 0.0.0.0
- vllm        : vLLM on AWS g5.xlarge — heavy workloads, on-demand ($0.73/hr)
                Set URL: keyring.set_password('jarvis_v2', 'vllm_base_url', 'http://<ip>:8000/v1')
- groq         : Groq cloud free tier (Qwen3-32B, DeepSeek R1 distills, fast)
                 Get key: https://console.groq.com (no credit card)
- deepseek     : DeepSeek API (V3.2 + R1 reasoner, $0.07/M tokens)
                 Get key: https://platform.deepseek.com
- openrouter   : OpenRouter free pool (Qwen3-Coder-480B, DeepSeek R1)
                 Get key: https://openrouter.ai
- grok         : xAI Grok API (key configured, for validation)
- claude       : Anthropic Claude — DEVELOPMENT USE ONLY

Models on Mac Studio (mlx_lm.server, 32GB RAM, Apple Metal):
  mlx-community/Qwen3-4B-4bit               ~2.5GB  — fast general (primary)
  Qwen/Qwen2.5-0.5B-Instruct               ~0.5GB  — ultra-fast simple tasks
  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  ~4.7GB  — local reasoning
  Qwen/Qwen2.5-Coder-1.5B-Instruct         ~1.0GB  — local coding
  mistralai/Mistral-7B-Instruct-v0.3       ~4.1GB  — creative writing

Models on g5.xlarge (vLLM remote, 24GB A10G VRAM):
  Qwen/Qwen3-30B-A3B                       ~18GB   — current primary (MoE)
  deepseek-ai/DeepSeek-R1-Distill-Qwen-14B ~8.5GB  — heavy reasoning
  mistralai/Devstral-Small-2505            ~14GB   — coding specialist
  Qwen/Qwen3-4B                            ~2.5GB  — fast routing model

Qwen3-Coder-480B-A35B — available via OpenRouter API (free tier).
  Full 480B model does NOT fit g5.xlarge (24GB VRAM). Use via cloud API.
"""

import re
import time
from enum import Enum
from typing import Any, Optional

import httpx
import structlog
from pydantic import BaseModel

from jarvis.core.mandate import MANDATE_SYSTEM_PROMPT

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Models / Enums
# ---------------------------------------------------------------------------

class LLMBackend(str, Enum):
    VLLM_LOCAL = "vllm_local"    # Mac Studio M2 Max — primary inference
    VLLM = "vllm"                # AWS g5.xlarge — heavy workloads
    GROQ = "groq"                # Free cloud, 70B inference
    DEEPSEEK = "deepseek"        # Cheap cloud, V3.2 + R1
    OPENROUTER = "openrouter"    # Free pool: Qwen3.5, Llama 3.1, Mistral
    GROK = "grok"                # xAI, configured
    CLAUDE = "claude"            # Dev-only — never use in production bots


class LLMRequest(BaseModel):
    prompt: str
    system: str = ""
    model: str | None = None
    backend: LLMBackend | None = None  # None = use task_type routing
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


# ---------------------------------------------------------------------------
# Task-type → (backend, model) routing table
#
# Uses HuggingFace model IDs for vLLM backends.
# All routes prefer free/local options. Cloud APIs used only when vLLM unavailable.
# ---------------------------------------------------------------------------

TASK_ROUTING: dict[str, list[tuple[str, str]]] = {
    # Fast, simple tasks — smallest/cheapest models first
    "simple": [
        ("vllm_local", "Qwen/Qwen2.5-0.5B-Instruct"),          # ultra-fast local
        ("vllm_local", "Qwen/Qwen3-4B"),                        # fast local
        ("vllm", "Qwen/Qwen3-4B"),                              # fast on GPU
        ("groq", "qwen/qwen3-32b"),                             # free cloud fallback
        ("deepseek", "deepseek-chat"),                           # cheap cloud fallback
    ],

    # Deep reasoning — DeepSeek R1 family
    "reasoning": [
        ("vllm_local", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        ("vllm", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
        ("vllm", "Qwen/Qwen3-30B-A3B"),
        ("deepseek", "deepseek-reasoner"),
        ("groq", "qwen/qwen3-32b"),                             # Qwen3 reasoning fallback
        ("openrouter", "qwen/qwen3-coder-480b-a35b-07-25:free"),
    ],

    # Code generation
    "coding": [
        ("vllm_local", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        ("vllm", "mistralai/Devstral-Small-2505"),              # Devstral-24B
        ("openrouter", "qwen/qwen3-coder-480b-a35b-07-25:free"), # Qwen3-Coder 480B free
        ("deepseek", "deepseek-chat"),
        ("groq", "qwen/qwen3-32b"),
    ],

    # Long context — Qwen3-Coder (256K) via OpenRouter, then GPU 128K
    "long_ctx": [
        ("openrouter", "qwen/qwen3-coder-480b-a35b-07-25:free"), # Qwen3-Coder 480B (256K)
        ("vllm", "Qwen/Qwen3-30B-A3B"),                        # GPU 128K
        ("vllm_local", "Qwen/Qwen3-4B"),                       # local 32K fallback
        ("groq", "qwen/qwen3-32b"),
    ],

    # Creative writing, captions, marketing copy
    "creative": [
        ("vllm_local", "mistralai/Mistral-7B-Instruct-v0.3"),
        ("vllm", "Qwen/Qwen3-30B-A3B"),
        ("groq", "qwen/qwen3-32b"),
        ("openrouter", "qwen/qwen3-coder-480b-a35b-07-25:free"),
        ("grok", "grok-4-fast-reasoning"),
    ],

    # Trading signals — reasoning essential
    "trading": [
        ("vllm_local", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        ("vllm", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
        ("deepseek", "deepseek-reasoner"),
        ("groq", "qwen/qwen3-32b"),
        ("openrouter", "qwen/qwen3-coder-480b-a35b-07-25:free"),
    ],

    # General analysis, scoring — GPU primary for quality
    "analysis": [
        ("vllm_local", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        ("vllm", "Qwen/Qwen3-30B-A3B"),
        ("deepseek", "deepseek-chat"),
        ("groq", "qwen/qwen3-32b"),
        ("openrouter", "qwen/qwen3-coder-480b-a35b-07-25:free"),
    ],

    # Content quality, Instagram/YouTube/TikTok copy
    "content": [
        ("vllm_local", "Qwen/Qwen3-4B"),
        ("vllm_local", "Qwen/Qwen2.5-0.5B-Instruct"),
        ("groq", "qwen/qwen3-32b"),
        ("deepseek", "deepseek-chat"),
    ],
}

# Default routing when task_type is unknown
_DEFAULT_ROUTING = TASK_ROUTING["simple"]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class LLMRouter:
    """Routes LLM calls to the best available backend for the given task type.

    Priority: vLLM Local (Mac) → vLLM Remote (g5) → Groq → DeepSeek
              → OpenRouter → Grok → Claude (dev-only)

    Usage::

        router = LLMRouter()

        # Simple routing by task type
        text = await router.generate("Summarize this...", task_type="simple")

        # Force reasoning path (routes to DeepSeek R1)
        result = await router.reason("Analyze this trade...", context={...})

        # Explicit backend override
        resp = await router.complete(LLMRequest(
            prompt="...",
            backend=LLMBackend.VLLM_LOCAL,
            task_type="coding",
        ))
    """

    # Model constants for common use cases
    REASONING_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    STANDARD_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    FAST_MODEL = "Qwen/Qwen3-4B"

    def __init__(self, settings=None, memory=None):
        from config.settings import get_settings
        self.settings = settings or get_settings()
        self._memory = memory
        self._backends: dict[str, Any] | None = None
        self._backends_checked_at: float = 0.0

    # ------------------------------------------------------------------
    # Backend detection — cached 5 minutes
    # ------------------------------------------------------------------

    async def _detect_backends(self) -> dict[str, Any]:
        """Probe all backends. Returns availability dict cached for 5 min."""
        now = time.monotonic()
        if self._backends is not None and now - self._backends_checked_at < 300:
            return self._backends

        backends: dict[str, Any] = {
            "vllm_local": set(),   # HuggingFace model IDs served locally
            "vllm": set(),         # HuggingFace model IDs on g5.xlarge
            "groq": False,
            "deepseek": False,
            "openrouter": False,
            "grok": False,
            "claude": False,
        }

        # vLLM local — probe localhost:8000/v1/models
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(f"{self.settings.vllm_local_base_url}/models")
                if resp.status_code == 200:
                    backends["vllm_local"] = {
                        m.get("id", "") for m in resp.json().get("data", [])
                    }
                    log.info("router.vllm_local_up", models=len(backends["vllm_local"]))
        except Exception as exc:
            log.debug("router.vllm_local_down", error=str(exc))

        # vLLM remote — g5.xlarge URL from Keychain
        vllm_url = self._get_keychain_key("vllm_base_url") or self.settings.vllm_base_url
        if vllm_url:
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    resp = await client.get(f"{vllm_url}/models")
                    if resp.status_code == 200:
                        backends["vllm"] = {
                            m.get("id", "") for m in resp.json().get("data", [])
                        }
                        log.info("router.vllm_remote_up", models=len(backends["vllm"]), url=vllm_url)
            except Exception as exc:
                log.debug("router.vllm_remote_down", error=str(exc))

        # Cloud APIs — key presence check only
        for backend, key_name in [
            ("groq", "groq_api_key"),
            ("deepseek", "deepseek_api_key"),
            ("openrouter", "openrouter_api_key"),
            ("grok", "xai_api_key"),
            ("claude", "anthropic_api_key"),
        ]:
            val = self._get_keychain_key(key_name)
            backends[backend] = bool(val)

        log.info(
            "router.backends_detected",
            vllm_local=len(backends["vllm_local"]),
            vllm_remote=len(backends["vllm"]),
            groq=backends["groq"],
            deepseek=backends["deepseek"],
            openrouter=backends["openrouter"],
            grok=backends["grok"],
        )
        self._backends = backends
        self._backends_checked_at = now
        return backends

    def _is_available(self, backend_name: str, model: str, backends: dict[str, Any]) -> bool:
        """Check if a specific (backend, model) pair is usable."""
        if backend_name in ("vllm_local", "vllm"):
            available = backends.get(backend_name, set())
            if not available:
                return False
            return any(
                model.lower() in m.lower() or m.lower() in model.lower()
                for m in available
            )
        return bool(backends.get(backend_name, False))

    def _select_route(
        self,
        task_type: str,
        reasoning: bool,
        backend_override: LLMBackend | None,
        backends: dict[str, Any],
    ) -> tuple[str, str]:
        """Pick the best available (backend_name, model_name) for this request."""

        if backend_override is not None:
            bname = backend_override.value
            override_models = {
                "vllm_local": (self.REASONING_MODEL if reasoning else self.FAST_MODEL),
                "vllm": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
                         if reasoning else "Qwen/Qwen3-30B-A3B"),
                "groq": ("qwen/qwen3-32b"
                         if reasoning else "qwen/qwen3-32b"),
                "deepseek": ("deepseek-reasoner" if reasoning else "deepseek-chat"),
                "grok": "grok-4-fast-reasoning",
                "claude": "claude-sonnet-4-6",
            }
            return bname, override_models.get(bname, self.STANDARD_MODEL)

        effective_task = "reasoning" if reasoning else task_type
        routing = TASK_ROUTING.get(effective_task, _DEFAULT_ROUTING)

        for backend_name, model in routing:
            if self._is_available(backend_name, model, backends):
                log.debug("router.route_selected", task_type=effective_task,
                          backend=backend_name, model=model)
                return backend_name, model

        # Last resort — any vLLM local model
        vllm_local_available = backends.get("vllm_local", set())
        if vllm_local_available:
            first = next(iter(vllm_local_available))
            return "vllm_local", first

        raise RuntimeError(
            f"No LLM backend available for task_type={task_type!r}. "
            "Start mlx_lm: venv/bin/python3.13 -m mlx_lm server --model mlx-community/Qwen3-4B-4bit --port 8001"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Route the request to the best available backend."""
        backends = await self._detect_backends()
        backend_name, model = self._select_route(
            task_type=request.task_type,
            reasoning=request.reasoning,
            backend_override=request.backend,
            backends=backends,
        )
        model = request.model or model

        log.info("router.dispatch", backend=backend_name, model=model,
                 task_type=request.task_type, reasoning=request.reasoning,
                 prompt_len=len(request.prompt))

        t0 = time.monotonic()

        if backend_name == "vllm_local":
            resp = await self._openai_compat_complete(
                request, model,
                base_url=self.settings.vllm_local_base_url,
                api_key="",
                backend_enum=LLMBackend.VLLM_LOCAL,
            )
        elif backend_name == "vllm":
            vllm_url = self._get_keychain_key("vllm_base_url") or self.settings.vllm_base_url
            resp = await self._openai_compat_complete(
                request, model,
                base_url=vllm_url,
                api_key="",
                backend_enum=LLMBackend.VLLM,
            )
        elif backend_name == "groq":
            resp = await self._openai_compat_complete(
                request, model,
                base_url=self.settings.groq_base_url,
                api_key=self._get_keychain_key("groq_api_key") or "",
                backend_enum=LLMBackend.GROQ,
            )
        elif backend_name == "deepseek":
            resp = await self._openai_compat_complete(
                request, model,
                base_url=self.settings.deepseek_base_url,
                api_key=self._get_keychain_key("deepseek_api_key") or "",
                backend_enum=LLMBackend.DEEPSEEK,
            )
        elif backend_name == "openrouter":
            resp = await self._openai_compat_complete(
                request, model,
                base_url=self.settings.openrouter_base_url,
                api_key=self._get_keychain_key("openrouter_api_key") or "",
                backend_enum=LLMBackend.OPENROUTER,
                extra_headers={
                    "HTTP-Referer": "https://github.com/grimreaper0/jarvis_v2",
                    "X-Title": "BILLY — FiestyGoat AI",
                },
            )
        elif backend_name == "grok":
            resp = await self._grok_complete(request, model)
        elif backend_name == "claude":
            resp = await self._claude_complete(request)
        else:
            raise RuntimeError(f"Unknown backend: {backend_name}")

        resp.latency_ms = round((time.monotonic() - t0) * 1000, 1)

        if request.reasoning or "deepseek-r1" in model.lower() or "reasoner" in model.lower():
            resp = _parse_reasoning(resp)

        await self._log_usage(resp, request.task_type)
        return resp

    # Convenience wrappers
    async def generate(
        self,
        prompt: str,
        task_type: str = "simple",
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        resp = await self.complete(LLMRequest(
            prompt=prompt, task_type=task_type, max_tokens=max_tokens, temperature=temperature,
        ))
        return resp.content

    async def reason(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Use the reasoning model (DeepSeek R1) for complex analysis."""
        full_prompt = prompt
        if context:
            ctx_lines = "\n".join(f"- {k}: {v}" for k, v in context.items())
            full_prompt = f"{prompt}\n\nContext:\n{ctx_lines}"

        resp = await self.complete(LLMRequest(
            prompt=full_prompt, task_type="reasoning", reasoning=True,
            max_tokens=max_tokens, temperature=temperature,
        ))
        return {
            "reasoning": resp.reasoning,
            "conclusion": resp.conclusion or resp.content,
            "raw_response": resp.content,
            "model": resp.model,
            "backend": resp.backend,
            "reasoning_available": resp.reasoning is not None,
            "reasoning_steps": len(resp.reasoning.split("\n")) if resp.reasoning else 0,
        }

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    async def _openai_compat_complete(
        self,
        request: LLMRequest,
        model: str,
        base_url: str,
        api_key: str,
        backend_enum: LLMBackend,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Shared handler for all OpenAI-compatible APIs: vLLM, Groq, DeepSeek, OpenRouter."""
        # Operating Mandate (Prime Directive 4) is always the first system message.
        # Task-specific system prompts are appended after — they cannot override the mandate.
        if request.system:
            system_content = f"{MANDATE_SYSTEM_PROMPT}\n\n---\nTask context:\n{request.system}"
        else:
            system_content = MANDATE_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": request.prompt},
        ]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        log.info(f"{backend_enum.value}.response", tokens=tokens, model=model)

        return LLMResponse(content=content, backend=backend_enum, model=model, tokens_used=tokens)

    async def _grok_complete(self, request: LLMRequest, model: str) -> LLMResponse:
        api_key = self._get_keychain_key("xai_api_key")
        if not api_key:
            raise RuntimeError("xai_api_key not found in Keychain")
        return await self._openai_compat_complete(
            request, model or "grok-4-fast-reasoning",
            base_url="https://api.x.ai/v1",
            api_key=api_key,
            backend_enum=LLMBackend.GROK,
        )

    async def _claude_complete(self, request: LLMRequest) -> LLMResponse:
        """Anthropic Claude — DEVELOPMENT USE ONLY. Never call from production bots."""
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

        api_key = self._get_keychain_key("anthropic_api_key")
        if not api_key:
            raise RuntimeError(
                "anthropic_api_key not found in Keychain. "
                "Claude is for development only — use vLLM or free cloud APIs in BILLY."
            )

        client = anthropic.AsyncAnthropic(api_key=api_key)
        model = request.model or "claude-sonnet-4-6"
        messages = [{"role": "user", "content": request.prompt}]
        kwargs: dict[str, Any] = {"model": model, "max_tokens": request.max_tokens, "messages": messages}
        if request.system:
            kwargs["system"] = request.system

        resp = await client.messages.create(**kwargs)
        content = resp.content[0].text
        tokens = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
        log.info("claude.response", input_tokens=resp.usage.input_tokens,
                 output_tokens=resp.usage.output_tokens)
        return LLMResponse(content=content, backend=LLMBackend.CLAUDE, model=model, tokens_used=tokens)

    # ------------------------------------------------------------------
    # LangChain integration (for LangGraph nodes that use as_llm pattern)
    # ------------------------------------------------------------------

    def as_langchain_llm(
        self,
        model: str | None = None,
        reasoning: bool = False,
        task_type: str = "simple",
    ):
        """Return a LangChain-compatible LLM backed by vLLM local (OpenAI-compat).

        For async LangGraph nodes, use ``await router.complete(...)`` directly.
        This helper is for nodes that require a synchronous LangChain BaseLLM.
        """
        try:
            from langchain_openai import ChatOpenAI
            selected = model or (self.REASONING_MODEL if reasoning else self.FAST_MODEL)
            return ChatOpenAI(
                base_url=self.settings.vllm_local_base_url,
                api_key="none",  # vLLM local: no auth required
                model=selected,
            )
        except ImportError:
            raise RuntimeError(
                "Install langchain-openai for vLLM integration: pip install langchain-openai"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_keychain_key(self, key_name: str) -> str | None:
        try:
            import keyring
            for service in ("jarvis_v2", "personal-agent-hub", "personal_agent_hub"):
                val = keyring.get_password(service, key_name)
                if val:
                    return val
            return None
        except Exception:
            return None

    async def _log_usage(self, resp: LLMResponse, task_type: str) -> None:
        log.info("router.usage", backend=resp.backend, model=resp.model,
                 tokens=resp.tokens_used, latency_ms=resp.latency_ms, task_type=task_type)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_reasoning(resp: LLMResponse) -> LLMResponse:
    """Parse DeepSeek R1 <think>...</think> tags from any backend's response."""
    match = re.search(r"<think>(.*?)</think>", resp.content, re.DOTALL | re.IGNORECASE)
    if match:
        resp.reasoning = match.group(1).strip()
        resp.conclusion = resp.content[match.end():].strip()
    else:
        resp.reasoning = None
        resp.conclusion = resp.content
    return resp
