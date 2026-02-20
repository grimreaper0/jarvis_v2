"""LLMRouter v3 — Free-first, task-type-aware multi-backend routing (async).

Priority: Local Ollama ($0) → Free Cloud APIs (Groq/DeepSeek/OpenRouter) → Paid (Claude/Grok)

Backends
--------
- ollama      : Local models (qwen2.5:0.5b, deepseek-r1:7b, etc.) — always $0
- groq         : Groq cloud (llama-3.3-70b, deepseek-r1 distills) — free tier
                 Get key: https://console.groq.com (no credit card required)
- deepseek     : DeepSeek API (deepseek-chat=V3.2, deepseek-reasoner=R1) — very cheap
                 Get key: https://platform.deepseek.com
- openrouter   : OpenRouter free pool (Qwen3-235B, Llama 3.1, Mistral) — free models
                 Get key: https://openrouter.ai
- claude       : Anthropic Claude API — development use only
- grok         : xAI Grok API — has key, use when needed

Add API keys via Keychain:
    python3 -c "import keyring; keyring.set_password('jarvis_v2', 'groq_api_key', 'gsk_...')"
    python3 -c "import keyring; keyring.set_password('jarvis_v2', 'deepseek_api_key', 'sk-...')"
    python3 -c "import keyring; keyring.set_password('jarvis_v2', 'openrouter_api_key', 'sk-or-...')"

Task-type routing
-----------------
Pass task_type in LLMRequest to get the optimal model for the job:

    await router.complete(LLMRequest(prompt="...", task_type="reasoning"))
    await router.complete(LLMRequest(prompt="...", task_type="coding"))

Valid task_types: simple, reasoning, coding, long_ctx, creative, trading, analysis, content
"""

import re
import time
from enum import Enum
from typing import Any, Optional

import httpx
import structlog
from pydantic import BaseModel

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Models / Enums
# ---------------------------------------------------------------------------

class LLMBackend(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"          # AWS g5.xlarge: Qwen3-30B, DeepSeek-R1-14B, Devstral-24B
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"
    CLAUDE = "claude"
    GROK = "grok"


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
# Each entry is an ordered list of (backend_name, model_name) tuples.
# The router picks the first tuple whose backend is available.
# ---------------------------------------------------------------------------

TASK_ROUTING: dict[str, list[tuple[str, str]]] = {
    # Fast, simple tasks — cheapest/fastest models first
    # qwen3:4b local → vLLM Qwen3-4B (fast router) → Groq
    "simple": [
        ("ollama", "qwen3:4b"),           # Qwen3 local (new, ~2.5GB) — pull to activate
        ("ollama", "qwen2.5:0.5b"),       # existing fast local
        ("ollama", "llama3.2:3b"),        # existing general local
        ("vllm", "Qwen/Qwen3-4B"),        # GPU server fast router (port 8001)
        ("groq", "llama-3.3-70b-versatile"),
        ("openrouter", "meta-llama/llama-3.1-8b-instruct:free"),
    ],

    # Deep reasoning, chain-of-thought — DeepSeek R1 family + Qwen3-30B
    "reasoning": [
        ("ollama", "deepseek-r1:7b"),          # existing local
        ("vllm", "deepseek-r1-14b"),           # GPU server: DeepSeek-R1-14B (better)
        ("vllm", "Qwen/Qwen3-30B-A3B"),        # GPU server: MoE primary
        ("groq", "deepseek-r1-distill-llama-70b"),
        ("deepseek", "deepseek-reasoner"),
        ("openrouter", "deepseek/deepseek-r1:free"),
    ],

    # Code generation — Devstral-24B on GPU server, qwen-coder locally
    "coding": [
        ("ollama", "qwen2.5-coder:1.5b"),          # existing local
        ("vllm", "mistralai/Devstral-Small-2505"),  # GPU server: Devstral-24B coding
        ("groq", "llama-3.3-70b-versatile"),
        ("openrouter", "qwen/qwen-2.5-coder-32b-instruct:free"),
        ("deepseek", "deepseek-chat"),
    ],

    # Long documents, big context — Qwen3-30B excels here (128K ctx)
    "long_ctx": [
        ("vllm", "Qwen/Qwen3-30B-A3B"),           # GPU server: 128K context window
        ("ollama", "qwen2.5:14b"),                 # local large (install if needed)
        ("groq", "llama-3.1-70b-versatile"),       # Groq has large context
        ("openrouter", "qwen/qwen3-235b-a22b:free"),
    ],

    # Creative writing, captions, marketing copy
    "creative": [
        ("ollama", "mistral:7b"),                  # local writing model
        ("vllm", "Qwen/Qwen3-30B-A3B"),            # GPU: best quality for creative
        ("groq", "mixtral-8x7b-32768"),
        ("openrouter", "mistralai/mistral-7b-instruct:free"),
        ("grok", "grok-4-fast-reasoning"),
    ],

    # Trading signals, market analysis — reasoning required
    "trading": [
        ("ollama", "deepseek-r1:7b"),
        ("vllm", "deepseek-r1-14b"),               # GPU server: better reasoning
        ("groq", "deepseek-r1-distill-llama-70b"),
        ("deepseek", "deepseek-reasoner"),
        ("groq", "llama-3.3-70b-versatile"),
    ],

    # General analysis, scoring, evaluation — Qwen3-30B as top quality option
    "analysis": [
        ("ollama", "deepseek-r1:7b"),
        ("vllm", "Qwen/Qwen3-30B-A3B"),            # GPU server: primary model
        ("groq", "llama-3.3-70b-versatile"),
        ("deepseek", "deepseek-chat"),
    ],

    # Content quality, Instagram/YouTube/TikTok copy — fast is fine
    "content": [
        ("ollama", "qwen3:4b"),           # Qwen3 local if installed
        ("ollama", "qwen2.5:0.5b"),
        ("ollama", "llama3.2:3b"),
        ("groq", "llama-3.3-70b-versatile"),
        ("openrouter", "meta-llama/llama-3.1-8b-instruct:free"),
    ],
}

# Default routing when task_type is unknown
_DEFAULT_ROUTING = TASK_ROUTING["simple"]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class LLMRouter:
    """Routes LLM calls to the best available backend for the given task type.

    Priority: Ollama (local, $0) → Groq (free) → DeepSeek → OpenRouter → Grok → Claude
    Claude is reserved for development use only.

    Usage::

        router = LLMRouter()

        # Simple routing by task type
        resp = await router.complete(LLMRequest(
            prompt="Summarize this...",
            task_type="simple",
        ))

        # Force reasoning model
        resp = await router.complete(LLMRequest(
            prompt="Analyze this trading opportunity...",
            task_type="reasoning",
        ))
        print(resp.reasoning)   # chain-of-thought (DeepSeek R1)
        print(resp.conclusion)  # final answer

        # Convenience wrappers
        text = await router.generate("Summarize this...", task_type="simple")
        result = await router.reason("Evaluate this trade...", context={...})
    """

    # Legacy model constants (kept for backward compat)
    REASONING_MODEL = "deepseek-r1:7b"
    STANDARD_MODEL = "qwen2.5:0.5b"

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

        # Structure: {backend_name: True/False or set of model names for ollama}
        backends: dict[str, Any] = {
            "ollama": set(),   # set of available model names
            "vllm": set(),     # set of available model IDs on GPU server
            "groq": False,
            "deepseek": False,
            "openrouter": False,
            "claude": False,
            "grok": False,
        }

        # Ollama — probe tags endpoint
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(f"{self.settings.ollama_base_url}/api/tags")
                if resp.status_code == 200:
                    backends["ollama"] = {
                        m.get("name", "") for m in resp.json().get("models", [])
                    }
        except Exception as exc:
            log.warning("router.ollama_probe_failed", error=str(exc))

        # vLLM GPU server — probe /v1/models endpoint
        vllm_url = self._get_keychain_key("vllm_base_url") or self.settings.vllm_base_url
        if vllm_url:
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    resp = await client.get(f"{vllm_url}/models")
                    if resp.status_code == 200:
                        backends["vllm"] = {
                            m.get("id", "") for m in resp.json().get("data", [])
                        }
                        log.info("router.vllm_available", models=len(backends["vllm"]), url=vllm_url)
            except Exception as exc:
                log.debug("router.vllm_unavailable", error=str(exc))

        # Cloud APIs — key presence check only (lazy, no network call)
        for backend, key_name in [
            ("groq", "groq_api_key"),
            ("deepseek", "deepseek_api_key"),
            ("openrouter", "openrouter_api_key"),
            ("claude", "anthropic_api_key"),
            ("grok", "xai_api_key"),
        ]:
            val = self._get_keychain_key(key_name)
            backends[backend] = bool(val)

        log.info(
            "router.backends_detected",
            ollama_models=len(backends["ollama"]),
            groq=backends["groq"],
            deepseek=backends["deepseek"],
            openrouter=backends["openrouter"],
            claude=backends["claude"],
            grok=backends["grok"],
        )
        self._backends = backends
        self._backends_checked_at = now
        return backends

    def _is_available(self, backend_name: str, model: str, backends: dict[str, Any]) -> bool:
        """Check if a specific (backend, model) pair is usable."""
        if backend_name == "ollama":
            available = backends.get("ollama", set())
            # Match exact name or prefix (e.g. "deepseek-r1:7b" matches "deepseek-r1:7b")
            return any(model in m or m.startswith(model.split(":")[0]) for m in available)
        if backend_name == "vllm":
            available = backends.get("vllm", set())
            # vLLM model IDs are full HF names e.g. "Qwen/Qwen3-30B-A3B"
            return bool(available) and any(
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

        # Explicit backend override — use it with task-appropriate model
        if backend_override is not None:
            bname = backend_override.value
            if backend_override == LLMBackend.OLLAMA:
                model = self.REASONING_MODEL if reasoning else self.STANDARD_MODEL
            elif backend_override == LLMBackend.VLLM:
                model = "deepseek-r1-14b" if reasoning else "Qwen/Qwen3-30B-A3B"
            elif backend_override == LLMBackend.GROQ:
                model = "deepseek-r1-distill-llama-70b" if reasoning else "llama-3.3-70b-versatile"
            elif backend_override == LLMBackend.DEEPSEEK:
                model = "deepseek-reasoner" if reasoning else "deepseek-chat"
            elif backend_override == LLMBackend.CLAUDE:
                model = "claude-sonnet-4-6"
            elif backend_override == LLMBackend.GROK:
                model = "grok-4-fast-reasoning"
            else:
                model = self.STANDARD_MODEL
            return bname, model

        # Reasoning flag shortcut → force reasoning task type
        effective_task = "reasoning" if reasoning else task_type
        routing = TASK_ROUTING.get(effective_task, _DEFAULT_ROUTING)

        for backend_name, model in routing:
            if self._is_available(backend_name, model, backends):
                log.debug(
                    "router.route_selected",
                    task_type=effective_task,
                    backend=backend_name,
                    model=model,
                )
                return backend_name, model

        # Last-resort fallback — anything available
        if backends.get("ollama"):
            first_model = next(iter(backends["ollama"]))
            return "ollama", first_model

        raise RuntimeError(
            f"No LLM backend available for task_type={task_type!r}. "
            "Ensure Ollama is running (ollama serve) or add free API keys to Keychain. "
            "See router.py docstring for setup instructions."
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Route the request to the best available backend and return a response."""
        backends = await self._detect_backends()
        backend_name, model = self._select_route(
            task_type=request.task_type,
            reasoning=request.reasoning,
            backend_override=request.backend,
            backends=backends,
        )
        # Use caller-specified model if provided
        model = request.model or model

        log.info(
            "router.dispatch",
            backend=backend_name,
            model=model,
            task_type=request.task_type,
            reasoning=request.reasoning,
            prompt_len=len(request.prompt),
        )

        t0 = time.monotonic()

        if backend_name == "ollama":
            resp = await self._ollama_complete(request, model)
        elif backend_name == "vllm":
            vllm_url = self._get_keychain_key("vllm_base_url") or self.settings.vllm_base_url
            resp = await self._openai_compat_complete(
                request, model,
                base_url=vllm_url,
                api_key="",  # vLLM typically unauthenticated on private network
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
                    "X-Title": "jarvis_v2",
                },
            )
        elif backend_name == "claude":
            resp = await self._claude_complete(request)
        elif backend_name == "grok":
            resp = await self._grok_complete(request, model)
        else:
            raise RuntimeError(f"Unknown backend: {backend_name}")

        resp.latency_ms = round((time.monotonic() - t0) * 1000, 1)

        # Parse DeepSeek R1 <think> tags regardless of backend
        if request.reasoning or "deepseek-r1" in model or "reasoner" in model:
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
        """Use the reasoning model (DeepSeek R1) for complex analysis."""
        full_prompt = prompt
        if context:
            ctx_lines = "\n".join(f"- {k}: {v}" for k, v in context.items())
            full_prompt = f"{prompt}\n\nContext:\n{ctx_lines}"

        resp = await self.complete(LLMRequest(
            prompt=full_prompt,
            task_type="reasoning",
            reasoning=True,
            max_tokens=max_tokens,
            temperature=temperature,
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

    async def _ollama_complete(self, request: LLMRequest, model: str) -> LLMResponse:
        payload = {
            "model": model,
            "prompt": request.prompt,
            "system": request.system,
            "stream": False,
            "options": {
                "num_ctx": 4096,
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

    async def _openai_compat_complete(
        self,
        request: LLMRequest,
        model: str,
        base_url: str,
        api_key: str,
        backend_enum: LLMBackend,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Generic handler for OpenAI-compatible APIs (Groq, DeepSeek, OpenRouter)."""
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
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        log.info(f"{backend_enum.value}.response", tokens=tokens, model=model)

        return LLMResponse(
            content=content,
            backend=backend_enum,
            model=model,
            tokens_used=tokens,
        )

    async def _claude_complete(self, request: LLMRequest) -> LLMResponse:
        """Anthropic Claude API — development use only."""
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

        api_key = self._get_keychain_key("anthropic_api_key")
        if not api_key:
            raise RuntimeError(
                "anthropic_api_key not found in Keychain. "
                "Claude is for development only — add a free cloud API key instead."
            )

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

    async def _grok_complete(self, request: LLMRequest, model: str) -> LLMResponse:
        """xAI Grok via OpenAI-compatible REST endpoint."""
        api_key = self._get_keychain_key("xai_api_key")
        if not api_key:
            raise RuntimeError("xai_api_key not found in Keychain")

        model = request.model or model or "grok-4-fast-reasoning"
        return await self._openai_compat_complete(
            request, model,
            base_url="https://api.x.ai/v1",
            api_key=api_key,
            backend_enum=LLMBackend.GROK,
        )

    # ------------------------------------------------------------------
    # LangChain integration
    # ------------------------------------------------------------------

    def as_langchain_llm(
        self,
        model: str | None = None,
        reasoning: bool = False,
        task_type: str = "simple",
    ):
        """Return a LangChain-compatible LLM for use in LangGraph nodes.

        Prefers Ollama (local). Falls back to ChatOpenAI-compat for cloud backends
        when langchain-openai is installed.

        For async LangGraph nodes, use ``await router.complete(...)`` directly.
        """
        try:
            from langchain_ollama import OllamaLLM
            if reasoning or task_type == "reasoning":
                selected = model or self.REASONING_MODEL
            elif task_type == "coding":
                selected = model or self.settings.ollama_model_coding
            elif task_type == "creative":
                selected = model or self.settings.ollama_model_writing
            else:
                selected = model or self.STANDARD_MODEL

            return OllamaLLM(
                base_url=self.settings.ollama_base_url,
                model=selected,
            )
        except ImportError:
            raise RuntimeError(
                "langchain-ollama not installed. Run: pip install langchain-ollama"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_keychain_key(self, key_name: str) -> str | None:
        try:
            import keyring
            # Try both service names for compatibility with jarvis_v1
            for service in ("jarvis_v2", "personal_agent_hub"):
                val = keyring.get_password(service, key_name)
                if val:
                    return val
            return None
        except Exception:
            return None

    async def _log_usage(self, resp: LLMResponse, task_type: str) -> None:
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
    """Parse DeepSeek R1 <think>...</think> tags from any backend's response."""
    match = re.search(r"<think>(.*?)</think>", resp.content, re.DOTALL | re.IGNORECASE)
    if match:
        resp.reasoning = match.group(1).strip()
        resp.conclusion = resp.content[match.end():].strip()
    else:
        resp.reasoning = None
        resp.conclusion = resp.content
    return resp
