"""LLMRouter — Ollama-first with Claude/Grok fallback."""
import structlog
from enum import Enum
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


class LLMResponse(BaseModel):
    content: str
    backend: LLMBackend
    model: str
    tokens_used: int = 0
    reasoning_tokens: int = 0


class LLMRouter:
    """Routes LLM calls to the appropriate backend.

    Priority: Ollama (local, $0) → Claude (fallback) → Grok (fallback)
    Reasoning tasks always route to deepseek-r1:7b via Ollama.
    """

    def __init__(self, settings=None):
        from config.settings import get_settings
        self.settings = settings or get_settings()
        self._ollama_client = None
        self._claude_client = None

    def _get_ollama_client(self):
        if self._ollama_client is None:
            from langchain_ollama import OllamaLLM
            self._ollama_client = OllamaLLM(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model_fast,
            )
        return self._ollama_client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model
        if request.reasoning or request.backend == LLMBackend.OLLAMA:
            model = model or (
                self.settings.ollama_model_reason
                if request.reasoning
                else self.settings.ollama_model_fast
            )
            return await self._ollama_complete(request, model)

        if request.backend == LLMBackend.CLAUDE:
            return await self._claude_complete(request)

        raise ValueError(f"Unsupported backend: {request.backend}")

    async def _ollama_complete(self, request: LLMRequest, model: str) -> LLMResponse:
        import httpx

        payload = {
            "model": model,
            "prompt": request.prompt,
            "system": request.system,
            "options": {"temperature": request.temperature, "num_predict": request.max_tokens},
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120) as client:
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
        raise NotImplementedError("Claude backend not yet configured — add anthropic_api_key")
