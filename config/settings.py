"""Application settings via pydantic-settings (reads from .env).

All environment variables are documented here. The .env file at
/Users/TehFiestyGoat/Development/jarvis_v2/.env is loaded automatically.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL (shared with jarvis_v1 — same DB, no schema changes)
    postgres_url: str = "postgresql://localhost/personal_agent_hub"

    # Neo4j knowledge graph
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "FiestyGoatNeo4j2026!"

    # Redis coordination layer
    redis_url: str = "redis://localhost:6379"

    # MLX Local — Mac Studio M2 Max (Apple Metal inference, OpenAI-compatible API)
    # Note: vLLM has no MPS backend — use mlx_lm.server instead (installed in venv)
    # Start: venv/bin/python3.13 -m mlx_lm server --model mlx-community/Qwen3-4B-4bit --port 8000 --host 0.0.0.0
    # On new rig (2x RTX 5060 Ti): VLLM_USE_V1=0 vllm serve Qwen/Qwen3-4B --port 8000 --tensor-parallel-size 2
    # Port 8000 — LLM inference. The Codex backend moved to port 8001 (native, no Docker)
    # Models served here use HuggingFace model IDs (mlx-community/ prefix for 4-bit quantized)
    vllm_local_base_url: str = "http://localhost:8000/v1"

    # vLLM Remote — AWS g5.xlarge GPU instance (on-demand, stop when not needed)
    # Models: Qwen3.5-397B-A17B, Qwen3-30B-A3B, DeepSeek-R1-14B, Devstral-24B
    # Set via: keyring.set_password('jarvis_v2', 'vllm_base_url', 'http://<ip>:8000/v1')
    # Stop instance to save ~$730/mo, start when heavy inference needed
    vllm_base_url: str = ""   # populated from Keychain at runtime

    # Ollama — legacy fallback only (used if vLLM local is not running)
    ollama_base_url: str = "http://localhost:11434"

    # Free cloud LLM APIs (add keys via: keyring.set_password('jarvis_v2', '<key_name>', '<value>'))
    # Groq — https://console.groq.com (free tier, no CC, 70B inference)
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # DeepSeek — https://platform.deepseek.com (V3.2 + R1 reasoning, $0.07/M tokens)
    deepseek_base_url: str = "https://api.deepseek.com"
    # OpenRouter — https://openrouter.ai (free model pool: Qwen3.5, Llama 3.1, etc.)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # External LLM API keys (stored in Keychain; env vars as override/CI fallback)
    anthropic_api_key: str = ""
    xai_api_key: str = ""

    # Alpaca paper trading
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # Confidence gate thresholds
    confidence_execute: float = 0.90
    confidence_delegate: float = 0.60

    # FastAPI server — FIXED at 8506, do not change
    api_port: int = 8506

    # Application metadata
    app_name: str = "jarvis-v2"
    app_version: str = "0.1.0"
    debug: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
