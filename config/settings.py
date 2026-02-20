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

    # Ollama (local, $0 cost)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model_fast: str = "qwen2.5:0.5b"        # 397MB  — simple/fast tasks
    ollama_model_reason: str = "deepseek-r1:7b"    # 4.7GB  — reasoning/trading
    ollama_model_coding: str = "qwen2.5-coder:1.5b"  # 986MB  — coding tasks
    ollama_model_general: str = "llama3.2:3b"      # 2.0GB  — general tasks
    ollama_model_writing: str = "mistral:7b"       # 4.1GB  — creative/writing
    ollama_model_longctx: str = "qwen2.5:14b"      # 8.5GB  — long-context (optional)

    # Free cloud LLM APIs (add keys via: python3 -c "import keyring; keyring.set_password('jarvis_v2', '<key_name>', '<value>')")
    # Groq — https://console.groq.com (free tier, no CC, 70B inference)
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # DeepSeek — https://platform.deepseek.com (V3.2 + R1 reasoning, $0.07/M tokens)
    deepseek_base_url: str = "https://api.deepseek.com"
    # OpenRouter — https://openrouter.ai (free model pool: Qwen3-235B, Llama 3.1, etc.)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # vLLM — AWS g5.xlarge GPU instance (Qwen3-30B, DeepSeek-R1-14B, Devstral-24B)
    # Set via: python3 -c "import keyring; keyring.set_password('jarvis_v2', 'vllm_base_url', 'http://<ip>:8000/v1')"
    # Stop instance to save ~$730/mo, start when heavy inference needed
    vllm_base_url: str = ""   # populated from Keychain at runtime

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

    # FastAPI server
    api_port: int = 8504

    # Application metadata
    app_name: str = "jarvis-v2"
    app_version: str = "0.1.0"
    debug: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
