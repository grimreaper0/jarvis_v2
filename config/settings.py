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
    ollama_model_fast: str = "qwen2.5:0.5b"
    ollama_model_reason: str = "deepseek-r1:7b"

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
