"""Application settings via pydantic-settings (reads from .env)."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    postgres_url: str = "postgresql://localhost/personal_agent_hub"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    redis_url: str = "redis://localhost:6379"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model_fast: str = "qwen2.5:0.5b"
    ollama_model_reason: str = "deepseek-r1:7b"

    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    confidence_execute: float = 0.90
    confidence_delegate: float = 0.60


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
