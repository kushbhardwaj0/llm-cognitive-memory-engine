import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    # LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    OLLAMA_TIMEOUT: float = 60.0

    # Memory Engine Settings
    WORKING_MEMORY_MAX_TOKENS: int = 2048
    EPISODIC_DB_PATH: str = "memory_episodic.db"
    SEMANTIC_GRAPH_PATH: str = "semantic_graph.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
