"""Application configuration management using environment variables and .env file."""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "Agentic RAG Backend")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"

    # PostgreSQL + pgvector settings (New PG server on port 5433)
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres@localhost:5433/agentic_rag",
    )
    pg_host: str = os.getenv("PG_HOST", "localhost")
    pg_port: int = int(os.getenv("PG_PORT", "5433"))
    pg_user: str = os.getenv("PG_USER", "postgres")
    pg_password: str = os.getenv("PG_PASSWORD", "")
    pg_database: str = os.getenv("PG_DATABASE", "agentic_rag")

    # Vector & Chunking settings
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))

    # LLM settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

    cors_origins: list[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
            ).split(",")
            if origin.strip()
        ]
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
