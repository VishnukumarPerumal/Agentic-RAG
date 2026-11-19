"""FastAPI main application entry point for the Agentic RAG Backend."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.db.database import engine, init_db
from app.routes import chat_router, data_router, ingestion_router
from app.utilities.config import settings


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize PostgreSQL tables, pgvector extension, HNSW index, and sample data on startup."""
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "LangGraph Multi-Agent RAG System featuring a Guardrail Agent, "
        "Data Retrieval Agent (MCP Tools, pgvector DB, PostgreSQL relational data), "
        "Answer Generation Agent, and Document Ingestion Pipeline."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion_router)
app.include_router(chat_router)
app.include_router(data_router)


@app.get("/", tags=["Health & System"])
def root() -> dict:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs_url": "/docs",
        "agents": [
            "guardrail_agent",
            "data_retrieval_agent (mcp_tools + pgvector_db + postgresql_db)",
            "answer_generation_agent",
        ],
    }


@app.get("/health", tags=["Health & System"])
def health_check() -> dict:
    """Verify PostgreSQL connection and pgvector extension version."""
    with engine.connect() as conn:
        pg_version = conn.execute(text("SHOW server_version;")).scalar()
        pgvector_version = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        ).scalar()

    return {
        "status": "healthy",
        "postgres_host": settings.pg_host,
        "postgres_port": settings.pg_port,
        "postgres_database": settings.pg_database,
        "postgres_version": pg_version,
        "pgvector_version": pgvector_version,
        "embedding_dimension": settings.embedding_dimension,
    }
