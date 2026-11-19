"""FastAPI routes for the LangGraph multi-agent RAG chat pipeline."""

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.schemas.chat_schemas import (
    ChatHistoryItemOut,
    ChatQueryRequest,
    ChatQueryResponse,
)
from app.repository.document_repository import DocumentRepository
from app.services.answer_agent import BASIC_RAG_SYSTEM_PROMPT
from app.services.rag_graph_service import RAGGraphService

router = APIRouter(prefix="/api/v1/chat", tags=["LangGraph Multi-Agent RAG Chat"])


@router.post("/query", response_model=ChatQueryResponse)
def execute_rag_chat_query(
    payload: ChatQueryRequest,
    db: Session = Depends(get_db),
) -> ChatQueryResponse:
    """
    Execute the LangGraph multi-agent RAG workflow:
    1. Guardrail Agent (safety, prompt injection, SQL injection, PII check)
    2. Data Retrieval Agent (pgvector DB chunks + PostgreSQL relational DB + MCP tools)
    3. Answer Generation Agent (basic RAG prompt synthesis)
    """
    graph_service = RAGGraphService(db)
    return graph_service.run_chat_query(payload)


@router.post("/stream")
def stream_rag_chat_query(
    payload: ChatQueryRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    Stream live LangGraph step execution events (Guardrail -> Data Retrieval -> Answer Generation)
    via Server-Sent Events (SSE) for real-time progress visualization in the chat UI.
    """
    graph_service = RAGGraphService(db)
    return StreamingResponse(
        graph_service.stream_chat_query(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history", response_model=list[ChatHistoryItemOut])
def get_chat_history(
    session_id: str | None = Query(default=None, description="Filter by conversation session_id"),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[ChatHistoryItemOut]:
    """Retrieve stored chat conversation history and agent traces."""
    repo = DocumentRepository(db)
    rows = repo.list_chat_history(session_id=session_id, limit=limit)
    return [ChatHistoryItemOut.model_validate(r) for r in rows]


@router.get("/graph-topology")
def get_langgraph_topology() -> dict:
    """Return metadata describing the LangGraph RAG multi-agent workflow and basic prompt."""
    return {
        "orchestrator": "LangGraph StateGraph",
        "nodes": [
            {
                "id": "guardrail_agent",
                "role": "Guardrail Agent",
                "description": "Validates query for prompt injection, destructive SQL, harmful content, and PII.",
            },
            {
                "id": "guardrail_blocked_node",
                "role": "Guardrail Rejection Handler",
                "description": "Short-circuits execution and returns a safe refusal when guardrail blocks a query.",
            },
            {
                "id": "data_retrieval_agent",
                "role": "Data Retrieval Agent",
                "description": "Retrieves context from pgvector DB, PostgreSQL relational tables, and MCP tools.",
                "sources": ["pgvector_db", "postgresql_db", "mcp_tools"],
            },
            {
                "id": "answer_generation_agent",
                "role": "Answer Generation Agent",
                "description": "Generates grounded response from retrieved context using basic RAG prompt.",
            },
        ],
        "edges": [
            {"from": "START", "to": "guardrail_agent", "type": "direct"},
            {
                "from": "guardrail_agent",
                "to": "data_retrieval_agent",
                "type": "conditional",
                "condition": "guardrail.allowed == True",
            },
            {
                "from": "guardrail_agent",
                "to": "guardrail_blocked_node",
                "type": "conditional",
                "condition": "guardrail.allowed == False",
            },
            {"from": "guardrail_blocked_node", "to": "END", "type": "direct"},
            {"from": "data_retrieval_agent", "to": "answer_generation_agent", "type": "direct"},
            {"from": "answer_generation_agent", "to": "END", "type": "direct"},
        ],
        "answer_agent_system_prompt": BASIC_RAG_SYSTEM_PROMPT,
    }
