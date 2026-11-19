"""Pydantic schemas for the LangGraph multi-agent RAG chat pipeline."""

from datetime import datetime
from pydantic import BaseModel, Field


class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="User question or prompt")
    session_id: str = Field(default="default-session", description="Conversation session identifier")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of vector chunks to retrieve")
    use_vector_db: bool = Field(default=True, description="Enable pgvector chunk retrieval")
    use_postgres_db: bool = Field(default=True, description="Enable PostgreSQL relational data retrieval")
    use_mcp_tools: bool = Field(default=True, description="Enable MCP tool execution")


class GuardrailEvaluationOut(BaseModel):
    allowed: bool
    category: str
    risk_score: float
    reason: str
    sanitized_query: str


class RetrievedContextOut(BaseModel):
    vector_chunks: list[dict] = Field(default_factory=list)
    postgres_records: list[dict] = Field(default_factory=list)
    mcp_tool_outputs: list[dict] = Field(default_factory=list)


class AgentStepTraceOut(BaseModel):
    agent: str
    status: str
    summary: str
    details: dict = Field(default_factory=dict)


class ChatQueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    guardrail: GuardrailEvaluationOut
    retrieved_context: RetrievedContextOut
    agent_trace: list[AgentStepTraceOut]
    llm_provider: str
    created_at: datetime


class ChatHistoryItemOut(BaseModel):
    id: int
    session_id: str
    user_query: str
    assistant_response: str
    guardrail_passed: bool
    sources_used: dict
    agent_trace: list
    created_at: datetime

    model_config = {"from_attributes": True}
