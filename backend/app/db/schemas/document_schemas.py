"""Pydantic schemas for document ingestion and vector search."""

from datetime import datetime
from pydantic import BaseModel, Field


class RawTextIngestRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Document title or filename")
    content: str = Field(..., min_length=1, description="Raw text content to chunk and embed")
    chunk_size: int | None = Field(default=None, ge=100, le=5000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)
    metadata: dict = Field(default_factory=dict)


class DocumentOut(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    status: str
    metadata_json: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class IngestResponse(BaseModel):
    document_id: int
    filename: str
    file_type: str
    file_size: int
    chunks_stored: int
    embedding_dimension: int
    status: str
    created_at: datetime


class VectorSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)
    document_id: int | None = Field(default=None)


class VectorChunkMatch(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    content: str
    similarity_score: float
    metadata: dict
