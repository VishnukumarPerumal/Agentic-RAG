"""FastAPI routes for the document ingestion pipeline and pgvector similarity search."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.schemas.document_schemas import (
    DocumentOut,
    IngestResponse,
    RawTextIngestRequest,
    VectorChunkMatch,
    VectorSearchRequest,
)
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion Pipeline & pgvector"])


@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_and_ingest_file(
    file: UploadFile = File(..., description="File to upload (.pdf, .txt, .md, .csv, .json, .html)"),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
    db: Session = Depends(get_db),
) -> IngestResponse:
    """
    Upload a file, extract text, split into chunks, generate embeddings,
    and store the chunks in PostgreSQL pgvector.
    """
    service = IngestionService(db)
    try:
        return await service.ingest_uploaded_file(
            file=file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/text", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_raw_text(
    payload: RawTextIngestRequest,
    db: Session = Depends(get_db),
) -> IngestResponse:
    """Chunk, embed, and store raw text directly in PostgreSQL pgvector."""
    service = IngestionService(db)
    try:
        return service.ingest_raw_text(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/documents", response_model=list[DocumentOut])
def list_ingested_documents(db: Session = Depends(get_db)) -> list[DocumentOut]:
    """List all uploaded documents and their stored vector chunk counts."""
    service = IngestionService(db)
    return service.list_documents()


@router.delete("/documents/{document_id}")
def delete_ingested_document(
    document_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """Delete a document and cascade-delete all of its pgvector chunks."""
    service = IngestionService(db)
    deleted = service.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id={document_id} not found.",
        )
    return {"deleted": True, "document_id": document_id}


@router.post("/search", response_model=list[VectorChunkMatch])
def search_vector_chunks(
    payload: VectorSearchRequest,
    db: Session = Depends(get_db),
) -> list[VectorChunkMatch]:
    """Perform direct cosine similarity search (<=>) against pgvector document chunks."""
    service = IngestionService(db)
    return service.search_vectors(
        query=payload.query,
        top_k=payload.top_k,
        document_id=payload.document_id,
    )
