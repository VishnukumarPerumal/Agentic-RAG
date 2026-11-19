"""Ingestion pipeline service for parsing files, splitting into chunks, embedding, and storing in pgvector."""

from fastapi import UploadFile
from sqlalchemy.orm import Session
from app.db.schemas.document_schemas import (
    DocumentOut,
    IngestResponse,
    RawTextIngestRequest,
    VectorChunkMatch,
)
from app.repository.document_repository import DocumentRepository
from app.repository.vector_repository import VectorRepository
from app.utilities.config import settings
from app.utilities.file_parser import parse_file_bytes
from app.utilities.llm_provider import generate_embedding, generate_embeddings_batch
from app.utilities.text_chunker import split_text_into_chunks


class IngestionService:
    """Handles document upload, chunking, embedding generation, and pgvector storage."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.doc_repo = DocumentRepository(db)
        self.vec_repo = VectorRepository(db)

    async def ingest_uploaded_file(
        self,
        file: UploadFile,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestResponse:
        filename = file.filename or "uploaded_document.txt"
        content_bytes = await file.read()
        if not content_bytes:
            raise ValueError(f"Uploaded file '{filename}' is empty.")

        extracted_text, file_meta = parse_file_bytes(filename, content_bytes)
        if not extracted_text.strip():
            raise ValueError(f"No readable text could be extracted from '{filename}'.")

        chunks = split_text_into_chunks(
            text=extracted_text,
            source_filename=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extra_metadata=file_meta,
        )
        if not chunks:
            raise ValueError(f"Failed to generate chunks from '{filename}'.")

        embeddings = generate_embeddings_batch([c.content for c in chunks])

        doc = self.doc_repo.create_document(
            filename=filename,
            file_type=file_meta.get("extension", "unknown"),
            file_size=len(content_bytes),
            chunk_count=len(chunks),
            metadata_json=file_meta,
            status="COMPLETED",
        )
        self.vec_repo.store_chunks(
            document_id=doc.id,
            chunks=chunks,
            embeddings=embeddings,
        )

        return IngestResponse(
            document_id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            chunks_stored=len(chunks),
            embedding_dimension=settings.embedding_dimension,
            status=doc.status,
            created_at=doc.created_at,
        )

    def ingest_raw_text(self, payload: RawTextIngestRequest) -> IngestResponse:
        chunks = split_text_into_chunks(
            text=payload.content,
            source_filename=payload.title,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            extra_metadata=payload.metadata,
        )
        if not chunks:
            raise ValueError("Provided text content produced zero chunks.")

        embeddings = generate_embeddings_batch([c.content for c in chunks])
        encoded_bytes = payload.content.encode("utf-8")

        doc = self.doc_repo.create_document(
            filename=payload.title,
            file_type=".txt",
            file_size=len(encoded_bytes),
            chunk_count=len(chunks),
            metadata_json={"source": "raw_text_api", **payload.metadata},
            status="COMPLETED",
        )
        self.vec_repo.store_chunks(
            document_id=doc.id,
            chunks=chunks,
            embeddings=embeddings,
        )

        return IngestResponse(
            document_id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            chunks_stored=len(chunks),
            embedding_dimension=settings.embedding_dimension,
            status=doc.status,
            created_at=doc.created_at,
        )

    def list_documents(self) -> list[DocumentOut]:
        docs = self.doc_repo.list_documents()
        return [DocumentOut.model_validate(d) for d in docs]

    def delete_document(self, document_id: int) -> bool:
        return self.doc_repo.delete_document(document_id)

    def search_vectors(
        self, query: str, top_k: int = 5, document_id: int | None = None
    ) -> list[VectorChunkMatch]:
        query_emb = generate_embedding(query)
        matches = self.vec_repo.similarity_search(
            query_embedding=query_emb,
            top_k=top_k,
            document_id=document_id,
        )
        return [VectorChunkMatch(**m) for m in matches]
