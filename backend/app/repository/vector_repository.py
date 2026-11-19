"""Repository for storing and searching vector chunks in PostgreSQL pgvector."""

from sqlalchemy.orm import Session
from app.models.document_models import Document, DocumentChunk
from app.utilities.text_chunker import TextChunk


class VectorRepository:
    """Data access layer for pgvector embeddings and semantic similarity search."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def store_chunks(
        self,
        document_id: int,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> list[DocumentChunk]:
        db_chunks: list[DocumentChunk] = []
        for chunk, emb in zip(chunks, embeddings, strict=True):
            row = DocumentChunk(
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding=emb,
                metadata_json=chunk.metadata,
            )
            db_chunks.append(row)
        self.db.add_all(db_chunks)
        self.db.commit()
        return db_chunks

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: int | None = None,
    ) -> list[dict]:
        """
        Execute pgvector cosine distance search (<=>) and return ranked chunks
        with cosine similarity score in [0, 1].
        """
        distance_expr = DocumentChunk.embedding.cosine_distance(query_embedding).label(
            "cosine_distance"
        )

        stmt = (
            self.db.query(DocumentChunk, Document.filename, distance_expr)
            .join(Document, Document.id == DocumentChunk.document_id)
        )
        if document_id is not None:
            stmt = stmt.filter(DocumentChunk.document_id == document_id)

        rows = stmt.order_by(distance_expr.asc()).limit(top_k).all()

        results: list[dict] = []
        for chunk_obj, filename, distance in rows:
            dist_val = float(distance) if distance is not None else 1.0
            similarity = max(0.0, min(1.0, round(1.0 - dist_val, 4)))
            results.append(
                {
                    "chunk_id": chunk_obj.id,
                    "document_id": chunk_obj.document_id,
                    "filename": filename,
                    "chunk_index": chunk_obj.chunk_index,
                    "content": chunk_obj.content,
                    "similarity_score": similarity,
                    "metadata": chunk_obj.metadata_json or {},
                }
            )
        return results

    def count_chunks(self) -> int:
        return self.db.query(DocumentChunk).count()
