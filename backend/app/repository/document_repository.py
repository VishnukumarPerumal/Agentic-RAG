"""Repository for Document metadata and ChatHistory persistence."""

from sqlalchemy import desc
from sqlalchemy.orm import Session
from app.models.document_models import ChatHistory, Document


class DocumentRepository:
    """Data access layer for ingested documents and chat history."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_document(
        self,
        filename: str,
        file_type: str,
        file_size: int,
        chunk_count: int,
        metadata_json: dict | None = None,
        status: str = "COMPLETED",
    ) -> Document:
        doc = Document(
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            chunk_count=chunk_count,
            status=status,
            metadata_json=metadata_json or {},
        )
        self.db.add(doc)
        self.db.flush()
        return doc

    def list_documents(self, limit: int = 100) -> list[Document]:
        return (
            self.db.query(Document)
            .order_by(desc(Document.created_at))
            .limit(limit)
            .all()
        )

    def get_document(self, document_id: int) -> Document | None:
        return self.db.query(Document).filter(Document.id == document_id).first()

    def delete_document(self, document_id: int) -> bool:
        doc = self.get_document(document_id)
        if not doc:
            return False
        self.db.delete(doc)
        self.db.commit()
        return True

    def save_chat_history(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str,
        guardrail_passed: bool,
        sources_used: dict,
        agent_trace: list,
    ) -> ChatHistory:
        record = ChatHistory(
            session_id=session_id,
            user_query=user_query,
            assistant_response=assistant_response,
            guardrail_passed=guardrail_passed,
            sources_used=sources_used,
            agent_trace=agent_trace,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def list_chat_history(
        self, session_id: str | None = None, limit: int = 50
    ) -> list[ChatHistory]:
        q = self.db.query(ChatHistory)
        if session_id:
            q = q.filter(ChatHistory.session_id == session_id)
        return q.order_by(desc(ChatHistory.created_at)).limit(limit).all()
