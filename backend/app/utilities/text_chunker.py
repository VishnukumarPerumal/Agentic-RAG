"""Text chunking utility built on LangChain RecursiveCharacterTextSplitter."""

from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utilities.config import settings


@dataclass
class TextChunk:
    chunk_index: int
    content: str
    metadata: dict


def split_text_into_chunks(
    text: str,
    source_filename: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    extra_metadata: dict | None = None,
) -> list[TextChunk]:
    """Split extracted document text into overlapping chunks with metadata."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    effective_size = chunk_size or settings.chunk_size
    effective_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_size,
        chunk_overlap=effective_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(cleaned)
    base_meta = extra_metadata.copy() if extra_metadata else {}

    chunks: list[TextChunk] = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_meta = {
            **base_meta,
            "source": source_filename,
            "chunk_index": idx,
            "total_chunks": len(raw_chunks),
            "char_length": len(chunk_text),
        }
        chunks.append(
            TextChunk(
                chunk_index=idx,
                content=chunk_text,
                metadata=chunk_meta,
            )
        )
    return chunks
