"""File parser utility for extracting text from PDF, TXT, Markdown, CSV, JSON, and HTML uploads."""

import csv
import io
import json
import re
from pathlib import Path
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json", ".html", ".htm"}


def parse_file_bytes(filename: str, content_bytes: bytes) -> tuple[str, dict]:
    """
    Extract clean plain text and metadata from uploaded file bytes.
    Returns (extracted_text, file_metadata).
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    metadata: dict = {
        "filename": filename,
        "extension": ext,
        "size_bytes": len(content_bytes),
    }

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(content_bytes))
        pages_text: list[str] = []
        for idx, page in enumerate(reader.pages):
            page_txt = (page.extract_text() or "").strip()
            if page_txt:
                pages_text.append(f"[Page {idx + 1}]\n{page_txt}")
        metadata["page_count"] = len(reader.pages)
        return "\n\n".join(pages_text), metadata

    decoded = content_bytes.decode("utf-8", errors="replace")

    if ext in {".txt", ".md"}:
        return decoded.strip(), metadata

    if ext == ".csv":
        reader = csv.reader(io.StringIO(decoded))
        rows = list(reader)
        if not rows:
            return "", metadata
        headers = rows[0]
        formatted_rows: list[str] = []
        for idx, row in enumerate(rows[1:], start=1):
            pairs = [
                f"{headers[i] if i < len(headers) else f'col_{i}'}: {val}"
                for i, val in enumerate(row)
            ]
            formatted_rows.append(f"Row {idx}: " + " | ".join(pairs))
        metadata["row_count"] = max(0, len(rows) - 1)
        metadata["columns"] = headers
        return "\n".join(formatted_rows), metadata

    if ext == ".json":
        try:
            parsed = json.loads(decoded)
            pretty_json = json.dumps(parsed, indent=2, ensure_ascii=False)
            return pretty_json, metadata
        except json.JSONDecodeError:
            return decoded.strip(), metadata

    if ext in {".html", ".htm"}:
        text = re.sub(r"<script[\s\S]*?</script>", " ", decoded, flags=re.IGNORECASE)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text, metadata

    return decoded.strip(), metadata
