from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentChunk:
    file_path: Path
    chunk_id: str
    text: str
    page_number: int | None = None
    line_offset: int = 0


def _extract_pdf_chunks(path: Path) -> list[DocumentChunk]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF extraction requires the 'pypdf' package. Install requirements.txt first."
        ) from exc

    reader = PdfReader(str(path))
    chunks: list[DocumentChunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        chunks.append(
            DocumentChunk(
                file_path=path,
                chunk_id=f"page_{page_number}",
                text=text,
                page_number=page_number,
            )
        )
    return chunks


def _extract_text_chunks(path: Path) -> list[DocumentChunk]:
    text = path.read_text(encoding="utf-8")
    return [DocumentChunk(file_path=path, chunk_id="full_text", text=text)]


def _extract_json_chunks(path: Path) -> list[DocumentChunk]:
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    pretty_text = json.dumps(parsed, ensure_ascii=False, indent=2)
    return [DocumentChunk(file_path=path, chunk_id="json", text=pretty_text)]


def _extract_csv_chunks(path: Path) -> list[DocumentChunk]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            rendered = ", ".join(f"{key}={value}" for key, value in row.items())
            lines.append(f"row={index}: {rendered}")
    return [
        DocumentChunk(file_path=path, chunk_id="csv", text="\n".join(lines))
    ]


def extract_document_chunks(path: str | Path) -> list[DocumentChunk]:
    document_path = Path(path)
    suffix = document_path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf_chunks(document_path)
    if suffix in {".txt", ".md"}:
        return _extract_text_chunks(document_path)
    if suffix == ".json":
        return _extract_json_chunks(document_path)
    if suffix == ".csv":
        return _extract_csv_chunks(document_path)

    raise ValueError(f"Unsupported file type: {document_path.suffix}")
