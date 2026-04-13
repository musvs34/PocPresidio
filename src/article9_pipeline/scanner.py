from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .documents import DocumentChunk, extract_document_chunks
from .vocabulary import build_recognizers, load_vocabulary

CONTEXT_WINDOW = 120
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".json", ".csv"}
RISK_PRIORITY = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _compute_line_number(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


def _context_hits(text: str, start: int, end: int, context_words: list[str]) -> list[str]:
    window_start = max(0, start - CONTEXT_WINDOW)
    window_end = min(len(text), end + CONTEXT_WINDOW)
    surrounding_text = text[window_start:window_end].lower()
    hits = [word for word in context_words if word.lower() in surrounding_text]
    return sorted(set(hits))


def _adjust_score(raw_score: float, hits: list[str]) -> float:
    if not hits:
        return round(raw_score, 4)
    boosted = min(1.0, raw_score + (0.05 * len(hits)))
    return round(boosted, 4)


def _risk_level(entity_name: str, score: float) -> str:
    if entity_name in {"HEALTH_ARTICLE9", "GENETIC_OR_BIOMETRIC_HINT"}:
        return "high"
    if entity_name == "POLITICAL_RELIGIOUS_ORIENTATION_HINT" or score >= 0.9:
        return "medium"
    return "low"


def _max_risk_level(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "none"
    return max(findings, key=lambda row: RISK_PRIORITY[row["risk_level"]])["risk_level"]


def _deduplicate_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for finding in findings:
        key = (
            finding["file_path"],
            finding["chunk_id"],
            finding["entity_name"],
            finding["start"],
            finding["end"],
            finding["matched_text"].lower(),
        )
        current = best_by_key.get(key)
        if current is None or finding["adjusted_score"] > current["adjusted_score"]:
            best_by_key[key] = finding
    return sorted(
        best_by_key.values(),
        key=lambda row: (row["file_name"], row["page_number"] or 0, row["start"]),
    )


def _scan_chunk(
    chunk: DocumentChunk,
    recognizers,
    context_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for recognizer in recognizers:
        results = recognizer.analyze(
            text=chunk.text,
            entities=[recognizer.supported_entities[0]],
            nlp_artifacts=None,
        )
        for result in results:
            entity_name = result.entity_type
            matched_text = chunk.text[result.start:result.end]
            context_hits = _context_hits(
                chunk.text,
                result.start,
                result.end,
                context_map.get(entity_name, []),
            )
            adjusted_score = _adjust_score(result.score, context_hits)
            findings.append(
                {
                    "file_name": chunk.file_path.name,
                    "file_path": str(chunk.file_path),
                    "file_type": chunk.file_path.suffix.lower(),
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "line_number": _compute_line_number(chunk.text, result.start),
                    "entity_name": entity_name,
                    "matched_text": matched_text,
                    "start": result.start,
                    "end": result.end,
                    "raw_score": round(result.score, 4),
                    "adjusted_score": adjusted_score,
                    "context_hits": "|".join(context_hits),
                    "risk_level": _risk_level(entity_name, adjusted_score),
                }
            )
    return findings


def scan_documents(
    input_dir: str | Path,
    vocabulary_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_path = Path(input_dir)
    vocabulary = load_vocabulary(vocabulary_path)
    recognizers, context_map = build_recognizers(vocabulary)

    all_findings: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for file_path in sorted(input_path.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            chunks = extract_document_chunks(file_path)
            file_findings: list[dict[str, Any]] = []
            for chunk in chunks:
                file_findings.extend(_scan_chunk(chunk, recognizers, context_map))
            deduped = _deduplicate_findings(file_findings)
            all_findings.extend(deduped)

            max_score = max((row["adjusted_score"] for row in deduped), default=0.0)
            summary_rows.append(
                {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_path.suffix.lower(),
                    "status": "ok" if chunks else "empty",
                    "chunk_count": len(chunks),
                    "finding_count": len(deduped),
                    "max_score": round(max_score, 4),
                    "max_risk_level": _max_risk_level(deduped),
                    "error_message": "",
                }
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_path.suffix.lower(),
                    "status": "error",
                    "chunk_count": 0,
                    "finding_count": 0,
                    "max_score": 0.0,
                    "max_risk_level": "none",
                    "error_message": str(exc),
                }
            )

    return all_findings, summary_rows


def write_outputs(
    findings: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    findings_csv = output_path / "article9_findings.csv"
    findings_json = output_path / "article9_findings.json"
    summary_csv = output_path / "article9_summary.csv"

    finding_fields = [
        "file_name",
        "file_path",
        "file_type",
        "chunk_id",
        "page_number",
        "line_number",
        "entity_name",
        "matched_text",
        "start",
        "end",
        "raw_score",
        "adjusted_score",
        "context_hits",
        "risk_level",
    ]
    summary_fields = [
        "file_name",
        "file_path",
        "file_type",
        "status",
        "chunk_count",
        "finding_count",
        "max_score",
        "max_risk_level",
        "error_message",
    ]

    with findings_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=finding_fields)
        writer.writeheader()
        writer.writerows(findings)

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    with findings_json.open("w", encoding="utf-8") as handle:
        json.dump(findings, handle, ensure_ascii=False, indent=2)

    return {
        "findings_csv": findings_csv,
        "findings_json": findings_json,
        "summary_csv": summary_csv,
    }
