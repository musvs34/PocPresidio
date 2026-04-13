from __future__ import annotations

import csv
import json
from pathlib import Path

from .documents import extract_document_chunks
from .engine import Article9Engine
from .models import AnalysisResult, CategoryResult, Evidence


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".json", ".csv"}
DECISION_PRIORITY = {"clear": 0, "review": 1, "alert": 2}


def analyze_directory(
    engine: Article9Engine,
    input_dir: str | Path,
) -> list[AnalysisResult]:
    input_path = Path(input_dir)
    analyses: list[AnalysisResult] = []

    for file_path in sorted(input_path.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        chunks = extract_document_chunks(file_path)
        for chunk in chunks:
            analyses.append(
                engine.analyze_text(
                    chunk.text,
                    text_id=f"{file_path.name}:{chunk.chunk_id}",
                    metadata={
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "file_type": file_path.suffix.lower(),
                        "chunk_id": chunk.chunk_id,
                        "page_number": chunk.page_number,
                    },
                )
            )
    return analyses


def _finding_row(result: AnalysisResult, category: CategoryResult, evidence: Evidence) -> dict:
    metadata = result.metadata
    return {
        "file_name": metadata.get("file_name", result.text_id),
        "file_path": metadata.get("file_path", result.text_id),
        "file_type": metadata.get("file_type", ""),
        "chunk_id": metadata.get("chunk_id", ""),
        "page_number": metadata.get("page_number"),
        "line_number": None,
        "entity_name": category.category_key,
        "category_label": category.category_label,
        "method": evidence.method,
        "matched_text": evidence.trigger,
        "matched_variant": evidence.matched_variant,
        "raw_score": evidence.score,
        "adjusted_score": evidence.score,
        "category_score": category.score,
        "decision": category.decision,
        "context_hits": "|".join(evidence.context_hits),
        "field_hits": "|".join(evidence.field_hits),
        "excerpt": evidence.sentence,
        "explanation": evidence.explanation,
    }


def flatten_findings(analyses: list[AnalysisResult]) -> list[dict]:
    rows: list[dict] = []
    for result in analyses:
        for category in result.category_results:
            if category.decision == "clear":
                continue
            for evidence in category.evidences:
                rows.append(_finding_row(result, category, evidence))
    return rows


def build_summary(analyses: list[AnalysisResult]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for result in analyses:
        metadata = result.metadata
        file_path = metadata.get("file_path", result.text_id)
        bucket = grouped.setdefault(
            file_path,
            {
                "file_name": metadata.get("file_name", result.text_id),
                "file_path": file_path,
                "file_type": metadata.get("file_type", ""),
                "status": "ok",
                "chunk_count": 0,
                "finding_count": 0,
                "max_score": 0.0,
                "max_risk_level": "clear",
                "top_categories": [],
                "error_message": "",
            },
        )
        bucket["chunk_count"] += 1
        document_categories = [
            category
            for category in result.category_results
            if category.decision in {"review", "alert"}
        ]
        bucket["finding_count"] += sum(len(category.evidences) for category in document_categories)
        if document_categories:
            bucket["max_score"] = max(
                bucket["max_score"],
                max(category.score for category in document_categories),
            )
            current_max = max(
                document_categories,
                key=lambda item: DECISION_PRIORITY[item.decision],
            )
            if DECISION_PRIORITY[current_max.decision] > DECISION_PRIORITY[bucket["max_risk_level"]]:
                bucket["max_risk_level"] = current_max.decision
            bucket["top_categories"].extend(
                f"{category.category_key}:{category.score:.2f}" for category in document_categories
            )

    summary_rows: list[dict] = []
    for bucket in grouped.values():
        top_categories = sorted(set(bucket["top_categories"]))
        summary_rows.append(
            {
                "file_name": bucket["file_name"],
                "file_path": bucket["file_path"],
                "file_type": bucket["file_type"],
                "status": bucket["status"],
                "chunk_count": bucket["chunk_count"],
                "finding_count": bucket["finding_count"],
                "max_score": round(bucket["max_score"], 4),
                "max_risk_level": bucket["max_risk_level"],
                "top_categories": "|".join(top_categories),
                "error_message": bucket["error_message"],
            }
        )
    return sorted(summary_rows, key=lambda row: row["file_name"])


def write_reports(
    analyses: list[AnalysisResult],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    findings_rows = flatten_findings(analyses)
    summary_rows = build_summary(analyses)
    findings_csv = output_path / "article9_findings.csv"
    findings_json = output_path / "article9_findings.json"
    summary_csv = output_path / "article9_summary.csv"
    audit_json = output_path / "article9_audit_log.json"

    finding_fields = [
        "file_name",
        "file_path",
        "file_type",
        "chunk_id",
        "page_number",
        "line_number",
        "entity_name",
        "category_label",
        "method",
        "matched_text",
        "matched_variant",
        "raw_score",
        "adjusted_score",
        "category_score",
        "decision",
        "context_hits",
        "field_hits",
        "excerpt",
        "explanation",
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
        "top_categories",
        "error_message",
    ]

    with findings_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=finding_fields)
        writer.writeheader()
        writer.writerows(findings_rows)

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    with findings_json.open("w", encoding="utf-8") as handle:
        json.dump(findings_rows, handle, ensure_ascii=False, indent=2)

    with audit_json.open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "text_id": result.text_id,
                    "metadata": result.metadata,
                    "decision_log": result.decision_log,
                    "categories": [
                        {
                            "category_key": category.category_key,
                            "category_label": category.category_label,
                            "score": category.score,
                            "decision": category.decision,
                            "method_scores": category.method_scores,
                            "explanation": category.explanation,
                            "evidences": [
                                {
                                    "method": evidence.method,
                                    "score": evidence.score,
                                    "trigger": evidence.trigger,
                                    "matched_variant": evidence.matched_variant,
                                    "sentence": evidence.sentence,
                                    "context_hits": evidence.context_hits,
                                    "field_hits": evidence.field_hits,
                                    "explanation": evidence.explanation,
                                    "metadata": evidence.metadata,
                                }
                                for evidence in category.evidences
                            ],
                        }
                        for category in result.category_results
                    ],
                }
                for result in analyses
            ],
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "findings_csv": findings_csv,
        "findings_json": findings_json,
        "summary_csv": summary_csv,
        "audit_json": audit_json,
    }
