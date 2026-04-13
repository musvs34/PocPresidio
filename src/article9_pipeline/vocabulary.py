from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from presidio_analyzer import Pattern, PatternRecognizer


@dataclass
class VocabularyConfig:
    terms_by_entity: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    contexts_by_entity: dict[str, list[str]] = field(default_factory=dict)
    regex_by_entity: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    scopes_by_entity: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    @property
    def entities(self) -> list[str]:
        return sorted(
            set(self.terms_by_entity)
            | set(self.contexts_by_entity)
            | set(self.regex_by_entity)
        )


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_vocabulary(path: str | Path) -> VocabularyConfig:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {csv_path}")

    terms: dict[str, list[dict[str, Any]]] = defaultdict(list)
    contexts: dict[str, list[str]] = defaultdict(list)
    regexes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scopes: dict[str, list[dict[str, str]]] = defaultdict(list)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            if not _parse_bool(row.get("enabled", "true")):
                continue

            entity_name = (row.get("entity_name") or "").strip()
            entry_type = (row.get("entry_type") or "").strip().lower()
            value = (row.get("value") or "").strip()
            language = (row.get("language") or "fr").strip()
            score = float(row.get("score") or 0.5)

            if not entity_name or not entry_type or not value:
                continue

            scopes[entity_name].append(
                {
                    "document_scope": (row.get("document_scope") or "").strip(),
                    "section_scope": (row.get("section_scope") or "").strip(),
                    "source": (row.get("source") or "").strip(),
                    "notes": (row.get("notes") or "").strip(),
                }
            )

            if entry_type == "term":
                terms[entity_name].append(
                    {"value": value, "score": score, "language": language}
                )
            elif entry_type == "context":
                contexts[entity_name].append(value)
            elif entry_type == "regex":
                regexes[entity_name].append(
                    {"value": value, "score": score, "language": language}
                )

    return VocabularyConfig(
        terms_by_entity=dict(terms),
        contexts_by_entity={k: sorted(set(v)) for k, v in contexts.items()},
        regex_by_entity=dict(regexes),
        scopes_by_entity=dict(scopes),
    )


def build_recognizers(
    vocabulary: VocabularyConfig,
    *,
    language: str = "fr",
) -> tuple[list[PatternRecognizer], dict[str, list[str]]]:
    recognizers: list[PatternRecognizer] = []
    context_map: dict[str, list[str]] = {}

    for entity in vocabulary.entities:
        term_entries = [
            entry for entry in vocabulary.terms_by_entity.get(entity, [])
            if entry["language"] == language
        ]
        regex_entries = [
            entry for entry in vocabulary.regex_by_entity.get(entity, [])
            if entry["language"] == language
        ]
        context_entries = vocabulary.contexts_by_entity.get(entity, [])

        patterns = [
            Pattern(
                name=f"{entity}_regex_{index}",
                regex=entry["value"],
                score=entry["score"],
            )
            for index, entry in enumerate(regex_entries, start=1)
        ]
        deny_list = [entry["value"] for entry in term_entries]
        deny_list_score = max((entry["score"] for entry in term_entries), default=0.5)

        if not patterns and not deny_list:
            continue

        recognizer = PatternRecognizer(
            supported_entity=entity,
            name=f"{entity}_recognizer",
            supported_language=language,
            patterns=patterns or None,
            deny_list=deny_list or None,
            deny_list_score=deny_list_score,
            context=context_entries or None,
        )
        recognizers.append(recognizer)
        context_map[entity] = context_entries

    return recognizers, context_map
