from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CategoryConfig:
    key: str
    label: str
    description: str
    terms: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    context_terms: list[str] = field(default_factory=list)
    field_hints: list[str] = field(default_factory=list)
    semantic_anchors: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    @property
    def lexical_terms(self) -> list[str]:
        values = self.terms + self.synonyms
        seen: set[str] = set()
        unique_values: list[str] = []
        for value in values:
            if value not in seen:
                unique_values.append(value)
                seen.add(value)
        return unique_values


@dataclass
class EngineConfig:
    language: str
    normalization: dict[str, Any]
    fuzzy: dict[str, Any]
    linguistics: dict[str, Any]
    semantic: dict[str, Any]
    scoring: dict[str, Any]
    categories: list[CategoryConfig]


@dataclass
class SentenceUnit:
    text: str
    normalized_text: str
    tokens: list[str]
    lemmas: list[str]
    roots: list[str]
    index: int
    start_char: int = 0
    end_char: int = 0


@dataclass
class Evidence:
    category_key: str
    category_label: str
    method: str
    score: float
    trigger: str
    sentence: str
    explanation: str
    context_hits: list[str] = field(default_factory=list)
    field_hits: list[str] = field(default_factory=list)
    matched_variant: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryResult:
    category_key: str
    category_label: str
    score: float
    decision: str
    evidences: list[Evidence] = field(default_factory=list)
    method_scores: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class AnalysisResult:
    text_id: str
    original_text: str
    normalized_text: str
    category_results: list[CategoryResult] = field(default_factory=list)
    decision_log: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
