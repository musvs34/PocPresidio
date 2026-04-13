from __future__ import annotations

from typing import Iterable

from .models import CategoryConfig, Evidence, SentenceUnit
from .normalization import normalize_text, root_form, tokenize_normalized


def _unique_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value and value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def _collect_hits(sentence_text: str, candidates: list[str]) -> list[str]:
    return _unique_preserve(
        candidate
        for candidate in candidates
        if normalize_text(candidate) in sentence_text
    )


def _base_score(base: float, context_hits: list[str], field_hits: list[str]) -> float:
    boosted = base + (0.05 * len(context_hits)) + (0.04 * len(field_hits))
    return round(min(1.0, boosted), 4)


class LexicalDetector:
    method_name = "lexical"

    def detect(self, sentence: SentenceUnit, category: CategoryConfig) -> list[Evidence]:
        findings: list[Evidence] = []
        context_hits = _collect_hits(sentence.normalized_text, category.context_terms)
        field_hits = _collect_hits(sentence.normalized_text, category.field_hints)
        normalized_sentence = f" {sentence.normalized_text} "

        for term in category.lexical_terms:
            normalized_term = normalize_text(term)
            if not normalized_term:
                continue
            if f" {normalized_term} " in normalized_sentence or normalized_term in sentence.normalized_text:
                findings.append(
                    Evidence(
                        category_key=category.key,
                        category_label=category.label,
                        method=self.method_name,
                        score=_base_score(0.62, context_hits, field_hits),
                        trigger=term,
                        matched_variant=normalized_term,
                        sentence=sentence.text,
                        explanation=(
                            f"Correspondance lexicale exacte du terme '{term}' "
                            f"dans la phrase."
                        ),
                        context_hits=context_hits,
                        field_hits=field_hits,
                        metadata={"sentence_index": sentence.index},
                    )
                )
        return findings


class LinguisticDetector:
    method_name = "linguistic"

    def detect(self, sentence: SentenceUnit, category: CategoryConfig) -> list[Evidence]:
        findings: list[Evidence] = []
        context_hits = _collect_hits(sentence.normalized_text, category.context_terms)
        field_hits = _collect_hits(sentence.normalized_text, category.field_hints)
        sentence_roots = [root for root in sentence.roots if root]

        for term in category.lexical_terms:
            normalized_term = normalize_text(term)
            term_tokens = tokenize_normalized(normalized_term)
            if not term_tokens:
                continue
            if normalized_term in sentence.normalized_text:
                continue

            term_roots = [root_form(token) for token in term_tokens if token]
            if not term_roots:
                continue

            overlap = [root for root in term_roots if root in sentence_roots]
            if len(overlap) == len(term_roots):
                findings.append(
                    Evidence(
                        category_key=category.key,
                        category_label=category.label,
                        method=self.method_name,
                        score=_base_score(0.52, context_hits, field_hits),
                        trigger=term,
                        matched_variant=" ".join(overlap),
                        sentence=sentence.text,
                        explanation=(
                            f"Correspondance linguistique par racines/lemmes pour '{term}' "
                            f"avec les racines detectees {overlap}."
                        ),
                        context_hits=context_hits,
                        field_hits=field_hits,
                        metadata={"sentence_index": sentence.index, "overlap": overlap},
                    )
                )
        return findings


class FuzzyDetector:
    method_name = "fuzzy"

    def __init__(self, threshold: int = 88, ngram_delta: int = 1):
        self.threshold = threshold
        self.ngram_delta = ngram_delta
        try:
            from rapidfuzz import fuzz

            self._fuzz = fuzz
            self.available = True
        except Exception:
            self._fuzz = None
            self.available = False

    def detect(self, sentence: SentenceUnit, category: CategoryConfig) -> list[Evidence]:
        if not self.available:
            return []

        findings: list[Evidence] = []
        context_hits = _collect_hits(sentence.normalized_text, category.context_terms)
        field_hits = _collect_hits(sentence.normalized_text, category.field_hints)
        sentence_tokens = tokenize_normalized(sentence.normalized_text)

        for term in category.lexical_terms:
            normalized_term = normalize_text(term)
            if not normalized_term or normalized_term in sentence.normalized_text:
                continue

            best_ratio = 0
            best_candidate = ""
            term_len = max(1, len(tokenize_normalized(normalized_term)))
            min_len = max(1, term_len - self.ngram_delta)
            max_len = max(min_len, term_len + self.ngram_delta)

            for ngram_len in range(min_len, max_len + 1):
                if ngram_len > len(sentence_tokens):
                    continue
                for index in range(0, len(sentence_tokens) - ngram_len + 1):
                    candidate = " ".join(sentence_tokens[index : index + ngram_len])
                    ratio = self._fuzz.WRatio(normalized_term, candidate)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_candidate = candidate

            if best_ratio >= self.threshold:
                score = _base_score((best_ratio / 100.0) * 0.7, context_hits, field_hits)
                findings.append(
                    Evidence(
                        category_key=category.key,
                        category_label=category.label,
                        method=self.method_name,
                        score=score,
                        trigger=term,
                        matched_variant=best_candidate,
                        sentence=sentence.text,
                        explanation=(
                            f"Approximation fuzzy entre '{term}' et '{best_candidate}' "
                            f"avec une similarite de {best_ratio}."
                        ),
                        context_hits=context_hits,
                        field_hits=field_hits,
                        metadata={"sentence_index": sentence.index, "similarity": best_ratio},
                    )
                )
        return findings


class SemanticDetector:
    method_name = "semantic"

    def __init__(self, enabled: bool, model_name: str, similarity_threshold: float):
        self.enabled = enabled
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.available = False
        self._model = None
        self._category_embeddings: dict[str, list[list[float]]] = {}
        self._sentence_cache: dict[str, list[float]] = {}
        if enabled:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.available = True
        except Exception:
            self._model = None
            self.available = False

    def _encode_sentence(self, text: str):
        cached = self._sentence_cache.get(text)
        if cached is not None:
            return cached
        vector = self._model.encode(text, normalize_embeddings=True)
        self._sentence_cache[text] = vector
        return vector

    def _category_anchor_vectors(self, category: CategoryConfig):
        cached = self._category_embeddings.get(category.key)
        if cached is not None:
            return cached
        vectors = self._model.encode(category.semantic_anchors, normalize_embeddings=True)
        self._category_embeddings[category.key] = vectors
        return vectors

    def detect(self, sentence: SentenceUnit, category: CategoryConfig) -> list[Evidence]:
        if not self.available or not category.semantic_anchors:
            return []

        import numpy as np

        sentence_vector = self._encode_sentence(sentence.text)
        anchor_vectors = self._category_anchor_vectors(category)
        similarities = np.dot(anchor_vectors, sentence_vector)
        best_index = int(similarities.argmax())
        best_score = float(similarities[best_index])

        if best_score < self.similarity_threshold:
            return []

        context_hits = _collect_hits(sentence.normalized_text, category.context_terms)
        field_hits = _collect_hits(sentence.normalized_text, category.field_hints)
        score = _base_score(best_score, context_hits, field_hits)
        anchor = category.semantic_anchors[best_index]
        return [
            Evidence(
                category_key=category.key,
                category_label=category.label,
                method=self.method_name,
                score=score,
                trigger=anchor,
                matched_variant=sentence.text,
                sentence=sentence.text,
                explanation=(
                    f"Proximite semantique entre la phrase et l'ancre '{anchor}' "
                    f"avec une similarite de {best_score:.3f}."
                ),
                context_hits=context_hits,
                field_hits=field_hits,
                metadata={"sentence_index": sentence.index, "similarity": round(best_score, 4)},
            )
        ]
