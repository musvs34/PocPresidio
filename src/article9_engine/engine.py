from __future__ import annotations

from .config import load_engine_config
from .detectors import FuzzyDetector, LexicalDetector, LinguisticDetector, SemanticDetector
from .linguistics import FrenchTextAnalyzer
from .models import AnalysisResult, CategoryResult, Evidence
from .normalization import normalize_text
from .scoring import score_category


class Article9Engine:
    def __init__(self, config_path: str = "configs/article9_categories.yml"):
        self.config_path = config_path
        self.config = load_engine_config(config_path)
        normalization_options = {
            "lowercase": self.config.normalization.get("lowercase", True),
            "strip_accents_enabled": self.config.normalization.get("strip_accents", True),
            "collapse_whitespace": self.config.normalization.get("collapse_whitespace", True),
            "normalize_apostrophes": self.config.normalization.get("normalize_apostrophes", True),
            "normalize_dashes": self.config.normalization.get("normalize_dashes", True),
        }
        self.normalization_options = normalization_options
        self.text_analyzer = FrenchTextAnalyzer(
            spacy_model=self.config.linguistics.get("spacy_model", "fr_core_news_md"),
            normalization_options=normalization_options,
        )
        self.lexical_detector = LexicalDetector()
        self.linguistic_detector = LinguisticDetector()
        self.fuzzy_detector = FuzzyDetector(
            threshold=int(self.config.fuzzy.get("similarity_threshold", 88)),
            ngram_delta=int(self.config.fuzzy.get("max_ngram_length_delta", 1)),
        )
        self.semantic_detector = SemanticDetector(
            enabled=bool(self.config.semantic.get("enabled", False)),
            model_name=self.config.semantic.get(
                "model_name",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            similarity_threshold=float(self.config.semantic.get("similarity_threshold", 0.62)),
        )

    def _decision_log(self) -> list[str]:
        semantic_status = (
            f"semantic:{self.semantic_detector.model_name}"
            if self.semantic_detector.available
            else "semantic:disabled_or_unavailable"
        )
        fuzzy_status = (
            "fuzzy:rapidfuzz" if self.fuzzy_detector.available else "fuzzy:disabled_or_unavailable"
        )
        return [
            f"linguistics_backend:{self.text_analyzer.backend}",
            fuzzy_status,
            semantic_status,
        ]

    def analyze_text(
        self,
        text: str,
        *,
        text_id: str = "inline_text",
        metadata: dict | None = None,
    ) -> AnalysisResult:
        normalized_text = normalize_text(text, **self.normalization_options)
        sentences = self.text_analyzer.analyze(text)
        category_results: list[CategoryResult] = []

        for category in self.config.categories:
            evidences: list[Evidence] = []
            for sentence in sentences:
                evidences.extend(self.lexical_detector.detect(sentence, category))
                evidences.extend(self.fuzzy_detector.detect(sentence, category))
                evidences.extend(self.linguistic_detector.detect(sentence, category))
                evidences.extend(self.semantic_detector.detect(sentence, category))

            deduped = self._deduplicate_evidences(evidences)
            category_results.append(score_category(category, deduped, self.config))

        return AnalysisResult(
            text_id=text_id,
            original_text=text,
            normalized_text=normalized_text,
            category_results=category_results,
            decision_log=self._decision_log(),
            metadata=metadata or {},
        )

    def _deduplicate_evidences(self, evidences: list[Evidence]) -> list[Evidence]:
        best_by_key: dict[tuple[str, str, str], Evidence] = {}
        for evidence in evidences:
            key = (evidence.method, evidence.trigger, evidence.sentence)
            current = best_by_key.get(key)
            if current is None or evidence.score > current.score:
                best_by_key[key] = evidence
        return sorted(best_by_key.values(), key=lambda item: item.score, reverse=True)
