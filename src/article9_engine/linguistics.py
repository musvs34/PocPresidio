from __future__ import annotations

import re

from .models import SentenceUnit
from .normalization import normalize_text, root_form, tokenize_normalized


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[\.\!\?\n;])\s+")


class FrenchTextAnalyzer:
    """Sentence segmentation and lemmatization with graceful fallback."""

    def __init__(self, spacy_model: str = "fr_core_news_md", normalization_options=None):
        self.spacy_model = spacy_model
        self.normalization_options = normalization_options or {}
        self._nlp = None
        self.backend = "regex"
        self._load_spacy()

    def _load_spacy(self) -> None:
        try:
            import spacy

            self._nlp = spacy.load(self.spacy_model)
            self.backend = f"spacy:{self.spacy_model}"
        except Exception:
            self._nlp = None
            self.backend = "regex"

    def analyze(self, text: str) -> list[SentenceUnit]:
        if self._nlp is not None:
            return self._analyze_with_spacy(text)
        return self._analyze_with_regex(text)

    def _analyze_with_spacy(self, text: str) -> list[SentenceUnit]:
        doc = self._nlp(text)
        sentences: list[SentenceUnit] = []
        for index, sent in enumerate(doc.sents):
            tokens = [
                token.text
                for token in sent
                if not token.is_space and not token.is_punct
            ]
            lemmas = [
                normalize_text(token.lemma_, **self.normalization_options)
                for token in sent
                if not token.is_space and not token.is_punct
            ]
            roots = [root_form(lemma) for lemma in lemmas if lemma]
            normalized = normalize_text(sent.text, **self.normalization_options)
            sentences.append(
                SentenceUnit(
                    text=sent.text.strip(),
                    normalized_text=normalized,
                    tokens=tokens,
                    lemmas=[lemma for lemma in lemmas if lemma],
                    roots=[root for root in roots if root],
                    index=index,
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                )
            )
        return [sentence for sentence in sentences if sentence.text]

    def _analyze_with_regex(self, text: str) -> list[SentenceUnit]:
        chunks = [chunk.strip() for chunk in SENTENCE_SPLIT_PATTERN.split(text) if chunk.strip()]
        sentences: list[SentenceUnit] = []
        cursor = 0
        for index, chunk in enumerate(chunks):
            start_char = text.find(chunk, cursor)
            if start_char < 0:
                start_char = cursor
            end_char = start_char + len(chunk)
            cursor = end_char
            normalized = normalize_text(chunk, **self.normalization_options)
            lemmas = tokenize_normalized(normalized)
            roots = [root_form(token) for token in lemmas]
            sentences.append(
                SentenceUnit(
                    text=chunk,
                    normalized_text=normalized,
                    tokens=lemmas,
                    lemmas=lemmas,
                    roots=roots,
                    index=index,
                    start_char=start_char,
                    end_char=end_char,
                )
            )
        return sentences
