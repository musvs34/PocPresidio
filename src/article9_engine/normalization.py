from __future__ import annotations

import re
import unicodedata


APOSTROPHES = {"’": "'", "`": "'", "´": "'", "ʼ": "'"}
DASHES = {"–": "-", "—": "-", "‐": "-", "‑": "-"}
PUNCTUATION_PATTERN = re.compile(r"[^a-z0-9\s'-]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_accents_enabled: bool = True,
    collapse_whitespace: bool = True,
    normalize_apostrophes: bool = True,
    normalize_dashes: bool = True,
) -> str:
    value = text or ""
    if normalize_apostrophes:
        for source, target in APOSTROPHES.items():
            value = value.replace(source, target)
    if normalize_dashes:
        for source, target in DASHES.items():
            value = value.replace(source, target)
    if lowercase:
        value = value.lower()
    if strip_accents_enabled:
        value = strip_accents(value)
    value = PUNCTUATION_PATTERN.sub(" ", value)
    if collapse_whitespace:
        value = WHITESPACE_PATTERN.sub(" ", value).strip()
    return value


def tokenize_normalized(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def root_form(token: str) -> str:
    value = token.strip("'")
    if len(value) <= 4:
        return value

    suffixes = [
        "issements",
        "issement",
        "atrices",
        "atrice",
        "ations",
        "ation",
        "utions",
        "ution",
        "ances",
        "ance",
        "ences",
        "ence",
        "ements",
        "ement",
        "ateurs",
        "ateur",
        "euses",
        "euse",
        "eaux",
        "eau",
        "iques",
        "ique",
        "istes",
        "iste",
        "ismes",
        "isme",
        "ables",
        "able",
        "ment",
        "ments",
        "tion",
        "tions",
        "ites",
        "ite",
        "ante",
        "ants",
        "ant",
        "tes",
        "te",
        "ses",
        "se",
        "ees",
        "ee",
        "es",
        "s",
        "e",
    ]

    root = value
    for suffix in suffixes:
        if len(root) > len(suffix) + 3 and root.endswith(suffix):
            root = root[: -len(suffix)]
            break

    if len(root) > 6:
        return root[:6]
    return root
