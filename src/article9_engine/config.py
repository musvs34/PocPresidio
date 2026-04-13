from __future__ import annotations

from pathlib import Path

from .models import CategoryConfig, EngineConfig


def load_engine_config(path: str | Path) -> EngineConfig:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Loading YAML configuration requires PyYAML. Install requirements.txt first."
        ) from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    categories = [
        CategoryConfig(
            key=item["key"],
            label=item["label"],
            description=item.get("description", ""),
            terms=item.get("terms", []),
            synonyms=item.get("synonyms", []),
            context_terms=item.get("context_terms", []),
            field_hints=item.get("field_hints", []),
            semantic_anchors=item.get("semantic_anchors", []),
            thresholds=item.get("thresholds", {}),
            weights=item.get("weights", {}),
        )
        for item in raw.get("categories", [])
    ]

    return EngineConfig(
        language=raw.get("language", "fr"),
        normalization=raw.get("normalization", {}),
        fuzzy=raw.get("fuzzy", {}),
        linguistics=raw.get("linguistics", {}),
        semantic=raw.get("semantic", {}),
        scoring=raw.get("scoring", {}),
        categories=categories,
    )
