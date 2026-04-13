from __future__ import annotations

import argparse
import json
from pathlib import Path

from article9_engine import load_engine_config


def ensure_spacy_model(model_name: str) -> dict[str, str]:
    try:
        import spacy

        spacy.load(model_name)
        return {"resource": "spacy_model", "name": model_name, "status": "already_available"}
    except Exception as local_exc:
        try:
            from spacy.cli import download

            download(model_name)
            return {
                "resource": "spacy_model",
                "name": model_name,
                "status": "downloaded",
                "local_error": str(local_exc),
            }
        except Exception as download_exc:
            return {
                "resource": "spacy_model",
                "name": model_name,
                "status": "failed",
                "local_error": str(local_exc),
                "download_error": str(download_exc),
            }


def ensure_sentence_transformer(model_name: str) -> dict[str, str]:
    from sentence_transformers import SentenceTransformer

    try:
        SentenceTransformer(model_name, local_files_only=True)
        return {
            "resource": "sentence_transformer",
            "name": model_name,
            "status": "already_available_local",
        }
    except Exception as local_exc:
        try:
            SentenceTransformer(model_name, local_files_only=False)
            return {
                "resource": "sentence_transformer",
                "name": model_name,
                "status": "prepared",
                "local_error": str(local_exc),
            }
        except Exception as download_exc:
            return {
                "resource": "sentence_transformer",
                "name": model_name,
                "status": "failed",
                "local_error": str(local_exc),
                "download_error": str(download_exc),
            }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare external NLP resources required by the Article 9 engine."
    )
    parser.add_argument(
        "--config",
        default="configs/article9_categories.yml",
        help="YAML configuration describing the NLP resources to prepare.",
    )
    args = parser.parse_args()

    config = load_engine_config(args.config)
    results: list[dict[str, str]] = []

    spacy_model = config.linguistics.get("spacy_model", "fr_core_news_md")
    results.append(ensure_spacy_model(spacy_model))

    if config.semantic.get("enabled", False):
        model_name = config.semantic.get(
            "model_name",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        results.append(ensure_sentence_transformer(model_name))

    print(json.dumps(results, ensure_ascii=False, indent=2))

    if any(item["status"] == "failed" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
