from __future__ import annotations

import argparse
import json
from pathlib import Path

from article9_engine import Article9Engine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze example French phrases against Article 9 categories."
    )
    parser.add_argument(
        "--config",
        default="configs/article9_categories.yml",
        help="YAML configuration describing sensitive categories.",
    )
    parser.add_argument(
        "--examples",
        default="examples/article9_sample_sentences.json",
        help="JSON file containing example French phrases.",
    )
    args = parser.parse_args()

    engine = Article9Engine(config_path=args.config)
    examples_path = Path(args.examples)
    payload = json.loads(examples_path.read_text(encoding="utf-8"))

    results = []
    for item in payload:
        analysis = engine.analyze_text(item["text"], text_id=item["text_id"])
        results.append(
            {
                "text_id": item["text_id"],
                "text": item["text"],
                "decision_log": analysis.decision_log,
                "categories": [
                    {
                        "category_key": category.category_key,
                        "label": category.category_label,
                        "score": category.score,
                        "decision": category.decision,
                        "explanation": category.explanation,
                    }
                    for category in analysis.category_results
                    if category.decision != "clear"
                ],
            }
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
