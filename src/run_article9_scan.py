from __future__ import annotations

import argparse
from pathlib import Path

from article9_engine import Article9Engine
from article9_engine.reporting import analyze_directory, write_reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan raw files for RGPD article 9 hints with a multicouche French engine."
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Directory containing the raw documents to scan.",
    )
    parser.add_argument(
        "--config",
        default="configs/article9_categories.yml",
        help="YAML configuration used to build the article 9 engine.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where CSV and JSON outputs will be written.",
    )
    args = parser.parse_args()

    engine = Article9Engine(config_path=args.config)
    analyses = analyze_directory(engine, args.input_dir)
    outputs = write_reports(analyses, args.output_dir)

    print(f"Scanned directory: {Path(args.input_dir).resolve()}")
    print(f"Chunks analyzed: {len(analyses)}")
    for label, path in outputs.items():
        print(f"{label}: {path.resolve()}")


if __name__ == "__main__":
    main()
