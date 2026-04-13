from __future__ import annotations

import argparse
from pathlib import Path

from article9_pipeline.scanner import scan_documents, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan raw files for RGPD article 9 hints using Presidio-style recognizers."
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Directory containing the raw documents to scan.",
    )
    parser.add_argument(
        "--vocabulary",
        default="configs/article9_vocabulary_template.csv",
        help="CSV vocabulary used to build the custom recognizers.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where CSV and JSON outputs will be written.",
    )
    args = parser.parse_args()

    findings, summary_rows = scan_documents(args.input_dir, args.vocabulary)
    outputs = write_outputs(findings, summary_rows, args.output_dir)

    print(f"Scanned directory: {Path(args.input_dir).resolve()}")
    print(f"Findings: {len(findings)}")
    print(f"Summary rows: {len(summary_rows)}")
    for label, path in outputs.items():
        print(f"{label}: {path.resolve()}")


if __name__ == "__main__":
    main()
