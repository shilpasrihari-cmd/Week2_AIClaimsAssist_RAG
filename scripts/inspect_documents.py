"""Inspect all PDF and DOCX files in data/policies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct execution from the repository root or from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.document_loader import DocumentExtraction, inspect_directory


def _print_result(result: DocumentExtraction) -> None:
    print(f"Filename: {result.source_filename}")
    print(f"Document ID: {result.document_id}")
    print(f"File type: {result.file_type}")
    if result.file_type == "PDF":
        print(f"Pages: {result.page_count}")
    else:
        print(f"Paragraphs: {result.paragraph_count}")
        print(f"Tables: {result.table_count}")
    print(f"Extracted character count: {result.extracted_character_count}")
    print(f"First 300 characters: {result.preview!r}")
    print(f"Extraction succeeded: {result.extraction_succeeded}")
    if result.error:
        print(f"Error: {result.error}")
    print("-" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "policies",
        help="Directory containing PDF and DOCX policy documents.",
    )
    args = parser.parse_args()
    results = inspect_directory(args.directory)
    if not results:
        print(f"No supported documents found in {args.directory}")
        return 1
    for result in results:
        _print_result(result)
    return 0 if all(result.extraction_succeeded for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
