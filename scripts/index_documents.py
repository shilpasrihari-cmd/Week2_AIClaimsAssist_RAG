"""Embed and index all policy chunks in the two configured namespaces."""

from __future__ import annotations

from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.indexer import IndexingError, create_indexer


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        indexer = create_indexer()
        data_root = Path(os.getenv("RAG_DATA_DIR", root / "data" / "sample"))
        counts = indexer.index_all(data_root)
    except IndexingError as exc:
        print(f"Indexing failed: {exc}")
        return 1
    for namespace, count in counts.items():
        print(f"Indexed vectors: namespace={namespace}, count={count}")
    stats = indexer.index.describe_index_stats()
    print(f"Index statistics: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
