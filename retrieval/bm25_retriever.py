"""BM25 keyword retrieval over the project's existing chunk strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rank_bm25 import BM25Okapi

from ingestion.chunker import Chunk, chunk_documents


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.IGNORECASE)
_POLICY_ID_PATTERN = re.compile(r"\b(?:MOT|COM)-\d+\b", re.IGNORECASE)
_STRATEGY_ALIASES = {
    "fixed_size": "fixed_size",
    "fixed-size": "fixed_size",
    "section_aware": "section_aware",
    "section-aware": "section_aware",
}


def tokenize(text: str) -> list[str]:
    """Tokenize both corpus text and queries consistently, preserving policy IDs."""

    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def extract_policy_id(text: str) -> str | None:
    match = _POLICY_ID_PATTERN.search(text)
    return match.group(0).upper() if match else None


@dataclass(frozen=True)
class BM25Result:
    chunk_id: str
    policy_id: str
    source_filename: str
    section_heading: str
    text: str
    score: float


class _BM25Index:
    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self.chunks = tuple(chunks)
        self.tokenized_chunks = [tokenize(chunk.text) for chunk in self.chunks]
        self.index = BM25Okapi(self.tokenized_chunks) if self.chunks else None

    def search(self, query: str, top_k: int, policy_id: str | None = None) -> list[BM25Result]:
        query_tokens = tokenize(query)
        requested_policy_id = policy_id or extract_policy_id(query)
        if not query_tokens or top_k <= 0 or self.index is None:
            return []
        if requested_policy_id is not None and requested_policy_id not in {chunk.policy_id for chunk in self.chunks}:
            return []

        scores = self.index.get_scores(query_tokens)
        candidates = (
            (position, chunk, float(scores[position]))
            for position, chunk in enumerate(self.chunks)
            if requested_policy_id is None or chunk.policy_id == requested_policy_id
        )
        ranked = sorted(candidates, key=lambda item: (-item[2], item[0]))
        positive = [item for item in ranked if item[2] > 0]
        return [
            BM25Result(
                chunk_id=chunk.chunk_id,
                policy_id=chunk.policy_id,
                source_filename=chunk.source_filename,
                section_heading=chunk.section_heading,
                text=chunk.text,
                score=score,
            )
            for _, chunk, score in positive[:top_k]
        ]


class BM25Retriever:
    """Separate BM25 indexes backed by exactly the existing generated chunks."""

    def __init__(self, indexes: dict[str, _BM25Index]) -> None:
        self._indexes = indexes

    @classmethod
    def from_directory(cls, directory: str | Path) -> "BM25Retriever":
        return cls(
            {
                strategy: _BM25Index(chunk_documents(directory, strategy))
                for strategy in ("fixed_size", "section_aware")
            }
        )

    def search(
        self,
        query: str,
        *,
        strategy: str = "fixed_size",
        top_k: int = 5,
        policy_id: str | None = None,
    ) -> list[BM25Result]:
        normalized_strategy = _STRATEGY_ALIASES.get(strategy)
        if normalized_strategy is None:
            raise ValueError("strategy must be 'fixed_size' or 'section_aware'")
        return self._indexes[normalized_strategy].search(query, top_k, policy_id)

    def chunk_counts(self) -> dict[str, int]:
        return {strategy: len(index.chunks) for strategy, index in self._indexes.items()}
