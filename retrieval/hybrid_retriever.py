"""Semantic, keyword, and RRF hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bm25_retriever import BM25Result, BM25Retriever
from .semantic_retriever import SemanticResult, SemanticRetriever


RRF_K = 60


@dataclass(frozen=True)
class HybridResult:
    chunk_id: str
    policy_id: str
    source_filename: str
    section_heading: str
    text: str
    retrieval_source: str
    ranking_score: float
    score_type: str
    semantic_score: float | None = None
    bm25_score: float | None = None


class HybridRetriever:
    def __init__(self, semantic: SemanticRetriever, keyword: BM25Retriever) -> None:
        self.semantic = semantic
        self.keyword = keyword

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        index=None,
        embedder=None,
    ) -> "HybridRetriever":
        return cls(
            SemanticRetriever.from_directory(directory, index=index, embedder=embedder),
            BM25Retriever.from_directory(directory),
        )

    @staticmethod
    def _validate_strategy(strategy: str) -> str:
        aliases = {"fixed_size": "fixed_size", "fixed-size": "fixed_size", "section_aware": "section_aware", "section-aware": "section_aware"}
        if strategy not in aliases:
            raise ValueError("strategy must be 'fixed_size' or 'section_aware'")
        return aliases[strategy]

    @staticmethod
    def _semantic_result(result: SemanticResult) -> HybridResult:
        return HybridResult(
            chunk_id=result.chunk_id,
            policy_id=result.policy_id,
            source_filename=result.source_filename,
            section_heading=result.section_heading,
            text=result.text,
            retrieval_source="semantic",
            ranking_score=result.semantic_score,
            score_type="semantic_similarity",
            semantic_score=result.semantic_score,
        )

    @staticmethod
    def _keyword_result(result: BM25Result) -> HybridResult:
        return HybridResult(
            chunk_id=result.chunk_id,
            policy_id=result.policy_id,
            source_filename=result.source_filename,
            section_heading=result.section_heading,
            text=result.text,
            retrieval_source="keyword",
            ranking_score=result.score,
            score_type="bm25_score",
            bm25_score=result.score,
        )

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        strategy: str = "section_aware",
        top_k: int = 5,
        policy_id: str | None = None,
    ) -> list[HybridResult]:
        if mode not in {"semantic_only", "keyword_only", "hybrid"}:
            raise ValueError("mode must be 'semantic_only', 'keyword_only', or 'hybrid'")
        normalized_strategy = self._validate_strategy(strategy)
        if self.semantic.strategy and self.semantic.strategy != normalized_strategy:
            raise ValueError("Semantic and requested chunking strategies do not match")
        if not query.strip() or top_k <= 0:
            return []
        if policy_id is not None and policy_id not in self.semantic.valid_policy_ids:
            return []

        semantic_results: list[SemanticResult] = []
        keyword_results: list[BM25Result] = []
        if mode in {"semantic_only", "hybrid"}:
            try:
                semantic_results = self.semantic.search(
                    query, strategy=normalized_strategy, top_k=5, policy_id=policy_id
                )
            except Exception:
                if mode == "semantic_only":
                    return []
        if mode in {"keyword_only", "hybrid"}:
            keyword_results = self.keyword.search(
                query, strategy=normalized_strategy, top_k=5, policy_id=policy_id
            )

        if mode == "semantic_only":
            return [self._semantic_result(result) for result in semantic_results[:top_k]]
        if mode == "keyword_only":
            return [self._keyword_result(result) for result in keyword_results[:top_k]]

        fused: dict[str, dict] = {}
        for rank, result in enumerate(semantic_results, start=1):
            entry = fused.setdefault(result.chunk_id, {"result": result, "rrf": 0.0, "semantic": None, "bm25": None})
            entry["rrf"] += 1 / (RRF_K + rank)
            entry["semantic"] = result.semantic_score
        for rank, result in enumerate(keyword_results, start=1):
            entry = fused.setdefault(result.chunk_id, {"result": result, "rrf": 0.0, "semantic": None, "bm25": None})
            entry["rrf"] += 1 / (RRF_K + rank)
            entry["bm25"] = result.score

        ordered = sorted(fused.values(), key=lambda entry: (-entry["rrf"], entry["result"].chunk_id))
        output: list[HybridResult] = []
        for entry in ordered[:top_k]:
            result = entry["result"]
            output.append(
                HybridResult(
                    chunk_id=result.chunk_id,
                    policy_id=result.policy_id,
                    source_filename=result.source_filename,
                    section_heading=result.section_heading,
                    text=result.text,
                    retrieval_source="hybrid" if entry["semantic"] is not None and entry["bm25"] is not None else "semantic" if entry["semantic"] is not None else "keyword",
                    ranking_score=entry["rrf"],
                    score_type="rrf_score",
                    semantic_score=entry["semantic"],
                    bm25_score=entry["bm25"],
                )
            )
        return output
