from types import SimpleNamespace

from retrieval.hybrid_retriever import HybridResult
from retrieval.reranker import CrossEncoderReranker


def _candidate(chunk_id: str, policy_id: str = "MOT-0000002", score: float = 0.01) -> HybridResult:
    return HybridResult(
        chunk_id=chunk_id,
        policy_id=policy_id,
        source_filename=f"{policy_id}-contract.docx",
        section_heading="Insuring Agreement",
        text=f"passage {chunk_id}",
        retrieval_source="hybrid",
        ranking_score=score,
        score_type="rrf_score",
    )


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def predict(self, pairs, **kwargs):
        self.calls += 1
        return self.scores[: len(pairs)]


def test_reranking_order_and_metadata_preservation() -> None:
    model = FakeCrossEncoder([0.1, 0.9, 0.4])
    reranker = CrossEncoderReranker(model=model)
    candidates = [_candidate("one", score=0.03), _candidate("two", score=0.02), _candidate("three", score=0.01)]

    results = reranker.rerank("deductible", candidates)

    assert [result.chunk_id for result in results] == ["two", "three", "one"]
    assert results[0].original_rrf_score == 0.02
    assert results[0].cross_encoder_score == 0.9
    assert results[0].original_rank == 2
    assert results[0].reranked_rank == 1
    assert results[0].source_filename == "MOT-0000002-contract.docx"
    assert results[0].policy_id == "MOT-0000002"
    assert model.calls == 1


def test_empty_candidates_and_policy_filtering() -> None:
    model = FakeCrossEncoder([0.9, 0.8])
    reranker = CrossEncoderReranker(model=model)
    candidates = [_candidate("two", "MOT-0000002"), _candidate("four", "MOT-0000004")]

    assert reranker.rerank("", candidates) == []
    assert reranker.rerank("GAP coverage", []) == []
    filtered = reranker.rerank("MOT-0000002 deductible", candidates)
    assert filtered
    assert all(result.policy_id == "MOT-0000002" for result in filtered)


def test_duplicate_ids_are_removed_and_only_ten_candidates_are_scored() -> None:
    model = FakeCrossEncoder([float(i) for i in range(12)])
    reranker = CrossEncoderReranker(model=model)
    candidates = [_candidate("duplicate") for _ in range(2)] + [_candidate(str(i)) for i in range(11)]

    results = reranker.rerank("query", candidates, top_k=20)

    assert len(results) == 10
    assert len({result.chunk_id for result in results}) == 10
    assert model.calls == 1


def test_model_failure_is_graceful() -> None:
    class BrokenModel:
        def predict(self, pairs, **kwargs):
            raise RuntimeError("model unavailable")

    reranker = CrossEncoderReranker(model=BrokenModel())

    assert reranker.rerank("query", [_candidate("one")]) == []
    assert reranker.last_error == "Cross-encoder reranking failed: model unavailable"
