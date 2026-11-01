from pathlib import Path

from retrieval.bm25_retriever import BM25Retriever, tokenize


POLICIES = Path(__file__).parents[1] / "data" / "policies"


def test_exact_policy_id_retrieval_and_filtering() -> None:
    retriever = BM25Retriever.from_directory(POLICIES)

    results = retriever.search("MOT-0000002", strategy="fixed_size", top_k=5)
    filtered = retriever.search(
        "MOT-0000002 deductible", strategy="section_aware", top_k=5, policy_id="MOT-0000002"
    )

    assert results
    assert all(result.policy_id == "MOT-0000002" for result in results)
    assert filtered
    assert all(result.policy_id == "MOT-0000002" for result in filtered)


def test_com_policy_id_retrieval_and_filtering() -> None:
    retriever = BM25Retriever.from_directory(POLICIES)

    results = retriever.search("COM-0000001 coverage", strategy="section_aware", top_k=5)

    assert results
    assert all(result.policy_id == "COM-0000001" for result in results)


def test_collision_and_deductible_queries_return_both_strategies() -> None:
    retriever = BM25Retriever.from_directory(POLICIES)

    for strategy in ("fixed_size", "section-aware"):
        collision = retriever.search("collision coverage", strategy=strategy, top_k=5)
        deductible = retriever.search("deductible", strategy=strategy, top_k=5)
        assert collision
        assert deductible
        assert all(result.score > 0 for result in collision + deductible)


def test_gap_query_and_metadata_preservation() -> None:
    retriever = BM25Retriever.from_directory(POLICIES)
    results = retriever.search("GAP coverage loan balance", strategy="section_aware", top_k=5)

    assert results
    assert any(result.policy_id == "MOT-0000004" for result in results)
    assert any(result.source_filename.endswith("endorsements.pdf") for result in results)
    assert all(result.chunk_id and result.section_heading and result.text for result in results)


def test_empty_queries_and_no_match_are_graceful() -> None:
    retriever = BM25Retriever.from_directory(POLICIES)

    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert retriever.search("term-that-does-not-exist-anywhere") == []
    assert retriever.search("deductible", top_k=0) == []


def test_tokenizer_is_shared_and_preserves_policy_id_tokens() -> None:
    assert tokenize("MOT-0000002's deductible") == ["mot-0000002", "s", "deductible"]
