from types import SimpleNamespace

import numpy as np
import pytest

from ingestion.chunker import Chunk
from ingestion.indexer import IndexingError, PineconeIndexer


class FakeEmbedder:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, sentences, **kwargs):
        return np.ones((len(sentences), self.dimension), dtype=np.float32)


class FakeIndex:
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.uploads = []

    def describe_index(self):
        return SimpleNamespace(dimension=self.dimension)

    def upsert(self, *, vectors, namespace):
        self.uploads.append((vectors, namespace))
        return SimpleNamespace(upserted_count=len(vectors))


def _chunk() -> Chunk:
    return Chunk(
        text="Coverage applies to the insured vehicle.",
        policy_id="MOT-0000002",
        document_id="MOT-0000002-contract",
        source_filename="MOT-0000002-contract.docx",
        document_type="docx",
        section_heading="Insuring Agreement",
        chunk_id="MOT-0000002-contract:section_aware:0000:abc123",
        chunking_strategy="section_aware",
    )


def test_embeddings_are_384_dimensional_and_metadata_is_preserved() -> None:
    index = FakeIndex()
    count = PineconeIndexer(index, FakeEmbedder()).index_chunks([_chunk()], "section-aware")

    assert count == 1
    vectors, namespace = index.uploads[0]
    assert namespace == "section-aware"
    assert len(vectors[0]["values"]) == 384
    assert vectors[0]["id"] == _chunk().chunk_id
    assert vectors[0]["metadata"]["policy_id"] == "MOT-0000002"
    assert vectors[0]["metadata"]["source_filename"] == "MOT-0000002-contract.docx"
    assert vectors[0]["metadata"]["section_heading"] == "Insuring Agreement"


def test_dimension_mismatch_fails_before_upload() -> None:
    index = FakeIndex(dimension=768)

    with pytest.raises(IndexingError, match="does not match expected 384"):
        PineconeIndexer(index, FakeEmbedder()).index_chunks([_chunk()], "section-aware")

    assert index.uploads == []


def test_embedding_shape_error_is_reported() -> None:
    index = FakeIndex()

    with pytest.raises(IndexingError, match="Embedding output shape"):
        PineconeIndexer(index, FakeEmbedder(dimension=3)).index_chunks([_chunk()], "fixed-size")


def test_upsert_error_does_not_expose_api_key() -> None:
    class BrokenIndex(FakeIndex):
        def upsert(self, *, vectors, namespace):
            raise RuntimeError("network unavailable")

    with pytest.raises(IndexingError, match="Pinecone upsert failed"):
        PineconeIndexer(BrokenIndex(), FakeEmbedder()).index_chunks([_chunk()], "fixed-size")
