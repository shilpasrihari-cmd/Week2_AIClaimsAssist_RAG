"""Generate semantic embeddings and index policy chunks in Pinecone."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import truststore
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from .chunker import Chunk, chunk_documents


truststore.inject_into_ssl()


EXPECTED_DIMENSION = 384
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NAMESPACE_BY_STRATEGY = {"fixed_size": "fixed-size", "section_aware": "section-aware"}


class EmbeddingModel(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


class IndexingError(RuntimeError):
    """Raised when an index cannot be validated or vectors cannot be uploaded."""


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise IndexingError(f"Required environment variable is missing: {name}")
    return value


def connect_to_index(index_name: str | None = None) -> Any:
    """Connect using `.env`/environment values without logging credentials."""

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = _required_env("PINECONE_API_KEY")
    name = index_name or os.getenv("PINECONE_INDEX_NAME", "aiclaimsassist")
    return Pinecone(api_key=api_key).Index(name)


def load_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return SentenceTransformer(model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))


class PineconeIndexer:
    """Embed and upsert chunks after validating the target index dimension."""

    def __init__(
        self,
        index: Any,
        embedder: EmbeddingModel,
        expected_dimension: int = EXPECTED_DIMENSION,
        batch_size: int = 100,
        index_description: Any | None = None,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.expected_dimension = expected_dimension
        self.batch_size = batch_size
        self.index_description = index_description

    def validate_index_dimension(self) -> None:
        """Fail before uploading if Pinecone's configured dimension is wrong."""

        description = self.index_description
        if description is None:
            description = self.index.describe_index()
        dimension = getattr(description, "dimension", None)
        if dimension is None and isinstance(description, dict):
            dimension = description.get("dimension") or description.get("spec", {}).get("dimension")
        if dimension != self.expected_dimension:
            raise IndexingError(
                f"Pinecone index dimension {dimension!r} does not match expected {self.expected_dimension}."
            )

    def _embed(self, chunks: Sequence[Chunk]) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(embeddings)
        if matrix.ndim != 2 or matrix.shape != (len(chunks), self.expected_dimension):
            raise IndexingError(
                f"Embedding output shape {matrix.shape} does not match "
                f"({len(chunks)}, {self.expected_dimension})."
            )
        return matrix

    def index_chunks(self, chunks: Sequence[Chunk], namespace: str) -> int:
        if not chunks:
            return 0
        self.validate_index_dimension()
        matrix = self._embed(chunks)
        total = 0
        for start in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[start : start + self.batch_size]
            batch_vectors = [
                {
                    "id": chunk.chunk_id,
                    "values": matrix[start + offset].tolist(),
                    "metadata": {**chunk.metadata, "text": chunk.text},
                }
                for offset, chunk in enumerate(batch_chunks)
            ]
            try:
                response = self.index.upsert(vectors=batch_vectors, namespace=namespace)
            except Exception as exc:
                raise IndexingError(f"Pinecone upsert failed for namespace {namespace!r}: {exc}") from exc
            uploaded = getattr(response, "upserted_count", None)
            total += int(uploaded) if uploaded is not None else len(batch_vectors)
        return total

    def index_all(self, directory: str | Path) -> dict[str, int]:
        counts: dict[str, int] = {}
        for strategy, namespace in NAMESPACE_BY_STRATEGY.items():
            chunks = chunk_documents(directory, strategy)
            counts[namespace] = self.index_chunks(chunks, namespace)
        return counts


def create_indexer(index_name: str | None = None) -> PineconeIndexer:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = _required_env("PINECONE_API_KEY")
    name = index_name or os.getenv("PINECONE_INDEX_NAME", "aiclaimsassist")
    client = Pinecone(api_key=api_key)
    return PineconeIndexer(
        client.Index(name),
        load_embedding_model(),
        index_description=client.describe_index(name),
    )
