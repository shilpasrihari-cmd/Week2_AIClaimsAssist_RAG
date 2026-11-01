"""Chunk the text-bearing documents in the copied sample corpus."""

from __future__ import annotations

import hashlib
from pathlib import Path

# Reuse the same splitter and section rules as the original policy chunker so
# sample-corpus chunks remain compatible with the existing retrievers.
from .chunker import Chunk, _sections, _token_splitter
from .sample_loader import SampleDocument, load_sample_documents


def _chunk_id(document: SampleDocument, strategy: str, index: int, text: str) -> str:
    # The document ID, strategy, position, and content digest make IDs stable
    # across repeated indexing while changing when source content changes.
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{document.doc_id}:{strategy}:{index:04d}:{digest}"


def _make_chunk(document: SampleDocument, strategy: str, heading: str, index: int, text: str) -> Chunk:
    # Preserve manifest entity IDs alongside the policy/document metadata for
    # claim, policyholder, adjuster, and policy-level filtering.
    return Chunk(
        text=text,
        policy_id=document.policy_id,
        document_id=document.doc_id,
        source_filename=document.source_filename,
        document_type=document.doc_type,
        section_heading=heading or document.doc_type,
        chunk_id=_chunk_id(document, strategy, index, text),
        chunking_strategy=strategy,
        entity_ids=document.entity_ids,
    )


def _fixed(document: SampleDocument) -> list[Chunk]:
    # Fixed-size mode is useful when documents have weak or inconsistent
    # headings, such as forms and tabular extracts.
    splitter = _token_splitter()
    return [
        _make_chunk(document, "fixed_size", document.doc_type, index, text.strip())
        for index, text in enumerate(splitter.split_text(document.extracted_text))
        if text.strip()
    ]


def _section_aware(document: SampleDocument) -> list[Chunk]:
    # Section-aware mode is preferred for policy clauses and knowledge-base
    # content because it keeps related statements together.
    splitter = _token_splitter()
    chunks: list[Chunk] = []
    for heading, section_text in _sections(document.extracted_text):
        section = f"{heading}\n{section_text}".strip()
        pieces = splitter.split_text(section) if splitter._length_function(section) > 300 else [section]
        for piece in pieces:
            if piece.strip():
                chunks.append(_make_chunk(document, "section_aware", heading, len(chunks), piece.strip()))
    return chunks


def sample_chunk_documents(directory: str | Path, strategy: str) -> list[Chunk]:
    if strategy not in {"fixed_size", "section_aware"}:
        raise ValueError("strategy must be 'fixed_size' or 'section_aware'")
    documents = load_sample_documents(directory)
    chunks: list[Chunk] = []
    for document in documents:
        # Images and unreadable files remain in the corpus for inspection but
        # are not sent to text retrieval without OCR/vision extraction.
        if not document.extraction_succeeded:
            continue
        chunks.extend(_fixed(document) if strategy == "fixed_size" else _section_aware(document))
    return chunks
