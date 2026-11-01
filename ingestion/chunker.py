"""Chunk extracted policy text with stable, metadata-rich results."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import truststore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .document_loader import DocumentExtraction, inspect_directory


# Use the host's trusted certificate store when tiktoken retrieves its
# vocabulary the first time. Subsequent runs use the local tiktoken cache.
truststore.inject_into_ssl()


# These values keep retrieval passages small enough for the LLM while retaining
# enough surrounding context to answer coverage and claims questions.
FIXED_CHUNK_SIZE = 300
FIXED_CHUNK_OVERLAP = 50
_POLICY_ID_PATTERN = re.compile(r"^((?:MOT|COM)-\d+)", re.IGNORECASE)
_SECTION_HEADINGS = {
    "insuring agreement",
    "definitions",
    "conditions",
    "exclusions",
    "policy declarations",
    "limits of insurance",
    "endorsements",
    "schedule of coverages",
    "coverage",
    "form code",
    "description",
}


@dataclass(frozen=True)
class Chunk:
    """A chunk and the metadata needed to trace it to a source document."""

    text: str
    policy_id: str
    document_id: str
    source_filename: str
    document_type: str
    section_heading: str
    chunk_id: str
    chunking_strategy: str
    entity_ids: tuple[str, ...] = ()

    @property
    def metadata(self) -> dict[str, str]:
        # Pinecone and citation validation use this metadata to trace a result
        # back to its source document and policy entity.
        return {
            "policy_id": self.policy_id,
            "document_id": self.document_id,
            "source_filename": self.source_filename,
            "document_type": self.document_type,
            "section_heading": self.section_heading,
            "chunk_id": self.chunk_id,
            "chunking_strategy": self.chunking_strategy,
            "entity_ids": ",".join(self.entity_ids),
        }


def _policy_id(document: DocumentExtraction) -> str:
    # Policy IDs are extracted from the source document ID so policy filtering
    # remains reliable even when a query does not repeat the policy number.
    match = _POLICY_ID_PATTERN.match(document.document_id)
    return match.group(1).upper() if match else document.document_id


def _chunk_id(document: DocumentExtraction, strategy: str, index: int, text: str) -> str:
    # Include a digest so an accidental change in chunk content cannot silently
    # reuse an old ID while keeping IDs stable for unchanged inputs.
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{document.document_id}:{strategy}:{index:04d}:{digest}"


def _metadata(document: DocumentExtraction, strategy: str, heading: str, index: int, text: str) -> Chunk:
    # Keep all chunk construction in one place so fixed-size and section-aware
    # retrieval produce the same metadata contract.
    return Chunk(
        text=text,
        policy_id=_policy_id(document),
        document_id=document.document_id,
        source_filename=document.source_filename,
        document_type=document.file_type.lower(),
        section_heading=heading or "Document",
        chunk_id=_chunk_id(document, strategy, index, text),
        chunking_strategy=strategy,
    )


def _token_splitter() -> RecursiveCharacterTextSplitter:
    # tiktoken makes the requested chunk size token-aware instead of relying on
    # character counts, which vary significantly across document formats.
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=FIXED_CHUNK_SIZE,
        chunk_overlap=FIXED_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        add_start_index=True,
    )


def _heading_at(text: str, character_index: int) -> str:
    """Find the nearest explicit section heading before a chunk start."""

    heading = "Document"
    for line in text[:character_index].splitlines():
        normalized = line.strip().rstrip(":")
        if normalized.lower() in _SECTION_HEADINGS:
            heading = normalized
    return heading


def fixed_size_chunks(document: DocumentExtraction) -> list[Chunk]:
    """Create approximately 300-token chunks with a 50-token overlap."""

    if not document.extracted_text.strip():
        return []
    splitter = _token_splitter()
    # LangChain records character offsets so each fixed chunk can inherit the
    # nearest heading for more useful citations and debugging.
    langchain_chunks = splitter.create_documents([document.extracted_text])
    chunks: list[Chunk] = []
    for index, langchain_chunk in enumerate(langchain_chunks):
        text = langchain_chunk.page_content.strip()
        if not text:
            continue
        start = int(langchain_chunk.metadata.get("start_index", 0))
        chunks.append(_metadata(document, "fixed_size", _heading_at(document.extracted_text, start), index, text))
    return chunks


def _is_heading(line: str) -> bool:
    # Headings are deliberately conservative; false positives could split a
    # short coverage clause and weaken the evidence shown to the answer model.
    normalized = line.strip().rstrip(":")
    if not normalized:
        return False
    lowered = normalized.lower()
    if lowered in _SECTION_HEADINGS:
        return True
    return lowered.startswith(("motor policy", "endorsement schedule", "coverage schedule"))


def _sections(text: str) -> Iterable[tuple[str, str]]:
    # Convert extracted lines into heading/body sections before token splitting.
    # This preserves compact provisions whenever they fit within one chunk.
    heading = "Document"
    body: list[str] = []
    for line in text.splitlines():
        if _is_heading(line):
            if any(part.strip() for part in body):
                yield heading, "\n".join(body).strip()
            heading = line.strip().rstrip(":")
            body = []
        else:
            body.append(line)
    if any(part.strip() for part in body):
        yield heading, "\n".join(body).strip()


def section_aware_chunks(document: DocumentExtraction) -> list[Chunk]:
    """Keep sections intact where possible and split only oversized sections."""

    if not document.extracted_text.strip():
        return []
    splitter = _token_splitter()
    # Short sections stay intact; only oversized sections use the token-aware
    # splitter, while each resulting piece retains the section heading.
    chunks: list[Chunk] = []
    for heading, section_text in _sections(document.extracted_text):
        section = f"{heading}\n{section_text}".strip()
        pieces = splitter.split_text(section) if splitter._length_function(section) > FIXED_CHUNK_SIZE else [section]
        for piece in pieces:
            text = piece.strip()
            if text:
                chunks.append(_metadata(document, "section_aware", heading, len(chunks), text))
    return chunks


def chunk_documents(directory: str | Path, strategy: str) -> list[Chunk]:
    """Chunk every supported document in deterministic filename order."""

    directory_path = Path(directory)
    # The sample corpus has a manifest and nested document folders, so it uses
    # the manifest-aware adapter instead of the original policy-folder loader.
    if (directory_path / "manifest.json").is_file():
        from .sample_chunker import sample_chunk_documents

        return sample_chunk_documents(directory_path, strategy)
    documents = inspect_directory(directory)
    chunker = {"fixed_size": fixed_size_chunks, "section_aware": section_aware_chunks}.get(strategy)
    if chunker is None:
        raise ValueError("strategy must be 'fixed_size' or 'section_aware'")
    return [chunk for document in documents for chunk in chunker(document)]
