from pathlib import Path

import pymupdf

from ingestion.chunker import (
    fixed_size_chunks,
    section_aware_chunks,
)
from ingestion.document_loader import extract_document


def _document(tmp_path: Path):
    path = tmp_path / "MOT-1234567-contract.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "Insuring Agreement\n" + "Coverage applies to an auto accident. " * 100 + "\nExclusions\nRacing is excluded.",
        )
        pdf.save(path)
    return extract_document(path)


def test_fixed_chunks_have_requested_metadata_and_stable_ids(tmp_path: Path) -> None:
    document = _document(tmp_path)
    first = fixed_size_chunks(document)
    second = fixed_size_chunks(document)

    assert first
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.policy_id == "MOT-1234567" for chunk in first)
    assert all(chunk.document_id == "MOT-1234567-contract" for chunk in first)
    assert all(chunk.source_filename == "MOT-1234567-contract.pdf" for chunk in first)
    assert all(chunk.document_type == "pdf" for chunk in first)
    assert all(chunk.chunking_strategy == "fixed_size" for chunk in first)
    assert all(chunk.text.strip() for chunk in first)


def test_section_aware_chunks_preserve_short_sections(tmp_path: Path) -> None:
    document = _document(tmp_path)
    chunks = section_aware_chunks(document)

    exclusions = [chunk for chunk in chunks if chunk.section_heading == "Exclusions"]
    assert exclusions
    assert any("Racing is excluded." in chunk.text for chunk in exclusions)
    assert all(chunk.section_heading for chunk in chunks)
    assert all(chunk.chunking_strategy == "section_aware" for chunk in chunks)


def test_empty_documents_produce_no_chunks(tmp_path: Path) -> None:
    path = tmp_path / "MOT-1234567-empty.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(path)
    document = extract_document(path)

    assert fixed_size_chunks(document) == []
    assert section_aware_chunks(document) == []


def test_com_policy_id_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "COM-7654321-contract.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Coverage\nCommercial vehicle coverage applies.")
        pdf.save(path)

    document = extract_document(path)
    chunks = section_aware_chunks(document)

    assert chunks
    assert all(chunk.policy_id == "COM-7654321" for chunk in chunks)
