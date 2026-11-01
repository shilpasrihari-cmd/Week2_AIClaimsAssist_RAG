from pathlib import Path

import pymupdf
from docx import Document

from ingestion.document_loader import extract_document, inspect_directory


def test_extract_pdf_preserves_filename_and_text(tmp_path: Path) -> None:
    path = tmp_path / "MOT-TEST-declarations.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Policy declaration text")
        pdf.save(path)

    result = extract_document(path)

    assert result.source_filename == path.name
    assert result.document_id == "MOT-TEST-declarations"
    assert result.file_type == "PDF"
    assert result.page_count == 1
    assert result.extraction_succeeded is True
    assert "Policy declaration text" in result.extracted_text


def test_extract_docx_includes_paragraphs_and_tables(tmp_path: Path) -> None:
    path = tmp_path / "MOT-TEST-contract.docx"
    document = Document()
    document.add_paragraph("Contract paragraph")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Coverage"
    table.cell(0, 1).text = "Comprehensive"
    document.save(path)

    result = extract_document(path)

    assert result.paragraph_count == 1
    assert result.table_count == 1
    assert "Contract paragraph" in result.extracted_text
    assert "Coverage\tComprehensive" in result.extracted_text
    assert result.extraction_succeeded is True


def test_unreadable_and_empty_documents_are_reported(tmp_path: Path) -> None:
    unreadable = tmp_path / "broken.pdf"
    unreadable.write_bytes(b"not a PDF")
    empty = tmp_path / "empty.docx"
    Document().save(empty)

    unreadable_result = extract_document(unreadable)
    empty_result = extract_document(empty)

    assert unreadable_result.extraction_succeeded is False
    assert unreadable_result.error
    assert empty_result.extraction_succeeded is False
    assert empty_result.extracted_character_count == 0
    assert empty_result.error == "Extracted text is empty."


def test_inspect_directory_only_reads_supported_files(tmp_path: Path) -> None:
    (tmp_path / "ignore.txt").write_text("ignore me", encoding="utf-8")
    path = tmp_path / "one.docx"
    document = Document()
    document.add_paragraph("text")
    document.save(path)

    results = inspect_directory(tmp_path)

    assert [result.source_filename for result in results] == ["one.docx"]
