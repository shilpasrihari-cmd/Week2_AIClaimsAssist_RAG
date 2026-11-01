"""Extract text and basic metadata from supported policy documents.

This module intentionally stops at extraction and inspection. It does not chunk,
embed, index, or call an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
from docx import Document


SUPPORTED_SUFFIXES = {".pdf", ".docx"}


@dataclass
class DocumentExtraction:
    """Extraction result that retains the source filename and document ID."""

    source_filename: str
    document_id: str
    file_type: str
    extracted_text: str = ""
    extracted_character_count: int = 0
    page_count: int | None = None
    paragraph_count: int | None = None
    table_count: int | None = None
    extraction_succeeded: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def preview(self) -> str:
        """Return the first 300 characters of extracted text."""

        return self.extracted_text[:300]


def _base_result(path: Path) -> DocumentExtraction:
    suffix = path.suffix.lower()
    return DocumentExtraction(
        source_filename=path.name,
        document_id=path.stem,
        file_type=suffix.removeprefix(".").upper() or "UNKNOWN",
    )


def _finish(result: DocumentExtraction, text: str) -> DocumentExtraction:
    result.extracted_text = text
    result.extracted_character_count = len(text)
    if not text.strip():
        result.error = "Extracted text is empty."
        result.extraction_succeeded = False
    else:
        result.extraction_succeeded = True
    return result


def _extract_pdf(path: Path, result: DocumentExtraction) -> DocumentExtraction:
    with pymupdf.open(path) as pdf:
        result.page_count = pdf.page_count
        text = "\n".join(page.get_text() for page in pdf)
    return _finish(result, text)


def _extract_docx(path: Path, result: DocumentExtraction) -> DocumentExtraction:
    document = Document(path)
    result.paragraph_count = len(document.paragraphs)
    result.table_count = len(document.tables)

    sections: list[str] = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            sections.append("\t".join(cell.text for cell in row.cells))
    return _finish(result, "\n".join(sections))


def extract_document(file_path: str | Path) -> DocumentExtraction:
    """Extract a PDF or DOCX, returning a failure result instead of hiding errors."""

    path = Path(file_path)
    result = _base_result(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        result.error = f"Unsupported file type: {path.suffix or '(none)'}"
        return result
    if not path.is_file():
        result.error = "File does not exist or is not a regular file."
        return result

    try:
        if path.suffix.lower() == ".pdf":
            return _extract_pdf(path, result)
        return _extract_docx(path, result)
    except Exception as exc:  # Keep inspection of one bad document from stopping the batch.
        result.error = f"{type(exc).__name__}: {exc}"
        return result


def inspect_directory(directory: str | Path) -> list[DocumentExtraction]:
    """Inspect every supported document directly inside *directory*."""

    directory_path = Path(directory)
    paths = sorted(
        (path for path in directory_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: path.name.lower(),
    )
    return [extract_document(path) for path in paths]
