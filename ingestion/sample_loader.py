"""Load text-bearing documents from the copied synthetic sample corpus."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docx import Document
import openpyxl

from .document_loader import extract_document


TEXT_SUFFIXES = {".csv", ".docx", ".json", ".jsonl", ".md", ".pdf", ".tsv", ".xlsx"}
POLICY_ID_PREFIXES = ("MOT-", "COM-")


@dataclass(frozen=True)
class SampleDocument:
    """A sample document with manifest identity and extracted text."""

    doc_id: str
    doc_type: str
    source_filename: str
    relative_path: str
    file_format: str
    entity_ids: tuple[str, ...]
    extracted_text: str
    extraction_succeeded: bool
    is_scanned: bool = False
    error: str | None = None

    @property
    def policy_id(self) -> str:
        return next((entity for entity in self.entity_ids if entity.startswith(POLICY_ID_PREFIXES)), "")


def _read_tabular(path: Path, delimiter: str) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(handle, delimiter=delimiter)
        return "\n".join("\t".join(cell.strip() for cell in row) for row in rows)


def _read_workbook(path: Path) -> str:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sections: list[str] = []
    try:
        for worksheet in workbook.worksheets:
            sections.append(f"Sheet: {worksheet.title}")
            for row in worksheet.iter_rows(values_only=True):
                values = ["" if value is None else str(value) for value in row]
                if any(value.strip() for value in values):
                    sections.append("\t".join(values))
    finally:
        workbook.close()
    return "\n".join(sections)


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".json", ".jsonl"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".csv":
        return _read_tabular(path, ",")
    if suffix == ".tsv":
        return _read_tabular(path, "\t")
    if suffix == ".xlsx":
        return _read_workbook(path)
    if suffix in {".docx", ".pdf"}:
        extracted = extract_document(path)
        if not extracted.extraction_succeeded:
            raise ValueError(extracted.error or "Text extraction failed")
        return extracted.extracted_text
    raise ValueError(f"Unsupported sample format: {suffix}")


def _document_from_entry(root: Path, entry: dict[str, Any]) -> SampleDocument:
    relative_path = str(entry["path"]).replace("/", "\\")
    path = root / Path(relative_path)
    file_format = str(entry.get("format", path.suffix.removeprefix("."))).lower()
    entity_ids = tuple(str(value) for value in entry.get("entity_ids", []))
    if relative_path.lower().split("\\", maxsplit=1)[0] == "faces":
        return SampleDocument(
            doc_id=str(entry["doc_id"]),
            doc_type=str(entry.get("doc_type", "image")),
            source_filename=path.name,
            relative_path=str(entry["path"]),
            file_format=file_format,
            entity_ids=entity_ids,
            extracted_text="",
            extraction_succeeded=False,
            is_scanned=bool(entry.get("is_scanned", False)),
            error="Face image excluded from text retrieval.",
        )
    if file_format not in {suffix.removeprefix(".") for suffix in TEXT_SUFFIXES}:
        return SampleDocument(
            doc_id=str(entry["doc_id"]),
            doc_type=str(entry.get("doc_type", "image")),
            source_filename=path.name,
            relative_path=str(entry["path"]),
            file_format=file_format,
            entity_ids=entity_ids,
            extracted_text="",
            extraction_succeeded=False,
            is_scanned=bool(entry.get("is_scanned", False)),
            error="Image-only document is preserved but not text-extracted.",
        )
    try:
        text = _read_text(path)
        return SampleDocument(
            doc_id=str(entry["doc_id"]),
            doc_type=str(entry.get("doc_type", "document")),
            source_filename=path.name,
            relative_path=str(entry["path"]),
            file_format=file_format,
            entity_ids=entity_ids,
            extracted_text=text,
            extraction_succeeded=bool(text.strip()),
            is_scanned=bool(entry.get("is_scanned", False)),
            error=None if text.strip() else "Extracted text is empty.",
        )
    except Exception as exc:
        return SampleDocument(
            doc_id=str(entry["doc_id"]),
            doc_type=str(entry.get("doc_type", "document")),
            source_filename=path.name,
            relative_path=str(entry["path"]),
            file_format=file_format,
            entity_ids=entity_ids,
            extracted_text="",
            extraction_succeeded=False,
            is_scanned=bool(entry.get("is_scanned", False)),
            error=f"{type(exc).__name__}: {exc}",
        )


def _root_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name == "manifest.json":
            continue
        relative_path = path.relative_to(root).as_posix()
        entries.append(
            {
                "doc_id": f"ROOT-{path.stem.upper()}",
                "doc_type": "corpus_metadata",
                "entity_ids": [],
                "format": path.suffix.removeprefix("."),
                "path": relative_path,
            }
        )
    return entries


def load_sample_documents(sample_root: str | Path) -> list[SampleDocument]:
    """Load manifest documents plus text-bearing corpus metadata files."""

    root = Path(sample_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entries = list(manifest.get("documents", []))
    manifest_paths = {str(entry["path"]).replace("/", "\\") for entry in entries}
    entries.extend(
        entry
        for entry in _root_entries(root)
        if str(entry["path"]).replace("/", "\\") not in manifest_paths
    )
    return [_document_from_entry(root, entry) for entry in sorted(entries, key=lambda item: str(item["path"]).lower())]
