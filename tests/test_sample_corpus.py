from pathlib import Path

from ingestion.chunker import chunk_documents
from ingestion.sample_loader import load_sample_documents


SAMPLE = Path(__file__).parents[1] / "data" / "sample"


def test_sample_loader_preserves_manifest_ids_and_excludes_faces_from_text() -> None:
    documents = load_sample_documents(SAMPLE)

    policy = next(document for document in documents if document.doc_id == "DOC-C-1001-FNOL")
    face = next(document for document in documents if document.relative_path.startswith("faces/"))

    assert policy.source_filename == "C-1001-fnol.pdf"
    assert policy.policy_id == "MOT-0000002"
    assert policy.extraction_succeeded
    assert face.extraction_succeeded is False
    assert face.error == "Face image excluded from text retrieval."


def test_sample_chunks_have_stable_ids_and_source_metadata() -> None:
    first = chunk_documents(SAMPLE, "section_aware")
    second = chunk_documents(SAMPLE, "section_aware")

    assert first
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.document_id and chunk.source_filename for chunk in first)
    assert any(chunk.document_id == "DOC-C-1001-FNOL" for chunk in first)
    assert any(chunk.entity_ids for chunk in first)
