from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest


APP = Path(__file__).parents[1] / "app.py"


def test_streamlit_smoke_and_required_policy_selection() -> None:
    app = AppTest.from_file(str(APP)).run(timeout=30)

    assert app.title[0].value == "AIClaimsAssist"
    policy_selector = next(
        selector for selector in app.selectbox if "Select a policy" in selector.options
    )
    assert policy_selector.options == [
        "Select a policy",
        "COM-0000001",
        "COM-0000003",
        "COM-0000005",
        "MOT-0000002",
        "MOT-0000004",
    ]
    assert policy_selector.value == "Select a policy"


def test_citation_source_lookup() -> None:
    from app import citation_source

    result = {
        "retrieved_sources": [
            {"chunk_id": "one", "source_filename": "policy.pdf"},
        ]
    }

    assert citation_source(result, "one")["source_filename"] == "policy.pdf"
    assert citation_source(result, "missing") is None


def test_documents_for_policy_filters_by_policy_id() -> None:
    from app import documents_for_policy

    documents = (
        SimpleNamespace(source_filename="mot.pdf", policy_id="MOT-0000002"),
        SimpleNamespace(source_filename="com.pdf", policy_id="COM-0000001"),
    )

    filtered = documents_for_policy(documents, "MOT-0000002")

    assert [document.source_filename for document in filtered] == ["mot.pdf"]


def test_internal_validation_messages_are_logged_and_hidden(caplog) -> None:
    from app import user_visible_missing_information

    with caplog.at_level("INFO"):
        visible = user_visible_missing_information(
            [
                "An unrelated coverage statement was removed from the answer.",
                "The retrieved passages do not fully establish the requested information.",
            ]
        )

    assert visible == ["The retrieved passages do not fully establish the requested information."]
    assert "suppressed_internal_validation_events count=1" in caplog.text
