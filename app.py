"""Streamlit UI for the AIClaimsAssist grounded RAG pipeline."""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any

import streamlit as st

from generation.rag_pipeline import RAGPipeline
from ingestion.sample_loader import SampleDocument, load_sample_documents
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker


PROJECT_ROOT = Path(__file__).resolve().parent
RAG_DATA_ROOT = PROJECT_ROOT / "data" / "sample"
_POLICY_ID_PATTERN = re.compile(r"\b(?:MOT|COM)-\d+\b", re.IGNORECASE)


def available_policy_options() -> tuple[str, ...]:
    policy_ids = {
        match.group(0).upper()
        for path in (RAG_DATA_ROOT / "docs" / "policy").rglob("*")
        if path.is_file()
        for match in _POLICY_ID_PATTERN.finditer(path.name)
    }
    return ("Select a policy", *sorted(policy_ids))


POLICY_OPTIONS = available_policy_options()
MODE_OPTIONS = {
    "Hybrid": "hybrid",
    "Semantic only": "semantic_only",
    "Keyword only": "keyword_only",
}
STRATEGY_OPTIONS = {
    "Section-aware": "section_aware",
    "Fixed-size": "fixed_size",
}
logger = logging.getLogger(__name__)
_INTERNAL_NOTICE_MARKERS = (
    "unrelated coverage statement was removed",
    "unsupported citation references",
    "no valid policy citations were returned",
)


def user_visible_missing_information(messages: list[str]) -> list[str]:
    visible: list[str] = []
    suppressed = 0
    for message in messages:
        if any(marker in message.lower() for marker in _INTERNAL_NOTICE_MARKERS):
            suppressed += 1
        else:
            visible.append(message)
    if suppressed:
        logger.info("suppressed_internal_validation_events count=%d", suppressed)
    return visible


def inject_background() -> None:
    background_path = PROJECT_ROOT / "assets" / "aiclaimsassist-background.png"
    if not background_path.exists():
        return
    encoded_image = base64.b64encode(background_path.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background:
                linear-gradient(rgba(3, 12, 28, 0.78), rgba(3, 12, 28, 0.86)),
                url("data:image/png;base64,{encoded_image}") center / cover fixed no-repeat;
        }}
        [data-testid="stHeader"] {{ background: rgba(0, 0, 0, 0); }}
        [data-testid="stSidebar"] {{ background: rgba(3, 12, 28, 0.86); }}
        [data-testid="stChatMessage"] {{ background: rgba(3, 12, 28, 0.52); border-radius: 0.75rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading retrieval and reranking models...")
def load_rag_pipeline() -> RAGPipeline:
    retriever = HybridRetriever.from_directory(RAG_DATA_ROOT)
    reranker = CrossEncoderReranker()
    return RAGPipeline(retriever, reranker)


@st.cache_data(show_spinner="Loading sample documents...")
def load_document_inventory() -> tuple[SampleDocument, ...]:
    return tuple(
        document
        for document in load_sample_documents(RAG_DATA_ROOT)
        if not document.relative_path.lower().startswith("faces/")
    )


def citation_source(result: dict[str, Any], chunk_id: str) -> dict[str, Any] | None:
    return next(
        (source for source in result.get("retrieved_sources", []) if source.get("chunk_id") == chunk_id),
        None,
    )


def document_file_uri(document_id: str, source_filename: str) -> str | None:
    for document in load_document_inventory():
        if document.doc_id == document_id or document.source_filename == source_filename:
            return (RAG_DATA_ROOT / document.relative_path).resolve().as_uri()
    return None


def documents_for_policy(
    documents: tuple[SampleDocument, ...], policy_id: str
) -> tuple[SampleDocument, ...]:
    """Filter the document browser by the selected policy, when one is selected."""

    if not policy_id or policy_id == "Select a policy":
        return documents
    return tuple(document for document in documents if document.policy_id == policy_id)


def render_result(result: dict[str, Any]) -> None:
    st.markdown(result.get("answer", ""))

    missing = user_visible_missing_information(result.get("missing_information", []))
    if missing:
        st.warning(" ".join(missing))

    metadata = {
        "Model": result.get("model_name", "unknown"),
        "Latency": f"{result.get('latency', 0.0):.2f} seconds",
        "Retrieval": result.get("retrieval_mode", "unknown"),
        "Chunking": result.get("chunking_strategy", "unknown"),
        "Passages": result.get("number_of_passages", 0),
    }
    st.caption(" · ".join(f"{key}: {value}" for key, value in metadata.items()))

    citations = result.get("citations", [])
    with st.expander(f"Validated citations ({len(citations)})"):
        if not citations:
            st.info("No validated citations were returned.")
        for citation in citations:
            document_uri = document_file_uri(citation["chunk_id"], citation["source_filename"])
            if document_uri:
                st.markdown(f"[Open source document: {citation['source_filename']}]({document_uri})")
            st.markdown(
                f"**[{citation['source_label']}]** · **{citation['source_filename']}** · "
                f"{citation['policy_id']} · {citation['section_heading']}  \n"
                f"Chunk: `{citation['chunk_id']}`"
            )

    if False:
        for source in result.get("retrieved_sources", []):
            st.markdown(
                f"**{source['source_filename']}** · {source['policy_id']} · "
                f"{source['section_heading']}  \nChunk: `{source['chunk_id']}`"
            )
            st.code(source["text"], language="text")


def render_document_browser(documents: tuple[SampleDocument, ...]) -> None:
    st.subheader("Available documents")
    if not documents:
        st.warning("No sample documents were found.")
        return

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.caption(f"{len(documents)} documents available")
        table = st.dataframe(
            [
                {
                    "Filename": document.source_filename,
                    "Document ID": document.doc_id,
                    "Type": document.doc_type,
                }
                for document in documents
            ],
            hide_index=True,
            width="stretch",
            key="document_table",
            on_select="rerun",
            selection_mode="single-row",
        )

    selected_rows = table.selection.rows
    selected = documents[selected_rows[0] if selected_rows else 0]
    with right:
        st.markdown(f"### {selected.source_filename}")
        details = {
            "Document ID": selected.doc_id,
            "Document type": selected.doc_type,
            "Format": selected.file_format,
            "Path": selected.relative_path,
            "Characters": len(selected.extracted_text),
            "Extraction": "Succeeded" if selected.extraction_succeeded else "Failed",
            "Entity IDs": ", ".join(selected.entity_ids) or "none",
        }
        for label, value in details.items():
            st.markdown(f"**{label}:** {value}")
        preview_path = RAG_DATA_ROOT / selected.relative_path
        image_formats = {"jpg", "jpeg", "png", "gif", "bmp", "webp", "tif", "tiff"}
        if selected.file_format.lower() in image_formats and preview_path.exists():
            st.image(str(preview_path), caption=selected.source_filename, use_container_width=True)
        elif selected.error:
            st.error(selected.error)
        else:
            st.text_area(
                "Extracted text preview",
                selected.extracted_text,
                height=520,
                disabled=True,
            )


def main() -> None:
    st.set_page_config(page_title="AIClaimsAssist", page_icon="🛡️", layout="wide")
    inject_background()
    st.title("AIClaimsAssist")
    st.subheader("Auto Insurance Claims Knowledge Assistant")

    with st.sidebar:
        st.header("Policy and retrieval")
        policy_options = available_policy_options()
        selected_policy = st.selectbox("Select policy (required)", policy_options)
        retrieval_label = st.selectbox("Retrieval mode", tuple(MODE_OPTIONS))
        strategy_label = st.selectbox("Chunking strategy", tuple(STRATEGY_OPTIONS))
        reranking_enabled = st.checkbox("Enable cross-encoder reranking", value=True)

    documents_tab, chat_tab = st.tabs(["Documents", "Ask a question"])
    with documents_tab:
        documents = documents_for_policy(load_document_inventory(), selected_policy)
        render_document_browser(documents)

    with chat_tab:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if st.button("New Chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and message.get("result"):
                    render_result(message["result"])
                else:
                    st.markdown(message["content"])

        prompt = st.chat_input("Ask a question about the selected policy")
        if prompt is None:
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if selected_policy == policy_options[0]:
                st.warning("Select a policy before asking a policy question.")
                result = {
                    "answer": "",
                    "citations": [],
                    "missing_information": [],
                    "retrieved_sources": [],
                    "model_name": "not run",
                    "latency": 0.0,
                    "retrieval_mode": MODE_OPTIONS[retrieval_label],
                    "chunking_strategy": STRATEGY_OPTIONS[strategy_label],
                    "number_of_passages": 0,
                }
            else:
                try:
                    pipeline = load_rag_pipeline()
                    result = pipeline.answer(
                        prompt,
                        policy_id=selected_policy,
                        retrieval_mode=MODE_OPTIONS[retrieval_label],
                        strategy=STRATEGY_OPTIONS[strategy_label],
                        reranking_enabled=reranking_enabled,
                    )
                    render_result(result)
                except Exception:
                    st.error("The RAG services are unavailable. Check configured credentials and connectivity.")
                    result = {
                        "answer": "The RAG services are unavailable.",
                        "citations": [],
                        "missing_information": ["The request could not be completed."],
                        "retrieved_sources": [],
                        "model_name": "unavailable",
                        "latency": 0.0,
                        "retrieval_mode": MODE_OPTIONS[retrieval_label],
                        "chunking_strategy": STRATEGY_OPTIONS[strategy_label],
                        "number_of_passages": 0,
                    }

        st.session_state.messages.append({"role": "assistant", "content": result["answer"], "result": result})


if __name__ == "__main__":
    main()
