"""
DocMind -- Streamlit application entrypoint.

Run with:
    streamlit run app.py

This file is intentionally "thin": it handles UI state, file upload, and
button wiring, and delegates all real work to core.pipeline.DocMindPipeline.
Every call into the pipeline is wrapped in try/except against the specific
error types the pipeline defines, so a user-facing, human-readable message
is shown instead of a raw Python traceback -- per the project's error
handling requirements.
"""

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # loads GROQ_API_KEY from .env if present; safe no-op if .env doesn't exist

from config import SUPPORTED_EXTENSIONS, DEFAULT_TOP_K
from core.embeddings.embedder import Embedder
from core.generation.local_llm import LocalLLM, LocalLLMUnavailableError
from core.generation.cloud_llm import CloudLLM, CloudLLMUnavailableError
from core.pipeline import DocMindPipeline, IngestionError
from core.embeddings.onnx_embedder import EmbeddingModelNotFoundError


st.set_page_config(page_title="DocMind - Private Document Intelligence", layout="wide")


# ---------------------------------------------------------------------------
# Cached resource loading (models load once per session, not per interaction)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_embedder():
    return Embedder()


@st.cache_resource(show_spinner=False)
def load_local_llm():
    return LocalLLM()


@st.cache_resource(show_spinner=False)
def load_cloud_llm():
    return CloudLLM()


def get_pipeline() -> DocMindPipeline:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = DocMindPipeline(
            embedder=load_embedder(),
            local_llm=load_local_llm(),
            cloud_llm=load_cloud_llm(),
        )
    return st.session_state.pipeline


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(pipeline: DocMindPipeline):
    with st.sidebar:
        st.title("DocMind")
        st.caption("Private Document Intelligence for Snapdragon PCs")

        st.divider()
        mode = st.radio(
            "Mode",
            options=["Private Offline Mode", "Cloud Comparison Mode"],
            index=0,
            help="Private Offline Mode never sends your document to the internet.",
        )
        st.session_state.use_cloud = mode == "Cloud Comparison Mode"

        st.divider()
        st.subheader("Model Status")

        if pipeline.embedder.is_ready:
            st.success("Embedding model: Ready (ONNX Runtime)")
            provider_info = pipeline.embedder.provider_info
            if provider_info:
                icon = "🔒" if not provider_info.is_accelerated else "⚡"
                st.caption(f"{icon} Execution provider: **{provider_info.label}**")
        else:
            st.error("Embedding model: Not set up")
            st.caption(pipeline.embedder.error_message)

        if pipeline.local_llm.is_ready:
            st.success("Local LLM: Ready (offline)")
        else:
            st.warning("Local LLM: Not set up")
            st.caption(pipeline.local_llm.error_message)

        if st.session_state.get("use_cloud"):
            if pipeline.cloud_llm and pipeline.cloud_llm.is_ready:
                st.info("Cloud LLM: Connected (Groq)")
            else:
                st.warning("Cloud LLM: Not configured")
                st.caption(pipeline.cloud_llm.error_message if pipeline.cloud_llm else "")

        st.divider()
        st.subheader("Privacy")
        if not st.session_state.get("use_cloud"):
            st.markdown(
                "🟢 **Private Offline Mode**\n\n"
                "- Documents stay on this device\n"
                "- No API key required\n"
                "- No document data is sent to the cloud"
            )
        else:
            st.markdown(
                "☁️ **Cloud Comparison Mode**\n\n"
                "- Document context is sent to the Groq API for this request\n"
                "- Used only for demo/comparison purposes"
            )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

def render_upload(pipeline: DocMindPipeline):
    st.header("Private Document Intelligence")
    uploaded_file = st.file_uploader(
        "Upload a document", type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
    )

    if uploaded_file is None:
        st.info("Upload a PDF, DOCX, or TXT file to get started.")
        return False

    current_name = st.session_state.get("current_document_name")
    if current_name == uploaded_file.name:
        return True  # already ingested this session

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            metrics = pipeline.ingest_document(tmp_path)
        except IngestionError as exc:
            st.error(f"Could not process document: {exc}")
            return False
        except EmbeddingModelNotFoundError as exc:
            st.error(str(exc))
            return False
        finally:
            # Best-effort cleanup of the temp file. Wrapped in try/except
            # because on Windows a file can briefly stay locked (e.g. by
            # antivirus scanning) right after being written -- this must
            # never crash the app even if deletion fails.
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    st.session_state.current_document_name = uploaded_file.name
    st.session_state.ingest_metrics = metrics
    st.success(
        f"Processed '{uploaded_file.name}' -- {metrics.chunk_count} chunks in "
        f"{metrics.document_processing_seconds + metrics.embedding_seconds:.2f}s"
    )
    return True


def render_ask_tab(pipeline: DocMindPipeline):
    question = st.text_input("Ask a question about the document")
    if st.button("Ask", type="primary") and question:
        with st.spinner("Thinking..."):
            try:
                answer, sources, metrics = pipeline.ask(
                    question, top_k=DEFAULT_TOP_K, use_cloud=st.session_state.get("use_cloud", False)
                )
            except (LocalLLMUnavailableError, CloudLLMUnavailableError) as exc:
                st.error(str(exc))
                return
            except IngestionError as exc:
                st.error(str(exc))
                return

        st.markdown("### Answer")
        st.write(answer)
        if sources:
            st.markdown("**Sources:** " + ", ".join(f"Page {p}" for p in sources))
        st.session_state.last_metrics = metrics


def render_summarize_tab(pipeline: DocMindPipeline):
    if st.button("Generate Summary", type="primary"):
        with st.spinner("Summarizing..."):
            try:
                summary, metrics = pipeline.summarize(use_cloud=st.session_state.get("use_cloud", False))
            except (LocalLLMUnavailableError, CloudLLMUnavailableError) as exc:
                st.error(str(exc))
                return
            except IngestionError as exc:
                st.error(str(exc))
                return
        st.markdown(summary)
        st.session_state.last_metrics = metrics


def render_insights_tab(pipeline: DocMindPipeline):
    if st.button("Extract Insights", type="primary"):
        with st.spinner("Extracting insights..."):
            try:
                insights, metrics = pipeline.extract_insights(use_cloud=st.session_state.get("use_cloud", False))
            except (LocalLLMUnavailableError, CloudLLMUnavailableError) as exc:
                st.error(str(exc))
                return
            except IngestionError as exc:
                st.error(str(exc))
                return
        st.markdown(insights)
        st.session_state.last_metrics = metrics


def render_performance_tab(pipeline: DocMindPipeline):
    ingest_metrics = st.session_state.get("ingest_metrics")
    last_metrics = st.session_state.get("last_metrics")

    st.subheader("Document Processing")
    if ingest_metrics:
        cols = st.columns(3)
        cols[0].metric("Processing time", f"{ingest_metrics.document_processing_seconds:.2f}s")
        cols[1].metric("Embedding time", f"{ingest_metrics.embedding_seconds:.2f}s")
        cols[2].metric("Chunks created", ingest_metrics.chunk_count)
    else:
        st.caption("Upload a document to see processing metrics.")

    st.subheader("Last Request")
    if last_metrics:
        cols = st.columns(4)
        cols[0].metric("Retrieval", f"{(last_metrics.retrieval_seconds or 0):.3f}s")
        cols[1].metric("Generation", f"{(last_metrics.generation_seconds or 0):.2f}s")
        cols[2].metric("Total", f"{(last_metrics.total_seconds or 0):.2f}s")
        tps = last_metrics.tokens_per_second
        cols[3].metric("Tokens/sec", f"{tps:.1f}" if tps else "N/A (cloud mode)")

        st.caption(f"Execution provider: **{last_metrics.active_execution_provider or 'N/A'}**")
        if last_metrics.model_name:
            st.caption(f"Model: **{last_metrics.model_name}**")
    else:
        st.caption("Ask a question, summarize, or extract insights to see request metrics.")


def main():
    pipeline = get_pipeline()
    render_sidebar(pipeline)

    document_ready = render_upload(pipeline)

    if document_ready:
        tabs = st.tabs(["Ask", "Summarize", "Insights", "Performance"])
        with tabs[0]:
            render_ask_tab(pipeline)
        with tabs[1]:
            render_summarize_tab(pipeline)
        with tabs[2]:
            render_insights_tab(pipeline)
        with tabs[3]:
            render_performance_tab(pipeline)


if __name__ == "__main__":
    main()
