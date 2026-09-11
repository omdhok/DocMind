"""
Tests for core/pipeline.py -- the RAG orchestration layer.

We use lightweight STUB embedder/LLM objects here (not the real ONNX
model or a real GGUF file) so these tests run instantly and without any
downloads, while still exercising the real ingestion -> embedding ->
retrieval -> generation control flow end to end.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from core.pipeline import DocMindPipeline, IngestionError
from core.generation.local_llm import LocalLLMUnavailableError
from core.generation.cloud_llm import CloudLLMUnavailableError


STUB_EMBED_DIM = 8


class StubEmbedder:
    """Deterministic fake embedder: same text always -> same vector."""

    is_ready = True
    error_message = None
    embedding_dim = STUB_EMBED_DIM
    active_provider = "CPUExecutionProvider"

    def encode(self, texts):
        vectors = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.random(STUB_EMBED_DIM).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


class NotReadyEmbedder(StubEmbedder):
    is_ready = False
    error_message = "Embedding model not found. Run scripts/export_embeddings.py"


class StubGenerationResult:
    def __init__(self, text, tokens_per_second=42.0):
        self.text = text
        self.tokens_per_second = tokens_per_second


class StubLocalLLM:
    is_ready = True
    error_message = None
    last_prompt = None

    def generate(self, prompt, max_tokens=512, **kwargs):
        self.last_prompt = prompt
        return StubGenerationResult(text=f"[stub answer based on {len(prompt)} char prompt]")


class NotReadyLocalLLM:
    is_ready = False
    error_message = "llama-cpp-python is not installed."

    def generate(self, *args, **kwargs):
        raise LocalLLMUnavailableError(self.error_message)


@pytest.fixture
def sample_txt(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "DocMind is a private document assistant.\n"
        "It answers questions using only the uploaded document.\n"
        "The quarterly revenue was 12 percent higher than last year.\n"
    )
    return str(path)


# ---------- ingestion + missing-document handling ----------

def test_ask_before_ingestion_raises_ingestion_error():
    pipeline = DocMindPipeline(embedder=StubEmbedder(), local_llm=StubLocalLLM())
    with pytest.raises(IngestionError):
        pipeline.ask("What is this about?")


def test_summarize_before_ingestion_raises_ingestion_error():
    pipeline = DocMindPipeline(embedder=StubEmbedder(), local_llm=StubLocalLLM())
    with pytest.raises(IngestionError):
        pipeline.summarize()


def test_ingest_unsupported_file_type_raises(tmp_path):
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("a,b,c")
    pipeline = DocMindPipeline(embedder=StubEmbedder(), local_llm=StubLocalLLM())
    with pytest.raises(IngestionError):
        pipeline.ingest_document(str(bad_file))


def test_ingest_empty_txt_raises(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    pipeline = DocMindPipeline(embedder=StubEmbedder(), local_llm=StubLocalLLM())
    with pytest.raises(IngestionError):
        pipeline.ingest_document(str(empty_file))


def test_ingest_when_embedder_not_ready_raises(sample_txt):
    pipeline = DocMindPipeline(embedder=NotReadyEmbedder(), local_llm=StubLocalLLM())
    with pytest.raises(IngestionError):
        pipeline.ingest_document(sample_txt)


# ---------- happy path: ingest + ask ----------

def test_ingest_then_ask_returns_answer_sources_and_metrics(sample_txt):
    pipeline = DocMindPipeline(embedder=StubEmbedder(), local_llm=StubLocalLLM())
    metrics = pipeline.ingest_document(sample_txt)
    assert metrics.chunk_count >= 1

    answer, sources, run_metrics = pipeline.ask("What was the revenue growth?")
    assert isinstance(answer, str) and len(answer) > 0
    assert isinstance(sources, list)
    assert run_metrics.active_execution_provider == "CPUExecutionProvider"
    assert run_metrics.generation_seconds is not None


def test_ask_prompt_contains_question_and_context(sample_txt):
    local_llm = StubLocalLLM()
    pipeline = DocMindPipeline(embedder=StubEmbedder(), local_llm=local_llm)
    pipeline.ingest_document(sample_txt)
    pipeline.ask("What was the revenue growth?")
    assert "What was the revenue growth?" in local_llm.last_prompt
    assert "revenue" in local_llm.last_prompt.lower()


def test_ask_empty_question_raises(sample_txt):
    pipeline = DocMindPipeline(embedder=StubEmbedder(), local_llm=StubLocalLLM())
    pipeline.ingest_document(sample_txt)
    with pytest.raises(IngestionError):
        pipeline.ask("   ")


# ---------- provider fallback surfaced through metrics ----------

def test_ask_reports_local_llm_not_ready(sample_txt):
    pipeline = DocMindPipeline(embedder=StubEmbedder(), local_llm=NotReadyLocalLLM())
    pipeline.ingest_document(sample_txt)
    with pytest.raises(LocalLLMUnavailableError):
        pipeline.ask("Anything?")


def test_ask_with_cloud_mode_but_no_cloud_llm_configured_raises(sample_txt):
    pipeline = DocMindPipeline(embedder=StubEmbedder(), local_llm=StubLocalLLM(), cloud_llm=None)
    pipeline.ingest_document(sample_txt)
    with pytest.raises(CloudLLMUnavailableError):
        pipeline.ask("Anything?", use_cloud=True)


# ---------- summarize / insights ----------

def test_summarize_returns_text_and_metrics(sample_txt):
    pipeline = DocMindPipeline(embedder=StubEmbedder(), local_llm=StubLocalLLM())
    pipeline.ingest_document(sample_txt)
    summary, metrics = pipeline.summarize()
    assert isinstance(summary, str) and len(summary) > 0
    assert metrics.chunk_count >= 1


def test_extract_insights_returns_text_and_metrics(sample_txt):
    pipeline = DocMindPipeline(embedder=StubEmbedder(), local_llm=StubLocalLLM())
    pipeline.ingest_document(sample_txt)
    insights, metrics = pipeline.extract_insights()
    assert isinstance(insights, str) and len(insights) > 0


# ---------- retrieval quality: relevance filtering + de-duplication ----------

def test_filter_and_dedupe_removes_duplicate_chunk_text():
    from core.retrieval.vector_store import ChunkMetadata, SearchResult

    dup_meta = ChunkMetadata(chunk_id=0, text="Same text here.", page_number=1, document_name="d.txt")
    dup_meta2 = ChunkMetadata(chunk_id=1, text="Same text here.", page_number=2, document_name="d.txt")
    results = [SearchResult(metadata=dup_meta, score=0.9), SearchResult(metadata=dup_meta2, score=0.8)]
    filtered = DocMindPipeline._filter_and_dedupe(results, top_k=4)
    assert len(filtered) == 1


def test_filter_and_dedupe_drops_low_relevance_results():
    from core.retrieval.vector_store import ChunkMetadata, SearchResult

    good = SearchResult(
        metadata=ChunkMetadata(chunk_id=0, text="Relevant text.", page_number=1, document_name="d.txt"),
        score=0.8,
    )
    weak = SearchResult(
        metadata=ChunkMetadata(chunk_id=1, text="Unrelated noise.", page_number=2, document_name="d.txt"),
        score=0.01,
    )
    filtered = DocMindPipeline._filter_and_dedupe([good, weak], top_k=4)
    assert len(filtered) == 1
    assert filtered[0].metadata.text == "Relevant text."


def test_filter_and_dedupe_keeps_best_match_even_below_threshold_if_no_others():
    from core.retrieval.vector_store import ChunkMetadata, SearchResult

    only_weak = SearchResult(
        metadata=ChunkMetadata(chunk_id=0, text="Weak match.", page_number=1, document_name="d.txt"),
        score=0.01,
    )
    filtered = DocMindPipeline._filter_and_dedupe([only_weak], top_k=4)
    assert len(filtered) == 1  # never returns an empty context when at least one result exists


def test_ask_reports_average_relevance_score(sample_txt):
    pipeline = DocMindPipeline(embedder=StubEmbedder(), local_llm=StubLocalLLM())
    pipeline.ingest_document(sample_txt)
    _, _, metrics = pipeline.ask("What was the revenue growth?")
    assert "avg_relevance_score" in metrics.extra


# ---------- large-document map-reduce summarization ----------

class CountingLocalLLM(StubLocalLLM):
    """Like StubLocalLLM, but counts how many times generate() was called."""

    def __init__(self):
        self.call_count = 0

    def generate(self, prompt, max_tokens=512, **kwargs):
        self.call_count += 1
        self.last_prompt = prompt
        return StubGenerationResult(text=f"[stub output #{self.call_count}]")


def test_summarize_small_document_uses_single_generation_call(sample_txt):
    llm = CountingLocalLLM()
    pipeline = DocMindPipeline(embedder=StubEmbedder(), local_llm=llm)
    pipeline.ingest_document(sample_txt)
    pipeline.summarize()
    assert llm.call_count == 1  # fits within budget -> no map-reduce needed


def test_summarize_large_document_uses_map_reduce(tmp_path):
    # Build a document well over MAX_SUMMARY_CONTEXT_CHARS (8000 chars).
    large_path = tmp_path / "large.txt"
    large_path.write_text(("Revenue and risk details for this section. " * 40 + "\n\n") * 15)

    llm = CountingLocalLLM()
    pipeline = DocMindPipeline(embedder=StubEmbedder(), local_llm=llm)
    pipeline.ingest_document(str(large_path))
    summary, metrics = pipeline.summarize()

    assert isinstance(summary, str) and len(summary) > 0
    # Map-reduce means more than one LLM call: several "map" segment
    # extractions plus one final "reduce" synthesis call.
    assert llm.call_count > 1



