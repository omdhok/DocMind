"""
Benchmark script.

Measures REAL latency for each pipeline stage on a sample document and
saves results to a JSON file. This is the evidence you attach to your
README / submission -- never hand-write these numbers.

Usage:
    python scripts/benchmark.py path/to/sample.pdf

Requires the embedding model and local LLM to already be set up
(see README.md "Model Setup").
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.embeddings.embedder import Embedder
from core.generation.local_llm import LocalLLM
from core.pipeline import DocMindPipeline, IngestionError
from core.generation.local_llm import LocalLLMUnavailableError
from core.embeddings.onnx_embedder import EmbeddingModelNotFoundError


BENCHMARK_QUESTION = "What is this document about?"
OUTPUT_PATH = Path("scripts/benchmark_results.json")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/benchmark.py <path_to_document>")
        sys.exit(1)

    doc_path = sys.argv[1]

    print("Loading embedder...")
    embedder = Embedder()
    if not embedder.is_ready:
        print(f"Embedding model not ready: {embedder.error_message}")
        sys.exit(1)

    print("Loading local LLM (this can take a while on first load)...")
    local_llm = LocalLLM()
    if not local_llm.is_ready:
        print(f"Local LLM not ready: {local_llm.error_message}")
        sys.exit(1)

    pipeline = DocMindPipeline(embedder=embedder, local_llm=local_llm)

    print(f"\nIngesting: {doc_path}")
    try:
        ingest_metrics = pipeline.ingest_document(doc_path)
    except IngestionError as exc:
        print(f"Ingestion failed: {exc}")
        sys.exit(1)

    print(f"Asking: '{BENCHMARK_QUESTION}'")
    try:
        answer, sources, ask_metrics = pipeline.ask(BENCHMARK_QUESTION)
    except LocalLLMUnavailableError as exc:
        print(f"Generation failed: {exc}")
        sys.exit(1)

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "document": str(doc_path),
        "execution_provider": embedder.active_provider,
        "chunk_count": ingest_metrics.chunk_count,
        "document_processing_seconds": ingest_metrics.document_processing_seconds,
        "embedding_seconds_ingest": ingest_metrics.embedding_seconds,
        "embedding_seconds_query": ask_metrics.embedding_seconds,
        "retrieval_seconds": ask_metrics.retrieval_seconds,
        "generation_seconds": ask_metrics.generation_seconds,
        "tokens_per_second": ask_metrics.tokens_per_second,
        "total_ask_seconds": ask_metrics.total_seconds,
    }

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    print("\n--- Benchmark Results ---")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
