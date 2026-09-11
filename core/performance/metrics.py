"""
Performance metrics module.

A small, dependency-free timer/collector used across the pipeline so the
Performance dashboard shows REAL measured values -- never hardcoded or
estimated numbers. Every stage (extraction, embedding, retrieval,
generation) records its own elapsed time via this class.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RunMetrics:
    """All measured timings + counts for a single question/summary/insights request."""
    document_processing_seconds: Optional[float] = None
    embedding_seconds: Optional[float] = None
    retrieval_seconds: Optional[float] = None
    generation_seconds: Optional[float] = None
    total_seconds: Optional[float] = None
    chunk_count: Optional[int] = None
    tokens_per_second: Optional[float] = None
    active_execution_provider: Optional[str] = None
    model_name: Optional[str] = None
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "document_processing_seconds": self.document_processing_seconds,
            "embedding_seconds": self.embedding_seconds,
            "retrieval_seconds": self.retrieval_seconds,
            "generation_seconds": self.generation_seconds,
            "total_seconds": self.total_seconds,
            "chunk_count": self.chunk_count,
            "tokens_per_second": self.tokens_per_second,
            "active_execution_provider": self.active_execution_provider,
            "model_name": self.model_name,
            **self.extra,
        }


class Timer:
    """
    Simple context-manager stopwatch.

    Usage:
        timer = Timer()
        with timer.measure("embedding"):
            do_embedding_work()
        print(timer.elapsed["embedding"])
    """

    def __init__(self):
        self.elapsed: Dict[str, float] = {}

    @contextmanager
    def measure(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.elapsed[label] = time.perf_counter() - start
