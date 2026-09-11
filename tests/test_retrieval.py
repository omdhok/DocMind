"""
Tests for core/retrieval/vector_store.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from core.retrieval.vector_store import VectorStore, ChunkMetadata, VectorStoreError


def make_metadata(n, doc_name="doc.pdf"):
    return [
        ChunkMetadata(chunk_id=i, text=f"chunk {i}", page_number=i + 1, document_name=doc_name)
        for i in range(n)
    ]


def test_search_on_empty_index_returns_empty_list():
    store = VectorStore(embedding_dim=4)
    results = store.search(np.array([1, 0, 0, 0], dtype="float32"))
    assert results == []


def test_add_and_search_returns_most_similar_first():
    store = VectorStore(embedding_dim=3)
    vectors = np.array(
        [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]], dtype="float32"
    )
    store.add(vectors, make_metadata(3))

    results = store.search(np.array([1, 0, 0], dtype="float32"), top_k=2)
    assert len(results) == 2
    # Closest vector should be chunk 0 (exact match), then chunk 1
    assert results[0].metadata.chunk_id == 0
    assert results[0].score >= results[1].score


def test_add_dimension_mismatch_raises():
    store = VectorStore(embedding_dim=4)
    wrong_dim_vectors = np.zeros((2, 3), dtype="float32")
    with pytest.raises(VectorStoreError):
        store.add(wrong_dim_vectors, make_metadata(2))


def test_add_metadata_count_mismatch_raises():
    store = VectorStore(embedding_dim=3)
    vectors = np.zeros((2, 3), dtype="float32")
    with pytest.raises(VectorStoreError):
        store.add(vectors, make_metadata(1))  # only 1 metadata for 2 vectors


def test_search_query_dimension_mismatch_raises():
    store = VectorStore(embedding_dim=3)
    store.add(np.zeros((1, 3), dtype="float32"), make_metadata(1))
    with pytest.raises(VectorStoreError):
        store.search(np.zeros(5, dtype="float32"))  # wrong dim query


def test_clear_resets_index():
    store = VectorStore(embedding_dim=3)
    store.add(np.zeros((2, 3), dtype="float32"), make_metadata(2))
    assert store.size == 2
    store.clear()
    assert store.size == 0


def test_top_k_larger_than_index_size_does_not_crash():
    store = VectorStore(embedding_dim=3)
    store.add(np.eye(2, 3, dtype="float32"), make_metadata(2))
    results = store.search(np.array([1, 0, 0], dtype="float32"), top_k=10)
    assert len(results) == 2  # capped at actual index size
