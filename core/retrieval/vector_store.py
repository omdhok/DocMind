"""
Vector store module.

An in-memory FAISS index, rebuilt per uploaded document/session (not a
persistent database -- a hackathon/prototype document assistant doesn't
need one, and it keeps the system simple and avoids stale-index bugs).

Since embeddings from core/embeddings are already L2-normalized, we use
FAISS's IndexFlatIP (inner product), where inner product of normalized
vectors == cosine similarity. This avoids needing a separate cosine
index type.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import faiss
import numpy as np


@dataclass
class ChunkMetadata:
    """Metadata attached to each vector so search results are traceable."""
    chunk_id: int
    text: str
    page_number: int
    document_name: str


@dataclass
class SearchResult:
    metadata: ChunkMetadata
    score: float  # cosine similarity, higher is more relevant


class VectorStoreError(Exception):
    """Raised for FAISS/vector-store specific problems."""


class VectorStore:
    """
    Wraps a FAISS IndexFlatIP index plus a parallel metadata list.
    """

    def __init__(self, embedding_dim: int):
        if embedding_dim <= 0:
            raise VectorStoreError(f"Invalid embedding dimension: {embedding_dim}")
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self._metadata: List[ChunkMetadata] = []

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(self, embeddings: np.ndarray, metadata: List[ChunkMetadata]) -> None:
        """
        Add a batch of embeddings + their metadata to the index.

        Args:
            embeddings: np.ndarray of shape (n, embedding_dim), float32.
            metadata: list of ChunkMetadata, same length as embeddings.

        Raises:
            VectorStoreError: on shape/dimension mismatches.
        """
        if embeddings.size == 0:
            return

        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected (*, {self.embedding_dim}), "
                f"got {embeddings.shape}"
            )

        if len(metadata) != embeddings.shape[0]:
            raise VectorStoreError(
                f"Metadata count ({len(metadata)}) does not match "
                f"embedding count ({embeddings.shape[0]})"
            )

        self.index.add(embeddings.astype(np.float32))
        self._metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[SearchResult]:
        """
        Retrieve the top_k most similar chunks to a query embedding.

        Args:
            query_embedding: np.ndarray of shape (embedding_dim,) or (1, embedding_dim).
            top_k: number of results to return.

        Returns:
            List of SearchResult, ordered by descending similarity.
        """
        if self.size == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.embedding_dim:
            raise VectorStoreError(
                f"Query embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {query.shape[1]}"
            )

        k = min(top_k, self.size)
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(SearchResult(metadata=self._metadata[idx], score=float(score)))

        return results

    def clear(self) -> None:
        """Reset the index (used when a new document is uploaded)."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self._metadata = []
