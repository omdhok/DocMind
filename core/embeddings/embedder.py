"""
Embedder facade.

This is the ONLY module the rest of the app (retrieval, RAG pipeline)
should import to get embeddings. It hides which concrete implementation
is used underneath, so if we ever add a second backend later, nothing
outside this file needs to change.

Currently backed by ONNXEmbedder (core/embeddings/onnx_embedder.py).
"""

from typing import List, Optional

import numpy as np

from core.embeddings.onnx_embedder import (
    ONNXEmbedder,
    EmbeddingModelNotFoundError,
    EXPECTED_EMBEDDING_DIM,
)
from core.execution.provider_manager import ProviderInfo


class Embedder:
    """
    High-level embedding interface with a clear "not ready" state instead
    of crashing the app when the ONNX model files haven't been set up yet.
    """

    def __init__(self, model_dir: str = "models/minilm-onnx"):
        self.model_dir = model_dir
        self._impl: Optional[ONNXEmbedder] = None
        self._error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        try:
            self._impl = ONNXEmbedder(model_dir=self.model_dir)
        except EmbeddingModelNotFoundError as exc:
            self._impl = None
            self._error = str(exc)

    @property
    def is_ready(self) -> bool:
        return self._impl is not None

    @property
    def error_message(self) -> Optional[str]:
        return self._error

    @property
    def provider_info(self) -> Optional[ProviderInfo]:
        return self._impl.provider_info if self._impl else None

    @property
    def active_provider(self) -> Optional[str]:
        return self._impl.active_provider if self._impl else None

    @property
    def embedding_dim(self) -> int:
        return EXPECTED_EMBEDDING_DIM

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.

        Raises:
            EmbeddingModelNotFoundError: if the model was never loaded.
                Callers (UI layer) should check `is_ready` first and show
                a friendly message instead of letting this propagate.
        """
        if self._impl is None:
            raise EmbeddingModelNotFoundError(
                self._error or "Embedding model is not loaded."
            )
        return self._impl.encode(texts)
