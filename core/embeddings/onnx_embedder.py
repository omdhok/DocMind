"""
ONNX Runtime embedding engine.

Loads a MiniLM-style sentence embedding model that has been exported to
ONNX (see scripts/export_embeddings.py) and runs it through ONNX Runtime
using whatever execution provider ExecutionProviderManager selects
(QNN > DirectML > CPU, whichever is genuinely available).

This module does NOT download or convert any model itself -- it only
loads files that already exist on disk. That keeps the runtime
dependency footprint small (onnxruntime + tokenizers, no torch needed)
and keeps model conversion (which does need torch/transformers) as a
one-time, separate step.
"""

from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from core.execution.provider_manager import ExecutionProviderManager, ProviderInfo
from config import ONNX_EMBEDDING_MODEL_DIR


DEFAULT_MODEL_DIR = ONNX_EMBEDDING_MODEL_DIR
EXPECTED_MODEL_FILE = "model.onnx"
EXPECTED_TOKENIZER_FILE = "tokenizer.json"
EXPECTED_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2's known output dimension


class EmbeddingModelNotFoundError(Exception):
    """
    Raised when the ONNX embedding model files are missing.

    This is an EXPECTED condition on a fresh checkout (model files are
    intentionally not committed to git -- see models/.gitkeep and the
    project README). The caller should catch this and show the user a
    clear instruction rather than a raw traceback.
    """


class ONNXEmbedder:
    """
    Wraps an ONNX Runtime session for a sentence-embedding model and
    exposes a simple encode(texts) -> np.ndarray interface.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / EXPECTED_MODEL_FILE
        self.tokenizer_path = self.model_dir / EXPECTED_TOKENIZER_FILE

        if not self.model_path.exists() or not self.tokenizer_path.exists():
            raise EmbeddingModelNotFoundError(
                f"Embedding model not found in '{self.model_dir}'.\n"
                f"Expected files:\n"
                f"  - {self.model_path}\n"
                f"  - {self.tokenizer_path}\n"
                f"Run: python scripts/export_embeddings.py to create them."
            )

        self.provider_manager = ExecutionProviderManager()
        self.provider_info: ProviderInfo = self.provider_manager.detect()

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=self.provider_manager.provider_list(),
        )

        # Record which provider ONNX Runtime actually picked for this
        # session -- this can differ from the requested priority list if
        # a provider silently fails to initialize for this specific model.
        self.active_provider = self.session.get_providers()[0]

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

        # Enable padding using whatever pad token this tokenizer actually
        # defines, instead of assuming "[PAD]"/id 0 (true for standard BERT
        # tokenizers like MiniLM's, but not guaranteed for every tokenizer).
        pad_token_id = self.tokenizer.token_to_id("[PAD]")
        if pad_token_id is not None:
            self.tokenizer.enable_padding(pad_id=pad_token_id, pad_token="[PAD]")
        else:
            # Fall back to ONNX Runtime-side padding via id 0, which is the
            # tokenizers library's own default and matches most WordPiece/
            # BPE vocabularies' reserved-token convention.
            self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=256)

        self._input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Encode a list of texts into L2-normalized embedding vectors.

        Args:
            texts: List of input strings.
            batch_size: Number of texts encoded per ONNX Runtime call.

        Returns:
            np.ndarray of shape (len(texts), EXPECTED_EMBEDDING_DIM), float32,
            each row L2-normalized (so dot product == cosine similarity).
        """
        if not texts:
            return np.zeros((0, EXPECTED_EMBEDDING_DIM), dtype=np.float32)

        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            all_embeddings.append(self._encode_batch(batch))

        return np.vstack(all_embeddings)

    def _encode_batch(self, batch: List[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(batch)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            onnx_inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, onnx_inputs)
        last_hidden_state = outputs[0]  # shape: (batch, seq_len, hidden_dim)

        pooled = _mean_pooling(last_hidden_state, attention_mask)
        normalized = _l2_normalize(pooled)
        return normalized.astype(np.float32)


def _mean_pooling(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Mean-pool token embeddings into a single sentence embedding,
    ignoring padded positions (the standard approach for MiniLM/BERT
    sentence-embedding models).
    """
    mask = attention_mask[..., np.newaxis].astype(np.float32)  # (batch, seq_len, 1)
    summed = np.sum(last_hidden_state * mask, axis=1)          # (batch, hidden_dim)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)  # avoid div-by-zero
    return summed / counts


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    return vectors / norms
