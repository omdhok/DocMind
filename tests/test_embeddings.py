"""
Tests for core/embeddings.

We do NOT download the real MiniLM model here (this environment may not
have Hugging Face access, and tests shouldn't depend on the internet).
Instead we build a tiny, structurally-valid ONNX model + tokenizer
on the fly that match the exact input/output contract ONNXEmbedder
expects (input_ids, attention_mask -> last_hidden_state). This proves
the pooling, normalization, provider-selection, and error-handling code
is correct, independent of which real model is eventually loaded.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnx
from onnx import helper, TensorProto
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from core.embeddings.onnx_embedder import (
    ONNXEmbedder,
    EmbeddingModelNotFoundError,
    _mean_pooling,
    _l2_normalize,
)
from core.embeddings.embedder import Embedder


HIDDEN_DIM = 8


def _build_dummy_onnx_model(path: Path, hidden_dim: int = HIDDEN_DIM):
    """
    Builds a tiny valid ONNX model with the same input/output contract as
    a real HF sentence-embedding model:
        inputs:  input_ids (int64), attention_mask (int64)
        output:  last_hidden_state (float32), shape (batch, seq, hidden_dim)

    The math itself is meaningless (not a real embedding model) -- this
    only exists to exercise the ONNXEmbedder plumbing.
    """
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])
    attention_mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"])
    output = helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["batch", "seq", hidden_dim])

    weight = helper.make_tensor(
        "weight", TensorProto.FLOAT, [hidden_dim], list(range(1, hidden_dim + 1))
    )

    cast_node = helper.make_node("Cast", ["input_ids"], ["float_ids"], to=TensorProto.FLOAT)
    unsqueeze_node = helper.make_node("Unsqueeze", ["float_ids"], ["unsq"], axes=[-1])
    mul_node = helper.make_node("Mul", ["unsq", "weight"], ["last_hidden_state"])

    graph = helper.make_graph(
        [cast_node, unsqueeze_node, mul_node],
        "dummy_embedding_model",
        [input_ids, attention_mask],
        [output],
        initializer=[weight],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _build_dummy_tokenizer(path: Path):
    """Builds a tiny valid tokenizer.json using a small fixed vocabulary."""
    vocab = {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3, "docmind": 4, "test": 5}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path))


@pytest.fixture
def dummy_model_dir(tmp_path):
    model_dir = tmp_path / "minilm-onnx"
    model_dir.mkdir()
    _build_dummy_onnx_model(model_dir / "model.onnx")
    _build_dummy_tokenizer(model_dir / "tokenizer.json")
    return model_dir


# ---------- error handling ----------

def test_missing_model_raises_clear_error(tmp_path):
    empty_dir = tmp_path / "does_not_exist"
    with pytest.raises(EmbeddingModelNotFoundError):
        ONNXEmbedder(model_dir=str(empty_dir))


def test_embedder_facade_reports_not_ready_when_model_missing(tmp_path):
    embedder = Embedder(model_dir=str(tmp_path / "missing"))
    assert embedder.is_ready is False
    assert "export_embeddings.py" in embedder.error_message


# ---------- pooling / normalization math ----------

def test_mean_pooling_ignores_padded_tokens():
    # batch of 1, seq_len 3, hidden_dim 2. Last token is padding.
    hidden_states = np.array([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]], dtype=np.float32)
    attention_mask = np.array([[1, 1, 0]], dtype=np.int64)
    pooled = _mean_pooling(hidden_states, attention_mask)
    # Average of [1,1] and [3,3] only -> [2,2], padded token must be ignored
    assert np.allclose(pooled, [[2.0, 2.0]])


def test_l2_normalize_produces_unit_vectors():
    vectors = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    normalized = _l2_normalize(vectors)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, [1.0, 1.0])


# ---------- end-to-end with the dummy ONNX model ----------

def test_onnx_embedder_encodes_with_correct_shape(dummy_model_dir):
    embedder = ONNXEmbedder(model_dir=str(dummy_model_dir))
    embeddings = embedder.encode(["hello world", "docmind test"])
    assert embeddings.shape == (2, HIDDEN_DIM)
    # Every row should be L2-normalized
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, [1.0, 1.0], atol=1e-5)


def test_onnx_embedder_reports_active_provider(dummy_model_dir):
    embedder = ONNXEmbedder(model_dir=str(dummy_model_dir))
    # On a machine with no QNN/DirectML installed, this must honestly be CPU.
    assert embedder.active_provider in embedder.provider_manager.available_providers()


def test_onnx_embedder_empty_input_returns_empty_array(dummy_model_dir):
    embedder = ONNXEmbedder(model_dir=str(dummy_model_dir))
    embeddings = embedder.encode([])
    assert embeddings.shape == (0, HIDDEN_DIM) or embeddings.shape[0] == 0
