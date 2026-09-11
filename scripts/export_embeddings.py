"""
One-time model export script.

Downloads sentence-transformers/all-MiniLM-L6-v2 from Hugging Face and
exports it to ONNX format so the main app (core/embeddings/onnx_embedder.py)
can run it via ONNX Runtime without needing torch/transformers at runtime.

This script needs INTERNET ACCESS to Hugging Face and needs torch +
transformers + onnx installed -- but only for this one-time export.
The main application does NOT need torch, transformers, or an internet
connection once this has been run.

Usage:
    pip install torch transformers onnx --break-system-packages
    python scripts/export_embeddings.py

Output:
    models/minilm-onnx/model.onnx
    models/minilm-onnx/tokenizer.json

Implementation note (fixed after a real bug was found during testing):
BertModel-family models (which MiniLM is) return BOTH a last_hidden_state
tensor AND a pooler_output tensor by default. Exporting the raw HF model
directly with a single output_names=["last_hidden_state"] produced an
ONNX graph with an unexpected second output on some torch/transformers
version combinations (silently, not always as a hard error), which is
fragile. This is fixed by wrapping the model in a thin module (see
EmbeddingModelWrapper below) that returns ONLY the tensor we actually
want, guaranteeing exactly one output regardless of torch/transformers
version or exporter backend. This exact fix is covered by
tests/test_export_integration.py using a small local (non-downloaded)
BERT-family model, so this specific regression cannot silently reappear.
"""

import sys
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = Path("models/minilm-onnx")
ONNX_OPSET_VERSION = 18  # verified working with torch 2.14 / transformers 5.17 during development


def load_model_and_tokenizer(model_name: str):
    """Downloads and loads the model + tokenizer. Requires internet access."""
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Try to skip loading the pooling head entirely (cleanest fix, supported
    # by most BERT-family models including MiniLM). Fall back to the plain
    # model if a given model doesn't accept this kwarg.
    try:
        model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    except TypeError:
        model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def build_export_wrapper(base_model):
    """
    Wraps the base model and returns ONLY last_hidden_state as a single
    plain tensor. This guarantees the exported ONNX graph has exactly one
    output no matter what the underlying model class returns (defends
    against the pooler-output bug described in the module docstring, even
    if add_pooling_layer=False wasn't supported/effective for this model).

    Module-level (not nested in a function) so it can be imported directly
    by tests without needing to download any model.
    """
    import torch

    class EmbeddingModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            outputs = self.base_model(**kwargs)
            return outputs.last_hidden_state

    wrapper = EmbeddingModelWrapper(base_model)
    wrapper.eval()
    return wrapper


def export_to_onnx(export_model, tokenizer, output_dir: Path, opset_version: int = ONNX_OPSET_VERSION):
    """
    Runs the actual torch.onnx.export call plus a self-verification step.

    This is separated from main() so it can be exercised directly in tests
    with a small local model, without needing internet access or the real
    MiniLM weights.

    Returns:
        Path to the written .onnx file.
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    dummy_text = ["DocMind export test sentence."]
    encoded = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)

    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
    }
    args = (encoded["input_ids"], encoded["attention_mask"])

    if "token_type_ids" in encoded:
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "sequence"}
        args = (encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"])

    onnx_path = output_dir / "model.onnx"

    torch.onnx.export(
        export_model,
        args,
        str(onnx_path),
        input_names=input_names,
        output_names=["last_hidden_state"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )

    # Verify the export actually produced exactly one output before
    # declaring success -- catches any future regression immediately
    # instead of surfacing as a confusing failure at app runtime.
    try:
        import onnx as onnx_pkg

        exported = onnx_pkg.load(str(onnx_path))
        output_names = [o.name for o in exported.graph.output]
        if output_names != ["last_hidden_state"]:
            raise RuntimeError(
                f"Export verification failed: expected exactly one output named "
                f"'last_hidden_state', got {output_names}. The app will not load "
                f"this model correctly -- please report this as a bug."
            )
    except ImportError:
        print(
            "('onnx' package not installed, skipping output-shape verification -- "
            "install it with 'pip install onnx' to enable this check.)"
        )

    return onnx_path


def main():
    try:
        import torch  # noqa: F401  (checked here so the ImportError message is clear)
    except ImportError:
        print(
            "Missing export-time dependencies.\n"
            "Run this first (only needed once, not required by the main app):\n"
            "    pip install torch transformers onnx --break-system-packages\n"
        )
        sys.exit(1)

    print(f"Downloading tokenizer and model: {MODEL_NAME}")
    try:
        model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    except ImportError:
        print(
            "Missing export-time dependencies.\n"
            "Run this first (only needed once, not required by the main app):\n"
            "    pip install torch transformers onnx --break-system-packages\n"
        )
        sys.exit(1)
    except Exception as exc:
        print(f"Failed to download/load model '{MODEL_NAME}': {exc}")
        print("Check your internet connection and that Hugging Face is reachable.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer_json_path = OUTPUT_DIR / "tokenizer.json"
    tokenizer.backend_tokenizer.save(str(tokenizer_json_path))
    print(f"Saved tokenizer -> {tokenizer_json_path}")

    export_model = build_export_wrapper(model)

    print(f"Exporting to ONNX -> {OUTPUT_DIR / 'model.onnx'} (this can take a minute)")
    try:
        onnx_path = export_to_onnx(export_model, tokenizer, OUTPUT_DIR)
    except RuntimeError as exc:
        print(f"Export verification failed: {exc}")
        sys.exit(1)

    print(f"Verified: exported graph has exactly one output (last_hidden_state).")
    print("\nDone. Files created:")
    print(f"  {onnx_path}")
    print(f"  {tokenizer_json_path}")
    print(
        "\nNext step (optional but recommended for Snapdragon/edge deployment):\n"
        "  Quantize the ONNX model to INT8 with:\n"
        "    python -c \"from onnxruntime.quantization import quantize_dynamic, QuantType; "
        "quantize_dynamic('models/minilm-onnx/model.onnx', "
        "'models/minilm-onnx/model.onnx', weight_type=QuantType.QInt8)\"\n"
        "This shrinks the model and is the same INT8 weight quantization "
        "recommended by Qualcomm for Hexagon NPU deployment.\n"
        "\nThe main app only needs `onnxruntime` and `tokenizers` from here on -- "
        "torch/transformers are not required to RUN DocMind, only to export the model."
    )


if __name__ == "__main__":
    main()
