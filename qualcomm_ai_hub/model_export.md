# Model Export for Snapdragon Deployment

This document describes the real steps to prepare DocMind's two models
(embedding model and LLM) for Snapdragon-optimized execution. It is
documentation, not automated code, for the parts that require a
Qualcomm AI Hub account or Snapdragon-specific tooling we cannot
verify from a generic development machine.

## 1. Embedding model (all-MiniLM-L6-v2) — implemented in this repo

This path is fully implemented and testable without any Qualcomm
account:

1. `python scripts/export_embeddings.py` — exports the model to
   `models/minilm-onnx/model.onnx` using `torch.onnx.export`, and saves
   the matching `tokenizer.json`.
2. (Recommended) Quantize to INT8, the same weight quantization
   Qualcomm recommends for Hexagon NPU deployment:
   ```
   python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; quantize_dynamic('models/minilm-onnx/model.onnx', 'models/minilm-onnx/model.onnx', weight_type=QuantType.QInt8)"
   ```
3. At runtime, `core/embeddings/onnx_embedder.py` loads this file through
   `onnxruntime.InferenceSession` with the provider priority
   `QNNExecutionProvider → DmlExecutionProvider → CPUExecutionProvider`.
   On a Snapdragon X Elite/X2 Elite Windows PC with `onnxruntime-qnn`
   installed and the QNN SDK runtime present, ONNX Runtime will route
   supported ops to the Hexagon NPU automatically — no application code
   changes needed, only the environment differs.

## 2. LLM (Llama-3.2-3B-Instruct) — two separate paths

**Path A: llama.cpp (implemented, used by default)**
`core/generation/local_llm.py` uses `llama-cpp-python` with a quantized
GGUF file. This runs on CPU (with SIMD optimizations) on any machine,
including Snapdragon Windows PCs, and is what makes "Private Offline
Mode" reliably demoable regardless of what hardware the judges' laptop
has. It does **not** go through the QNN Execution Provider — llama.cpp
and ONNX Runtime are separate runtimes.

**Path B: Qualcomm Genie/QNN (documented, optional, requires AI Hub access)**
Qualcomm publishes a documented workflow for compiling Llama models to
run through their "Genie" on-device generation runtime with QNN context
binaries, targeting Snapdragon X Elite specifically. At a high level,
this involves:

1. Installing `qai_hub` and `qai_hub_models` (`pip install qai-hub
   qai_hub_models`) and configuring an AI Hub API token from
   https://aihub.qualcomm.com.
2. Using AI Hub's documented Llama export tooling to quantize and
   compile a Llama-3.2-3B chat model, targeting the device profile
   `"Snapdragon X Elite CRD"`.
3. Following Qualcomm's Genie tutorial to package the compiled QNN
   context binaries into a runnable on-device generation pipeline.

We have **not** completed this path ourselves in this repository (it
requires an AI Hub account, real Snapdragon X Elite device-cloud access,
and meaningfully more setup time than fits a hackathon timeline
alongside everything else). We are documenting it honestly as a real,
Qualcomm-supported path rather than implementing a fake version of it.
If pursued, it would replace `core/generation/local_llm.py`'s backend
for Snapdragon-specific builds while keeping the same `LocalLLM`
interface, so the rest of the app (RAG pipeline, UI, prompts) would not
need to change.

## Summary table

| Model | Export path | Status | Runtime used |
|---|---|---|---|
| all-MiniLM-L6-v2 (embeddings) | `scripts/export_embeddings.py` → ONNX → INT8 | Implemented | ONNX Runtime (QNN/DirectML/CPU) |
| Llama-3.2-3B-Instruct (LLM) | GGUF via llama.cpp | Implemented (default) | llama-cpp-python (CPU) |
| Llama-3.2-3B-Instruct (LLM) | AI Hub + Genie/QNN | Documented, not implemented | Qualcomm Genie (Hexagon NPU) |
