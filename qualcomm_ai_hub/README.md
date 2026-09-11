# Qualcomm AI Hub Integration

This folder documents DocMind's real, verifiable relationship to the
Qualcomm AI stack. It is kept separate from the main application on
purpose: **the app must run and be fully usable without any Qualcomm AI
Hub account, credentials, or Snapdragon hardware.** This folder is
where the Snapdragon-specific evidence lives.

## What is actually implemented in the app (works everywhere)

- `core/execution/provider_manager.py` — a real `ExecutionProviderManager`
  that queries `onnxruntime.get_available_providers()` at runtime and
  selects, in priority order: `QNNExecutionProvider` → `DmlExecutionProvider`
  → `CPUExecutionProvider`. It only ever reports a provider that ONNX
  Runtime itself confirms is installed and initializes successfully.
- `core/embeddings/onnx_embedder.py` — runs the MiniLM embedding model
  through `onnxruntime.InferenceSession` using that provider list, and
  exposes `active_provider` so the UI shows the real, currently-active
  provider (never a hardcoded label).

On a normal Windows or Linux development machine (no Snapdragon, no QNN
SDK installed), this code correctly and honestly falls back to
`CPUExecutionProvider`. That is expected, not a bug.

## What activates ONLY on a real Snapdragon Windows PC

- `QNNExecutionProvider` requires the `onnxruntime-qnn` Python package
  (a Windows-only, Snapdragon-specific build of ONNX Runtime) plus the
  Qualcomm QNN/QAIRT SDK runtime libraries to be present on the system.
  Installing `onnxruntime` (the regular PyPI package) on a non-Snapdragon
  machine will never make `QNNExecutionProvider` appear — this is by
  design, not a bug in DocMind.
- `DmlExecutionProvider` (DirectML) is available on many Windows PCs
  with a DirectX 12 capable GPU/NPU, not just Snapdragon devices — it's
  a useful secondary acceleration path but is not itself a
  "Qualcomm-specific" claim.

## Files in this folder

- `model_export.md` — the real, documented steps to export/quantize
  models for Snapdragon (embedding model via ONNX quantization; LLM via
  Qualcomm's Genie runtime for Llama models).
- `profiling.md` — how to get real latency/memory numbers from actual
  Snapdragon silicon via Qualcomm AI Hub's hosted device cloud, even
  without owning the hardware.

## Honesty policy for this project

Every performance claim in this repository is labeled with its source:

| Label | Meaning |
|---|---|
| **Measured locally (CPU)** | Timed on the development machine, real numbers, no Snapdragon involved. |
| **Measured via AI Hub device cloud** | Real numbers from an actual physical Snapdragon device, obtained through Qualcomm AI Hub's hosted profiling job system, not run by us locally. |
| **Measured live on Snapdragon hardware** | Run and timed on a real Snapdragon Windows PC we had physical/remote access to. |
| **Documented capability (not yet measured)** | A real, Qualcomm-documented capability we have not personally verified with a number. |

We do not label anything "NPU-accelerated" unless it falls into one of
the first three categories for that specific claim.
