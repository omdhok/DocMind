# DocMind

**Private On-Device Document Intelligence for Snapdragon PCs**

> Upload a document, ask anything, and get AI-powered answers without sending your private documents to the cloud.

## Evolution During the Challenge

DocMind began as a small existing project (an AI PDF Q&A assistant built
with Streamlit, pypdf, and the Groq cloud API) and was substantially
transformed during the Snapdragon AI Lab Build & Present Challenge 2026
submission period. Only features that are actually implemented in this
repository are listed below — nothing here is aspirational.

**Before (original project):**
- Cloud-only AI PDF Q&A assistant
- PDF-only input (via pypdf)
- Every question sent document text to a cloud LLM (Groq, Llama 3.3 70B)
- No local inference, no local embeddings, no vector retrieval
- No execution-provider awareness, no Snapdragon/NPU consideration
- No citations, no summarization, no structured insights
- No performance metrics, no automated tests

**After (DocMind, this submission):**
- Multi-format ingestion: PDF, DOCX, and TXT
- Local ONNX Runtime embedding pipeline with automatic execution-provider
  detection and fallback (QNN → DirectML → CPU), never hardcoded
- Local FAISS vector retrieval with a relevance-score floor and
  duplicate-chunk filtering (`core/pipeline.py::_filter_and_dedupe`)
- Local LLM inference via `llama-cpp-python` (GGUF, quantized
  Llama-3.2-3B) — Private Offline Mode needs no API key and no internet
  connection
- Grounded Q&A with page/section-level source citations and an explicit
  anti-hallucination system instruction
- Document summarization and structured insight extraction, using
  single-pass generation for documents that fit the context budget and
  a genuine map-reduce pipeline (`core/pipeline.py::_generate_from_full_document`)
  for larger documents, instead of naively truncating to the first N
  characters
- The original Groq/cloud path is preserved as an optional, clearly
  labeled "Cloud Comparison Mode" — never the default — to make the
  privacy/latency contrast directly demoable rather than deleting the
  original project's capability
- A live performance dashboard showing real, measured latency per
  pipeline stage and the actual active execution provider
- 62 automated test cases (61 passing with a full `pip install -r
  requirements.txt`; 1 additional test requires torch/transformers for
  an optional deeper ONNX export integration check) across ingestion,
  chunking, embeddings, retrieval, prompt construction, and pipeline
  orchestration
- Qualcomm AI Hub documentation folder (`qualcomm_ai_hub/`) describing
  the real, honest relationship between this code and Snapdragon/QNN
  hardware acceleration — clearly separating what is implemented and
  verified from what is a documented, hardware-dependent future path

This is a genuine architectural pivot — from a stateless cloud-API
wrapper to a modular, privacy-first, locally-running RAG system with a
Snapdragon-oriented execution architecture — not a rename or a UI reskin
of the original project.

## Problem

Knowledge workers, students, and professionals routinely feed confidential
documents (contracts, medical reports, financial statements, research) into
cloud LLM APIs just to ask a question. That means sensitive data leaves the
device, requires an internet connection, incurs per-token cost, and is
unusable in regulated or offline environments.

## Solution

DocMind runs the entire document-intelligence pipeline — text extraction,
chunking, embedding, retrieval, and answer generation — locally, using a
small quantized open-source LLM and embedding model. In **Private Offline
Mode**, no document content, question, or answer ever leaves the device.
An optional, clearly-labeled **Cloud Comparison Mode** (using the original
Groq API implementation) is kept only to demonstrate the contrast.

## Key Features

- Upload PDF, DOCX, or TXT documents
- Grounded Q&A with source page citations (answers only from document content)
- Document summarization (executive summary, key points, dates, action items)
- Structured insight extraction (entities, numbers, risks, topics)
- Private Offline Mode (default) vs. Cloud Comparison Mode (optional)
- Live performance dashboard with real, measured metrics
- Transparent execution-provider detection (CPU / DirectML / Qualcomm QNN)

## Architecture

```
User → Streamlit UI → Ingestion (pypdf/python-docx) → Cleaning → Chunking
     → ONNX Embedding Model (MiniLM, ONNX Runtime: QNN → DirectML → CPU)
     → FAISS Vector Search → Prompt Construction → Local LLM (llama.cpp, GGUF)
     → Grounded Answer + Source Citations + Performance Metrics
```

A polished diagram of the same architecture (with an implemented/
hardware-dependent/optional legend) is at `assets/architecture.svg`.

Full component breakdown: see `core/` — each subpackage (ingestion,
embeddings, retrieval, generation, execution, performance) is
independently testable and has its own test file under `tests/`.

## Technology Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF/DOCX extraction | pypdf, python-docx |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 → ONNX → ONNX Runtime |
| Vector search | FAISS (in-memory, per-session) |
| Local LLM | Llama-3.2-3B-Instruct (GGUF) via llama-cpp-python |
| Cloud comparison (optional) | Groq API, Llama-3.3-70B |
| Execution provider detection | onnxruntime.get_available_providers() |

## Local/Offline Architecture

Private Offline Mode makes zero network calls once models are downloaded:
extraction, chunking, embedding, retrieval, and generation are all local.
This is enforced in code, not just claimed — `core/generation/local_llm.py`
and `core/embeddings/onnx_embedder.py` load files from disk only; the only
network-capable module in the whole app is `core/generation/cloud_llm.py`,
which is never called unless the user explicitly selects Cloud Comparison
Mode and a `GROQ_API_KEY` is configured.

## Snapdragon Optimization

See `qualcomm_ai_hub/README.md` for the full, honest breakdown. Summary:

| Capability | Status | Evidence |
|---|---|---|
| Local LLM (offline, no cloud) | Implemented | `core/generation/local_llm.py`, tested |
| ONNX embeddings with provider fallback | Implemented | `core/embeddings/onnx_embedder.py`, `tests/test_embeddings.py` (7 test functions) |
| CPU execution | Verified | Measured on development machine |
| DirectML execution | Documented capability | Requires Windows; not verified on this dev machine |
| Qualcomm QNN / Hexagon NPU execution | Optional, verified only if tested on real hardware or AI Hub device cloud | See `qualcomm_ai_hub/profiling.md` |
| Offline mode | Implemented | Wi-Fi-off manual test (see Demo Script) |

**We do not claim NPU acceleration unless it is backed by one of the
sources in `qualcomm_ai_hub/README.md`'s evidence table.**

## Qualcomm AI Hub

See `qualcomm_ai_hub/` for:
- `README.md` — what's implemented vs. documented, and our honesty policy
- `model_export.md` — real export/quantization steps for both models
- `profiling.md` — how to get real Snapdragon latency numbers via AI Hub's
  hosted device cloud, without needing to own the hardware

## Installation

**Recommended Python version: 3.10, 3.11, or 3.12** (widest wheel availability
for `faiss-cpu` and `onnxruntime` on Windows — verified against PyPI's
published wheel list for both packages as of this audit).

```bash
git clone <your-repo-url>
cd docmind-snapdragon
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

`faiss-cpu` ships real prebuilt wheels for Windows (verified via PyPI) — no
special handling needed. `llama-cpp-python` does **not** ship any prebuilt
wheel on PyPI for any platform (verified during this audit: the PyPI release
is source-only) — see **Common Errors** below for the correct fix.

## Model Setup

Two one-time setup steps, both documented with exact commands:

**1. Embedding model (ONNX export)**
```bash
pip install torch transformers onnx --break-system-packages
python scripts/export_embeddings.py
```
This creates `models/minilm-onnx/model.onnx` and `tokenizer.json`. `torch`,
`transformers`, and `onnx` are only needed for this one-time step — the app
itself never imports them. The export script self-verifies the output shape
of the model it produces and will fail loudly (not silently) if something
is wrong — this check was added after a real bug was found during an audit
of this project (see "Known Issues Found & Fixed" below).

**2. Local LLM (GGUF download)**
```bash
python scripts/download_models.py
```
Or manually download any instruction-tuned Llama-3.2-3B GGUF file (Q4_K_M
quantization recommended) and place it at
`models/llama-3.2-3b-instruct.Q4_K_M.gguf`.

## Running the Application

```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Upload a document, then use the Ask /
Summarize / Insights / Performance tabs.

## Testing

```bash
pip install -r requirements.txt
pytest -q
```

Verified during this audit in a genuinely clean virtual environment
(fresh `python -m venv`, `pip install -r requirements.txt`, nothing else
pre-installed): **61 passed, 1 skipped**.

62 test cases total, across ingestion, chunking, embeddings (including
building a small synthetic ONNX model on the fly via the `onnx` package,
which is why `onnx` is a declared dependency even though the main app
only imports `onnxruntime`), retrieval (including relevance-threshold
filtering and de-duplication), generation, and the RAG pipeline
orchestration layer (including the map-reduce summarization path for
large documents). Tests do not require internet access or real model
files — they use synthetic ONNX models and stub LLMs/embedders to verify
logic correctness independently of any specific model weights.
`llama-cpp-python`'s own "not installed" error-message test is simulated
via `monkeypatch` so it passes regardless of whether that package is
actually installed in the test environment.

**1 of these 62 requires an additional, intentionally-optional
dependency and is skipped by default:** the parametrized cases in
`tests/test_export_integration.py` need `torch`/`transformers` (see
"Optional deeper integration test" below) and are skipped via
`pytest.importorskip` — visible in the test output as `... skipped`, not
hidden or silently omitted from the file.

**Optional deeper integration test** (requires torch/transformers, ~1GB
download, not part of the main app's dependencies):
```bash
pip install torch transformers onnx --break-system-packages
pytest tests/test_export_integration.py -v
```
This exercises the real ONNX export path against a small, locally-built
BERT-family model (no internet/HF access needed for the test itself) and
is what caught the export bug described below. It's automatically skipped
by a plain `pytest` run if torch/transformers aren't installed.

## Benchmarking

```bash
python scripts/benchmark.py path/to/sample.pdf
```
Requires both models to be set up (see Model Setup). Measures real
document-processing, embedding, retrieval, and generation latency, and
saves results to `scripts/benchmark_results.json`. Never hand-edit this
file — regenerate it if you need updated numbers.

## Privacy

In Private Offline Mode:
- No API key is required
- No document content, question, or answer is transmitted over the network
- All processing (extraction, embedding, retrieval, generation) happens
  on-device

This is an architectural property of the code (see "Local/Offline
Architecture" above), not just a UI label. We make no claims beyond what
the code actually does — DocMind does not implement encryption-at-rest or
sandboxing beyond what your OS provides.

## Limitations

- Scanned/image-only PDFs (no text layer) are not OCR'd by default and
  will show a clear error rather than silently producing empty answers.
- Summarization/insights on documents that fit within
  `MAX_SUMMARY_CONTEXT_CHARS` (`config.py`) use a single generation pass;
  larger documents use a map-reduce pass (per-segment extraction, then a
  final synthesis call) instead of truncating, so the whole document is
  covered — see `core/pipeline.py::_generate_from_full_document`. This
  means summarization/insights on very large documents take multiple
  LLM calls and proportionally longer than a single-pass request.
- DOCX files don't have a true "page" concept; citations from DOCX show
  synthetic "section" numbers, not real page numbers.
- Qualcomm QNN/Hexagon NPU acceleration requires Windows-on-Snapdragon
  hardware and the `onnxruntime-qnn` package; it is not active on a
  generic Windows/Linux/macOS development machine, by design.

## Future Work

- Complete the Genie/QNN LLM export path documented in
  `qualcomm_ai_hub/model_export.md`
- OCR fallback for scanned documents
- Persistent (multi-session) vector index for large document libraries
- Cross-encoder reranking for retrieval quality on longer documents

## Demo

See `docs/DEMO_SCRIPT.md` for the full 2–3 minute walkthrough.

## Challenge Evaluation Mapping

See `docs/SUBMISSION.md` for the full criterion-by-criterion mapping.

## Common Errors and Fixes

| Error | Fix |
|---|---|
| `llama-cpp-python` build fails/hangs | PyPI ships **no prebuilt wheel for llama-cpp-python on any platform** (verified — it's source-only). Use the maintainer's official prebuilt CPU wheel index instead: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`. If you need to build from source anyway, install Visual Studio Build Tools ("Desktop development with C++" workload) on Windows first. |
| "Embedding model not found" | Run `python scripts/export_embeddings.py` (needs `pip install torch transformers onnx` first). |
| "Local LLM model not found" | Run `python scripts/download_models.py` or manually place a GGUF file at the path shown in the error. |
| App shows "Execution provider: CPU" on a Snapdragon PC | Install `onnxruntime-qnn` and the Qualcomm QNN SDK runtime libraries — CPU is the correct, honest fallback until those are present. |
| "No readable text was found in this document" | The PDF is likely a scanned image with no text layer; OCR is not enabled by default (see Limitations). |
| Cloud Comparison Mode greyed out | Set `GROQ_API_KEY` in a `.env` file (copy from `.env.example`). |
| `export_embeddings.py` fails with a connection/403 error | Hugging Face is unreachable from your network/environment — check your internet connection; this step needs real internet access. |
| `ModuleNotFoundError: requests` when running `download_models.py` | Run `pip install -r requirements.txt` again — `requests` is included; if you installed dependencies individually you may have missed it. |

## Known Issues Found & Fixed During Audit

A full audit of this project (dependencies, ONNX export, Windows paths,
Streamlit state, and the full test suite) found and fixed these real
issues — documented here rather than silently patched, since the review
process itself is part of demonstrating technical credibility:

| Issue | Root cause | Fix | Evidence |
|---|---|---|---|
| ONNX export could silently produce a second, unwanted output tensor | BERT-family models (MiniLM included) return both `last_hidden_state` and a pooler output; naming only one `output_names` doesn't guarantee the exporter drops the other | Export now wraps the model so only `last_hidden_state` is ever returned, regardless of torch/transformers version or exporter backend | `tests/test_export_integration.py`, run against a real local BERT model |
| `scripts/download_models.py` used `requests` without declaring it | Overlooked dependency | Added `requests>=2.31.0` to `requirements.txt` | Verified by grepping all `import` statements against `requirements.txt` |
| `llama-cpp-python` install guidance assumed a prebuilt wheel exists | Incorrect assumption | Verified via PyPI's JSON API that only a source distribution is published; documented the real, maintainer-published prebuilt-wheel index instead | PyPI API query performed during this audit |
| Tokenizer padding assumed `[PAD]` is always id 0 | True for MiniLM/BERT but not guaranteed for every tokenizer | `onnx_embedder.py` now looks up the real pad token id via `token_to_id("[PAD]")` with a safe fallback | Code review |
| Uploaded temp files were never cleaned up | Oversight | `app.py` now deletes the temp file after ingestion, wrapped in try/except for Windows file-locking edge cases | Code review |
| `test_local_llm_missing_library_reports_install_instructions` silently depended on `llama-cpp-python` NOT being installed in the test environment | The test created an invalid GGUF file expecting an `ImportError`, but `llama-cpp-python` is a real, declared runtime dependency — once installed, the code reaches a different (also-correct) "failed to load" error path instead, breaking the assertion | Test now uses `monkeypatch` to force the "not installed" condition explicitly, and a second test was added that covers the "installed but invalid file" path separately, guarded by `pytest.importorskip` | `tests/test_generation.py`, run and passing |
| Summarization/insights truncated large documents to the first ~8000 characters instead of covering the whole document | Simple truncation strategy in the original `_select_chunks_within_budget` helper | Replaced with a real map-reduce pipeline: documents that fit the context budget still use one fast single-pass call; larger documents are split into segments, each summarized independently ("map"), then synthesized into one whole-document result ("reduce") — see `core/pipeline.py::_generate_from_full_document` | `tests/test_pipeline.py::test_summarize_large_document_uses_map_reduce`, verified with a real 29K-character synthetic document during this audit |
| Retrieval had no relevance floor or de-duplication | Chunk overlap in `chunker.py` can retrieve near-duplicate chunks, and low-similarity chunks were passed to the LLM as if relevant | Added `MIN_RELEVANCE_SCORE` (config.py) and `DocMindPipeline._filter_and_dedupe`, which drops chunks below the relevance floor and removes duplicate chunk text before building the prompt, while always keeping at least one result so context is never silently emptied | `tests/test_pipeline.py`, 4 new passing tests |
| `pytest -q` failed at collection (not a clean skip) on a fresh `pip install -r requirements.txt` | `tests/test_embeddings.py` imports the `onnx` package unconditionally to build a synthetic test model, but `onnx` was never declared in `requirements.txt` | Added `onnx>=1.15.0` to `requirements.txt` with a comment explaining it's a test-time dependency for building synthetic models (distinct from `onnxruntime`, which the app itself uses at runtime) | Found by actually running `pip install -r requirements.txt` + `pytest -q` in a fresh virtual environment with nothing pre-installed during this audit; confirmed fixed by re-running the same two commands |
