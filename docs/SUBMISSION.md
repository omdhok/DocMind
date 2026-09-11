# DocMind — Submission

## Project Title
DocMind: Private On-Device Document Intelligence for Snapdragon PCs

## 100-word description
DocMind is a private document assistant that answers questions,
summarizes, and extracts structured insights from PDFs, DOCX, and text
files — entirely on-device. Built as a significant evolution of an
existing cloud-based PDF Q&A tool, DocMind replaces the cloud LLM
pipeline with a local, quantized Llama-3.2-3B model and an ONNX
Runtime-based embedding pipeline that automatically selects the fastest
available execution provider (Qualcomm QNN, DirectML, or CPU), targeting
Snapdragon-powered Windows PCs. A transparent performance dashboard
shows real, measured latency and the actual active execution provider —
never a marketing number.

## 200-word description
Knowledge workers routinely upload confidential documents to cloud AI
services just to ask a simple question, exposing private data,
requiring internet access, and incurring ongoing API costs. DocMind
solves this by running the entire document-intelligence pipeline —
extraction, chunking, embedding, retrieval, and answer generation —
locally. Private Offline Mode makes zero network calls: an ONNX
Runtime-based MiniLM embedding model and a quantized local Llama-3.2-3B
LLM (via llama.cpp) handle every step on-device. The application is
architected specifically for Snapdragon-powered HP PCs: the embedding
pipeline uses ONNX Runtime's execution-provider abstraction with a
priority order of Qualcomm's QNN provider (Hexagon NPU) first, then
DirectML, then CPU — genuinely detected at runtime, never hardcoded or
assumed. An optional, clearly-labeled Cloud Comparison Mode retains the
original Groq-based implementation purely to demonstrate the privacy
and latency contrast. Every answer includes source page citations and
is instructed to refuse rather than hallucinate when the document
doesn't contain the answer. A live performance dashboard surfaces real,
measured metrics for every request. This is a substantial architectural
transformation of an existing project — from a cloud-API wrapper to a
genuinely private, Snapdragon-optimized, locally-running AI application.

## Problem Statement
Private documents (contracts, medical records, financial statements)
are routinely sent to third-party cloud AI APIs just to answer simple
questions, creating privacy exposure, internet dependency, and recurring
cost — with no viable offline alternative for most users.

## Solution
A document assistant that runs its full AI pipeline locally: local
embeddings via ONNX Runtime, local retrieval via FAISS, and local
generation via a quantized Llama-3.2-3B model, with the architecture
built to take advantage of Snapdragon's Hexagon NPU when the appropriate
Qualcomm runtime is present, and to gracefully use CPU otherwise.

## Innovation
Most student AI projects either wrap a cloud API or run an oversized
model on CPU and call it "on-device." DocMind instead implements a real
execution-provider detection and fallback chain, is honest about exactly
which claims are hardware-verified versus documented-but-unverified, and
retains a working cloud-comparison mode as a direct, demoable contrast —
rather than simply deleting the original project's cloud integration.

## Technical Implementation
Modular pipeline (`core/ingestion`, `core/embeddings`, `core/retrieval`,
`core/generation`, `core/execution`, `core/performance`), 62 automated
test cases — 61 passing with a full `pip install -r requirements.txt`
(verified in a clean virtual environment during this audit), 1 more
running only with an optional extra dependency (torch/transformers) —
covering extraction, chunking, embedding math, retrieval (including
relevance filtering and de-duplication), prompt construction,
missing-document handling, provider fallback, and map-reduce
summarization — all runnable without internet access or real model
weights via synthetic ONNX models and stub LLMs.

## Snapdragon Optimization
`ExecutionProviderManager` selects `QNNExecutionProvider` >
`DmlExecutionProvider` > `CPUExecutionProvider`, strictly based on what
ONNX Runtime reports as genuinely installed. See `qualcomm_ai_hub/` for
the full model-export and device-cloud-profiling documentation, and
README.md's evidence table for exactly which claims are verified versus
documented.

## Deployment & Accessibility
Runs on any Windows, macOS, or Linux machine via `pip install -r
requirements.txt` + two one-time model-setup scripts — no GPU, no CUDA,
no Docker required. Fully functional on a standard development machine
during judging even without Snapdragon hardware present, while being
architected to automatically take advantage of Snapdragon acceleration
when it is.

## Privacy
Private Offline Mode makes no network calls after model setup — this is
enforced by the module structure (`local_llm.py` and `onnx_embedder.py`
only read local files), not just a UI label.

## Expected Impact
Demonstrates a practical, honest template for privacy-preserving,
edge-optimized AI applications on Snapdragon Windows PCs — directly
relevant to regulated industries (legal, healthcare, finance) where
cloud AI tools are often disallowed.

## Evaluation Criterion Mapping

| Criterion | How DocMind addresses it |
|---|---|
| Technical Implementation | Modular RAG pipeline, real ONNX Runtime provider-fallback logic, 62 automated test cases (61 passing with a full `pip install -r requirements.txt`, verified in a clean venv), grounded-answer prompting with anti-hallucination instruction |
| Application Use Case & Innovation | Solves a real, relatable privacy problem; genuine architectural pivot from the original cloud-based project, not a reskin |
| Deployment & Accessibility | Works on any OS/hardware via pip, no GPU/CUDA/Docker requirement, clear setup scripts and error messages for every failure mode |
| Presentation & Documentation | README with an explicit implemented-vs-documented evidence table, demo script, Qualcomm AI Hub documentation folder, this submission document |
