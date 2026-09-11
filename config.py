"""
Central configuration for DocMind.

Keep all tunable constants here so later modules (embeddings, retrieval,
local LLM, Qualcomm QNN execution provider) can import from a single
place instead of hardcoding values inside each file.
"""

# --- Chunking (used by core/ingestion/chunker.py) ---
DEFAULT_CHUNK_SIZE = 1000     # characters per chunk
DEFAULT_CHUNK_OVERLAP = 150   # characters of overlap between consecutive chunks

# --- Ingestion ---
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

# --- Embeddings ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_EMBEDDING_MODEL_DIR = "models/minilm-onnx"
ONNXRUNTIME_PROVIDER_PRIORITY = ["QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]

# --- Local LLM ---
LOCAL_LLM_GGUF_PATH = "models/llama-3.2-3b-instruct.Q4_K_M.gguf"
LOCAL_LLM_CONTEXT_SIZE = 4096

# --- Cloud comparison mode (optional, never default) ---
CLOUD_MODEL_NAME = "llama-3.3-70b-versatile"

# --- Retrieval ---
DEFAULT_TOP_K = 4
# Minimum cosine similarity for a retrieved chunk to be considered
# relevant enough to include as context. Chunks scoring below this are
# dropped rather than silently fed to the LLM -- this is what lets the
# model correctly say "not found in the document" instead of grounding
# an answer in a barely-related chunk. 0.15 is a deliberately low/safe
# floor for a MiniLM-class model (embeddings are not perfectly
# calibrated), tuned to filter out near-zero-similarity noise without
# discarding genuinely relevant but loosely-worded matches.
MIN_RELEVANCE_SCORE = 0.15

# --- Summarization / insights ---
MAX_SUMMARY_CONTEXT_CHARS = 8000
