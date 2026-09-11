"""
Local LLM module.

Wraps a local, quantized GGUF model (Llama-3.2-3B-Instruct by default)
using llama-cpp-python. This is DocMind's "Private Offline Mode" engine:
once the GGUF file is downloaded, no internet connection or API key is
required to generate answers.

`llama_cpp` is imported LAZILY (inside __init__, not at module load time)
so that:
  - The rest of the app can be imported and unit-tested even on a machine
    where llama-cpp-python isn't installed yet or failed to build.
  - A missing/failed install produces one clear error message instead of
    crashing every other part of DocMind.

Note on Qualcomm/Snapdragon: llama.cpp on Windows currently accelerates
via CPU (with SIMD) and, on supported systems, Vulkan/OpenCL GPU backends.
It does NOT go through the QNN Execution Provider used elsewhere in this
project for the embedding model -- that is a genuinely separate, optional
path (documented in qualcomm_ai_hub/model_export.md) using Qualcomm's
"Genie" runtime, which is more involved to set up than llama.cpp. We are
not claiming NPU acceleration for this LLM path unless that separate
Genie/QNN pipeline has actually been built and verified.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from core.generation.prompts import SYSTEM_INSTRUCTION
from config import LOCAL_LLM_GGUF_PATH, LOCAL_LLM_CONTEXT_SIZE


DEFAULT_MODEL_PATH = LOCAL_LLM_GGUF_PATH
DEFAULT_CONTEXT_SIZE = LOCAL_LLM_CONTEXT_SIZE
DEFAULT_MAX_NEW_TOKENS = 512


class LocalLLMUnavailableError(Exception):
    """
    Raised when the local LLM can't be used -- either llama-cpp-python
    isn't installed, or the GGUF model file is missing.

    This is an EXPECTED state on a fresh checkout. The UI layer should
    catch this and show setup instructions instead of a traceback.
    """


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    generation_seconds: float

    @property
    def tokens_per_second(self) -> float:
        if self.generation_seconds <= 0:
            return 0.0
        return self.completion_tokens / self.generation_seconds


class LocalLLM:
    """
    Thin wrapper around llama_cpp.Llama for grounded document Q&A,
    summarization, and insight extraction.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        n_ctx: int = DEFAULT_CONTEXT_SIZE,
        n_threads: Optional[int] = None,
    ):
        self.model_path = Path(model_path)
        self._llm = None
        self._error: Optional[str] = None
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._load(n_threads)

    def _load(self, n_threads: Optional[int]) -> None:
        if not self.model_path.exists():
            self._error = (
                f"Local LLM model not found at '{self.model_path}'.\n"
                f"Run: python scripts/download_models.py to download it, "
                f"or see README.md 'Model Setup'."
            )
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            self._error = (
                "llama-cpp-python is not installed.\n"
                "Run: pip install llama-cpp-python\n"
                "(On Windows, if the build fails, see README.md 'Common Errors' "
                "for the prebuilt-wheel fallback command.)"
            )
            return

        try:
            kwargs = {"model_path": str(self.model_path), "n_ctx": self.n_ctx, "verbose": False}
            if n_threads:
                kwargs["n_threads"] = n_threads
            self._llm = Llama(**kwargs)
        except Exception as exc:
            self._error = f"Failed to load local LLM: {exc}"

    @property
    def is_ready(self) -> bool:
        return self._llm is not None

    @property
    def error_message(self) -> Optional[str]:
        return self._error

    def generate(
        self,
        user_prompt: str,
        system_instruction: str = SYSTEM_INSTRUCTION,
        max_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.2,
    ) -> GenerationResult:
        """
        Generate a response from the local LLM.

        Args:
            user_prompt: The full user-facing prompt (already includes
                retrieved context -- see core/generation/prompts.py).
            system_instruction: System-level instruction, defaults to the
                grounded-answer instruction that forbids hallucination.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature (low = more deterministic,
                appropriate for factual document Q&A).

        Raises:
            LocalLLMUnavailableError: if the model isn't loaded.
        """
        if self._llm is None:
            raise LocalLLMUnavailableError(self._error or "Local LLM is not loaded.")

        import time

        start = time.perf_counter()
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - start

        text = response["choices"][0]["message"]["content"].strip()
        usage = response.get("usage", {})

        return GenerationResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            generation_seconds=elapsed,
        )
