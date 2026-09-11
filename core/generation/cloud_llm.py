"""
Cloud LLM module -- "Cloud Comparison Mode" only.

This wraps the Groq API (the original project's implementation) so users
can optionally compare a cloud-hosted large model against DocMind's
local/offline mode. It is NEVER the default and NEVER required for the
app's core functionality.

Requires a GROQ_API_KEY environment variable. If it's not set, this
module reports itself as unavailable rather than crashing the app --
Cloud Comparison Mode simply won't be selectable in the UI.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

from core.generation.prompts import SYSTEM_INSTRUCTION
from config import CLOUD_MODEL_NAME

DEFAULT_CLOUD_MODEL = CLOUD_MODEL_NAME


class CloudLLMUnavailableError(Exception):
    """Raised when Cloud Comparison Mode can't be used (no API key / no package / no internet)."""


@dataclass
class CloudGenerationResult:
    text: str
    generation_seconds: float
    model_name: str


class CloudLLM:
    """
    Thin wrapper around the Groq API. Import of the `groq` package is
    lazy so the rest of the app works even if it isn't installed --
    Cloud Comparison Mode is optional, not core.
    """

    def __init__(self, model_name: str = DEFAULT_CLOUD_MODEL):
        self.model_name = model_name
        self._client = None
        self._error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            self._error = (
                "GROQ_API_KEY is not set. Cloud Comparison Mode is optional and "
                "disabled without it. Set it in a .env file (see .env.example) "
                "if you want to use this mode for a live demo comparison."
            )
            return

        try:
            from groq import Groq
        except ImportError:
            self._error = "The 'groq' package is not installed. Run: pip install groq"
            return

        try:
            self._client = Groq(api_key=api_key)
        except Exception as exc:
            self._error = f"Failed to initialize Groq client: {exc}"

    @property
    def is_ready(self) -> bool:
        return self._client is not None

    @property
    def error_message(self) -> Optional[str]:
        return self._error

    def generate(self, user_prompt: str, system_instruction: str = SYSTEM_INSTRUCTION) -> CloudGenerationResult:
        if self._client is None:
            raise CloudLLMUnavailableError(self._error or "Cloud LLM is not available.")

        start = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        elapsed = time.perf_counter() - start

        text = completion.choices[0].message.content.strip()
        return CloudGenerationResult(text=text, generation_seconds=elapsed, model_name=self.model_name)
