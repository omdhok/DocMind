"""
Tests for core/generation.

We do NOT load a real local LLM or call the real Groq API in tests --
that would require multi-GB downloads / network access / a paid API key.
Instead we test:
  1. Prompt construction (pure string logic, fully verifiable).
  2. Graceful "unavailable" behavior when the model file / package / API
     key is missing (a required error-handling case from the project spec).
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest

from core.generation.prompts import build_qa_prompt, build_summary_prompt, build_insights_prompt, SYSTEM_INSTRUCTION
from core.generation.local_llm import LocalLLM, LocalLLMUnavailableError
from core.generation.cloud_llm import CloudLLM, CloudLLMUnavailableError


# ---------- prompt construction ----------

def test_qa_prompt_includes_question_and_page_numbers():
    chunks = [{"text": "Revenue grew 12% in Q3.", "page_number": 4}]
    prompt = build_qa_prompt("What was Q3 revenue growth?", chunks)
    assert "What was Q3 revenue growth?" in prompt
    assert "Page 4" in prompt
    assert "Revenue grew 12% in Q3." in prompt


def test_qa_prompt_handles_no_context_gracefully():
    prompt = build_qa_prompt("Anything in here?", [])
    assert "no relevant context found" in prompt


def test_summary_prompt_has_required_sections():
    chunks = [{"text": "Some document text.", "page_number": 1}]
    prompt = build_summary_prompt(chunks)
    for heading in ["Executive Summary:", "Key Points:", "Important Facts:", "Action Items:"]:
        assert heading in prompt


def test_insights_prompt_has_required_sections():
    chunks = [{"text": "Some document text.", "page_number": 1}]
    prompt = build_insights_prompt(chunks)
    for heading in ["Key Entities:", "Dates:", "Important Numbers:", "Risks or Concerns:"]:
        assert heading in prompt


def test_system_instruction_forbids_hallucination():
    assert "does not provide enough" in SYSTEM_INSTRUCTION
    assert "invent" in SYSTEM_INSTRUCTION.lower()


# ---------- local LLM graceful unavailability ----------

def test_local_llm_missing_model_file_is_not_ready(tmp_path):
    llm = LocalLLM(model_path=str(tmp_path / "does_not_exist.gguf"))
    assert llm.is_ready is False
    assert "download_models.py" in llm.error_message


def test_local_llm_generate_raises_clear_error_when_not_ready(tmp_path):
    llm = LocalLLM(model_path=str(tmp_path / "missing.gguf"))
    with pytest.raises(LocalLLMUnavailableError):
        llm.generate("test prompt")


def test_local_llm_missing_library_reports_install_instructions(tmp_path, monkeypatch):
    # Create a dummy (invalid) file so the code reaches past the
    # "file not found" check. This test asserts the *install-instructions*
    # error path specifically, so it must not depend on whether
    # llama-cpp-python happens to be installed in the environment running
    # the tests (it IS a declared runtime dependency of the app, so CI/dev
    # machines that installed requirements.txt will have it available).
    # We force the "not installed" condition explicitly instead of relying
    # on environment state, which is what made this test previously fragile.
    import builtins

    dummy_model = tmp_path / "dummy.gguf"
    dummy_model.write_bytes(b"not a real model")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "llama_cpp":
            raise ImportError("No module named 'llama_cpp' (simulated for this test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    llm = LocalLLM(model_path=str(dummy_model))
    assert llm.is_ready is False
    assert "pip install llama-cpp-python" in llm.error_message


def test_local_llm_corrupt_model_file_reports_load_failure(tmp_path):
    # Separate case: llama-cpp-python IS installed/importable, but the
    # model file itself is not a valid GGUF. This must fail cleanly with
    # a "Failed to load" message, not an unhandled exception -- and this
    # test only runs meaningfully when llama_cpp is actually installed.
    pytest.importorskip("llama_cpp")

    dummy_model = tmp_path / "dummy.gguf"
    dummy_model.write_bytes(b"not a real model")
    llm = LocalLLM(model_path=str(dummy_model))
    assert llm.is_ready is False
    assert "Failed to load local LLM" in llm.error_message


# ---------- cloud LLM graceful unavailability ----------

def test_cloud_llm_missing_api_key_is_not_ready(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cloud = CloudLLM()
    assert cloud.is_ready is False
    assert "GROQ_API_KEY" in cloud.error_message


def test_cloud_llm_generate_raises_clear_error_when_not_ready(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cloud = CloudLLM()
    with pytest.raises(CloudLLMUnavailableError):
        cloud.generate("test prompt")
