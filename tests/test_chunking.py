"""
Tests for core/ingestion/chunker.py and core/ingestion/text_cleaner.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest

from core.ingestion.pdf_extractor import ExtractedDocument, PageText
from core.ingestion.chunker import chunk_text, chunk_document, Chunk
from core.ingestion.text_cleaner import clean_text, is_meaningful_text


# ---------- chunker ----------

def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("   ") == []


def test_chunk_text_short_text_single_chunk():
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_splits_long_text():
    text = "A" * 2500
    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_text_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_chunk_document_preserves_page_numbers():
    document = ExtractedDocument(
        source_path="fake.pdf",
        pages=[
            PageText(page_number=1, text="Hello world. " * 50),
            PageText(page_number=2, text="Second page content. " * 50),
        ],
    )
    chunks = chunk_document(document, chunk_size=200, overlap=20)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert {c.page_number for c in chunks} == {1, 2}


def test_chunk_document_assigns_sequential_ids():
    document = ExtractedDocument(
        source_path="fake.pdf",
        pages=[PageText(page_number=1, text="word " * 300)],
    )
    chunks = chunk_document(document, chunk_size=200, overlap=20)
    ids = [c.chunk_id for c in chunks]
    assert ids == list(range(len(chunks)))


# ---------- text cleaner ----------

def test_clean_text_fixes_hyphenated_line_breaks():
    raw = "This is a hyphen-\nated word."
    assert "hyphenated" in clean_text(raw)


def test_clean_text_collapses_excess_newlines():
    raw = "Paragraph one.\n\n\n\n\nParagraph two."
    cleaned = clean_text(raw)
    assert "\n\n\n" not in cleaned


def test_clean_text_collapses_repeated_spaces():
    raw = "Too    many      spaces."
    assert "  " not in clean_text(raw)


def test_clean_text_empty_input_returns_empty_string():
    assert clean_text("") == ""
    assert clean_text(None or "") == ""


def test_is_meaningful_text_rejects_near_empty_content():
    assert is_meaningful_text("") is False
    assert is_meaningful_text("a b") is False  # below default min_words
    assert is_meaningful_text("This has enough real words in it") is True
