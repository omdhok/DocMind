"""
Text chunking module.

Splits extracted document text into overlapping chunks suitable for
embedding + retrieval later. This is a simple, dependency-free sliding
window chunker -- no LangChain, no extra libraries. It's easy to swap
for something smarter later without touching the rest of the pipeline.
"""

from dataclasses import dataclass
from typing import List

from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from core.ingestion.pdf_extractor import ExtractedDocument


@dataclass
class Chunk:
    """A single chunk of text, ready to be embedded in a later module."""
    chunk_id: int
    text: str
    page_number: int  # which page this chunk came from (for citations later)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split a single string into overlapping chunks by character count.

    Args:
        text: The text to split.
        chunk_size: Max characters per chunk.
        overlap: Characters of overlap between consecutive chunks
            (keeps context from being cut off at chunk boundaries).

    Returns:
        List of text chunks (empty list if input text is empty/whitespace).
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == text_length:
            break
        start = end - overlap  # step forward, keeping overlap with previous chunk

    return chunks


def chunk_document(
    document: ExtractedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Chunk an entire ExtractedDocument, page by page, preserving which
    page each chunk came from. Keeping page numbers now means later
    modules (retrieval, LLM answers) can cite "page 4" instead of just
    dumping raw text at the user.

    Args:
        document: Output of extract_text_from_pdf().
        chunk_size: Max characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of Chunk objects with sequential chunk_id values.
    """
    all_chunks: List[Chunk] = []
    next_id = 0

    for page in document.pages:
        page_chunks = chunk_text(page.text, chunk_size=chunk_size, overlap=overlap)
        for piece in page_chunks:
            all_chunks.append(
                Chunk(chunk_id=next_id, text=piece, page_number=page.page_number)
            )
            next_id += 1

    return all_chunks
