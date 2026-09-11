"""
Text cleaning module.

Extraction (PDF/DOCX) often produces messy text: repeated whitespace,
broken hyphenation across line breaks, stray control characters, and
page-header/footer noise. This module cleans text BEFORE chunking so
embeddings and the LLM see clean, well-formed text.

Kept deliberately simple -- regex-based, no NLP dependency.
"""

import re


def clean_text(text: str) -> str:
    """
    Normalize extracted text for downstream chunking/embedding.

    Steps:
      1. Fix hyphenated line breaks ("infor-\\nmation" -> "information").
      2. Collapse multiple newlines into a single paragraph break.
      3. Collapse repeated spaces/tabs.
      4. Strip non-printable control characters.
      5. Trim leading/trailing whitespace.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text. Returns "" if input is empty/whitespace.
    """
    if not text:
        return ""

    # 1. De-hyphenate words split across a line break, e.g. "docu-\nment"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 2. Collapse 3+ newlines into a double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 3. Collapse repeated spaces/tabs (but keep newlines intact)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 4. Strip non-printable/control characters (keep \n and \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text.strip()


def is_meaningful_text(text: str, min_words: int = 3) -> bool:
    """
    Heuristic check used to detect pages/documents with no real extractable
    text (e.g. a scanned PDF with no OCR layer). Used to trigger a clear
    user-facing error instead of silently embedding empty/junk text.

    Args:
        text: Cleaned text to check.
        min_words: Minimum whitespace-separated tokens to count as meaningful.

    Returns:
        True if the text looks like real content.
    """
    if not text:
        return False
    words = text.split()
    return len(words) >= min_words
