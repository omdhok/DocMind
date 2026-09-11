"""
PDF text extraction module.

Responsible ONLY for turning a PDF file into raw text, page by page.
It knows nothing about chunking, embeddings, or LLMs -- keeping this
module single-purpose makes it easy to later add DOCX/TXT extractors
or an OCR fallback without touching any other part of DocMind.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class PageText:
    """Text extracted from a single PDF page."""
    page_number: int   # 1-indexed, matches how a human would refer to the page
    text: str


@dataclass
class ExtractedDocument:
    """Result of extracting a full document."""
    source_path: str
    pages: List[PageText] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenate all pages into one string, separated by blank lines."""
        return "\n\n".join(page.text for page in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be found, opened, or parsed."""


def extract_text_from_pdf(file_path: str) -> ExtractedDocument:
    """
    Extract text from a PDF file, page by page.

    Args:
        file_path: Path to a .pdf file on disk.

    Returns:
        ExtractedDocument containing per-page text.

    Raises:
        PDFExtractionError: if the file doesn't exist, isn't a PDF,
            has no pages, or can't be parsed.
    """
    path = Path(file_path)

    if not path.exists():
        raise PDFExtractionError(f"File not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise PDFExtractionError(f"Not a PDF file: {file_path}")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise PDFExtractionError(f"Could not open PDF: {exc}") from exc

    pages: List[PageText] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            # A single unreadable page shouldn't kill the whole document.
            # Keep an empty string so page numbering stays correct.
            text = ""
        pages.append(PageText(page_number=index + 1, text=text.strip()))

    if not pages:
        raise PDFExtractionError(f"No pages found in PDF: {file_path}")

    return ExtractedDocument(source_path=str(path), pages=pages)
