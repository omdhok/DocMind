"""
Manual test script for Module 1.

Run this with a real PDF to see extraction + chunking working end to end.
This is NOT part of the final app -- it's a quick way to sanity-check the
pipeline from the command line before the Streamlit UI exists (Module 2+).

Usage:
    python scripts/run_ingestion_demo.py path/to/your/file.pdf
"""

import sys
from pathlib import Path

# Allow running this script directly (e.g. `python scripts/run_ingestion_demo.py`)
# by adding the project root to the import path.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.ingestion.pdf_extractor import extract_text_from_pdf, PDFExtractionError
from core.ingestion.chunker import chunk_document


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_ingestion_demo.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        document = extract_text_from_pdf(pdf_path)
    except PDFExtractionError as e:
        print(f"Extraction failed: {e}")
        sys.exit(1)

    print(f"Extracted {document.page_count} page(s) from: {pdf_path}\n")

    chunks = chunk_document(document)
    print(f"Created {len(chunks)} chunk(s).\n")

    for chunk in chunks[:3]:
        print(f"--- Chunk {chunk.chunk_id} (page {chunk.page_number}) ---")
        print(chunk.text[:300])
        print()

    if len(chunks) > 3:
        print(f"... and {len(chunks) - 3} more chunk(s) not shown.")


if __name__ == "__main__":
    main()
