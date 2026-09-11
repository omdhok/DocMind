"""
Prompt templates.

Centralizing prompts here (instead of scattering f-strings through the
app) makes it easy to tune wording once and have it apply everywhere,
and makes the anti-hallucination instruction impossible to accidentally
skip in one code path but not another.
"""

SYSTEM_INSTRUCTION = (
    "You are DocMind, a private document assistant. Answer only using the "
    "supplied document context. If the answer cannot be found in the "
    "document, clearly say that the document does not provide enough "
    "information. Never invent facts, numbers, or details that are not "
    "present in the context. Keep answers concise and directly relevant "
    "to the question."
)


def build_qa_prompt(question: str, context_chunks: list) -> str:
    """
    Build the user-facing prompt for grounded Q&A.

    Args:
        question: The user's question.
        context_chunks: List of dicts with keys "text" and "page_number",
            typically produced from vector_store.SearchResult objects.

    Returns:
        A single prompt string combining context + question.
    """
    context_blocks = []
    for chunk in context_chunks:
        context_blocks.append(f"[Page {chunk['page_number']}]\n{chunk['text']}")
    context_text = "\n\n".join(context_blocks) if context_blocks else "(no relevant context found)"

    return (
        f"DOCUMENT CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"Answer the question using ONLY the document context above. "
        f"If the context does not contain the answer, say so explicitly."
    )


def build_summary_prompt(context_chunks: list) -> str:
    """
    Build the prompt for the "Summarize Document" feature.

    Uses the same context-grounding discipline as Q&A -- the summary must
    be built only from the supplied chunks, not general knowledge.
    """
    context_blocks = [chunk["text"] for chunk in context_chunks]
    context_text = "\n\n".join(context_blocks) if context_blocks else "(no document content available)"

    return (
        f"DOCUMENT CONTENT:\n{context_text}\n\n"
        "Summarize this document using ONLY the content above. Structure your "
        "response with these exact section headings:\n\n"
        "Executive Summary:\n"
        "Key Points:\n"
        "Important Facts:\n"
        "Important Dates:\n"
        "Action Items:\n\n"
        "If a section has no relevant content in the document, write "
        "\"None identified in the document\" under that heading instead of "
        "inventing content."
    )


def build_insights_prompt(context_chunks: list) -> str:
    """
    Build the prompt for the "Extract Insights" feature.
    """
    context_blocks = [chunk["text"] for chunk in context_chunks]
    context_text = "\n\n".join(context_blocks) if context_blocks else "(no document content available)"

    return (
        f"DOCUMENT CONTENT:\n{context_text}\n\n"
        "Extract structured insights using ONLY the content above. Use "
        "these exact section headings:\n\n"
        "Key Entities:\n"
        "Dates:\n"
        "Important Numbers:\n"
        "Action Items:\n"
        "Main Topics:\n"
        "Risks or Concerns:\n\n"
        "If a section has no relevant content in the document, write "
        "\"None identified in the document\" under that heading instead of "
        "inventing content."
    )


def build_map_chunk_prompt(context_chunks: list) -> str:
    """
    Build the prompt for the "map" step of map-reduce summarization on
    large documents (see core/pipeline.py, _map_reduce_generate).

    Produces a dense, factual partial summary of ONE segment of the
    document -- preserving concrete facts, numbers, and dates rather
    than writing prose -- so the later "reduce" step has enough signal
    to synthesize a whole-document result without re-reading raw text.
    """
    context_blocks = [chunk["text"] for chunk in context_chunks]
    context_text = "\n\n".join(context_blocks) if context_blocks else "(no content)"

    return (
        f"DOCUMENT SEGMENT:\n{context_text}\n\n"
        "Extract the key facts, figures, dates, names, and important points "
        "from ONLY this segment, as a dense bullet list. Use ONLY information "
        "present in this segment -- do not invent anything. Be concise; this "
        "is an intermediate extraction step, not a final summary."
    )


def build_reduce_summary_prompt(partial_summaries: list) -> str:
    """
    Build the "reduce" step prompt: synthesize partial per-segment
    extractions (from build_map_chunk_prompt) into one coherent,
    whole-document summary. This lets DocMind summarize documents larger
    than a single context window without simply truncating to the first
    N characters (a map-reduce approach rather than naive truncation).
    """
    combined = "\n\n".join(
        f"--- Segment {i + 1} key points ---\n{s}" for i, s in enumerate(partial_summaries)
    )
    return (
        f"The following are key-point extractions from consecutive segments "
        f"of one full document, in order:\n\n{combined}\n\n"
        "Using ONLY the information above, write a single coherent summary of "
        "the WHOLE document. Structure your response with these exact section "
        "headings:\n\n"
        "Executive Summary:\n"
        "Key Points:\n"
        "Important Facts:\n"
        "Important Dates:\n"
        "Action Items:\n\n"
        "If a section has no relevant content, write \"None identified in the "
        "document\" under that heading instead of inventing content. Merge "
        "duplicate points from different segments rather than repeating them."
    )


def build_reduce_insights_prompt(partial_summaries: list) -> str:
    """Reduce-step prompt for the Insights feature (see build_reduce_summary_prompt)."""
    combined = "\n\n".join(
        f"--- Segment {i + 1} key points ---\n{s}" for i, s in enumerate(partial_summaries)
    )
    return (
        f"The following are key-point extractions from consecutive segments "
        f"of one full document, in order:\n\n{combined}\n\n"
        "Using ONLY the information above, extract structured insights for the "
        "WHOLE document. Use these exact section headings:\n\n"
        "Key Entities:\n"
        "Dates:\n"
        "Important Numbers:\n"
        "Action Items:\n"
        "Main Topics:\n"
        "Risks or Concerns:\n\n"
        "If a section has no relevant content, write \"None identified in the "
        "document\" under that heading instead of inventing content. Merge "
        "duplicate points from different segments rather than repeating them."
    )
