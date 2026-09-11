"""
RAG pipeline orchestrator.

This is the ONLY module app.py needs to talk to for the core workflow:
    ingest a document -> ask questions / summarize / extract insights.

It wires together:
    core.ingestion   (extraction + cleaning + chunking)
    core.embeddings  (Embedder)
    core.retrieval   (VectorStore)
    core.generation  (LocalLLM / CloudLLM + prompts)
    core.performance (RunMetrics / Timer)

Design choices relevant to the "no crash on error" requirement:
    - Every public method returns a result object OR raises one of the
      specific, already-defined error types from the underlying modules
      (EmbeddingModelNotFoundError, LocalLLMUnavailableError, etc).
      app.py is expected to catch these and show a clean message --
      nothing here uses bare `except: pass` or swallows errors silently.
"""

from pathlib import Path
from typing import List, Optional

from core.ingestion.pdf_extractor import extract_text_from_pdf, PDFExtractionError, ExtractedDocument
from core.ingestion.docx_extractor import extract_text_from_docx, DocxExtractionError
from core.ingestion.text_cleaner import clean_text, is_meaningful_text
from core.ingestion.chunker import chunk_document, Chunk
from core.embeddings.embedder import Embedder
from core.retrieval.vector_store import VectorStore, ChunkMetadata, VectorStoreError
from core.generation.local_llm import LocalLLM, LocalLLMUnavailableError
from core.generation.cloud_llm import CloudLLM, CloudLLMUnavailableError
from core.generation import prompts
from core.performance.metrics import RunMetrics, Timer
from config import MAX_SUMMARY_CONTEXT_CHARS, DEFAULT_TOP_K, MIN_RELEVANCE_SCORE

# MAX_SUMMARY_CONTEXT_CHARS: rough character budget for summarization/insights
# context so we don't silently overflow the local LLM's context window on a
# very large document. This is a simple, honest limitation (documented in
# README) rather than a hidden truncation.


class DocumentTooLargeWarning:
    """Marker class used to flag (not error on) large-document truncation."""


class IngestionError(Exception):
    """Raised for any document-ingestion failure the UI should show cleanly."""


class DocMindPipeline:
    def __init__(self, embedder: Embedder, local_llm: LocalLLM, cloud_llm: Optional[CloudLLM] = None):
        self.embedder = embedder
        self.local_llm = local_llm
        self.cloud_llm = cloud_llm

        self.vector_store: Optional[VectorStore] = None
        self.chunks: List[Chunk] = []
        self.document_name: Optional[str] = None
        self.last_ingest_metrics: Optional[RunMetrics] = None

    # ---------------- Ingestion ----------------

    def ingest_document(self, file_path: str) -> RunMetrics:
        """
        Extract, clean, chunk, embed, and index a document.

        Raises:
            IngestionError: for any failure in this pipeline stage,
                with a human-readable message.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        timer = Timer()

        with timer.measure("document_processing"):
            try:
                if suffix == ".pdf":
                    document: ExtractedDocument = extract_text_from_pdf(str(path))
                elif suffix == ".docx":
                    document = extract_text_from_docx(str(path))
                elif suffix == ".txt":
                    document = self._load_txt(path)
                else:
                    raise IngestionError(
                        f"Unsupported file type '{suffix}'. Supported types: .pdf, .docx, .txt"
                    )
            except (PDFExtractionError, DocxExtractionError) as exc:
                raise IngestionError(str(exc)) from exc

            for page in document.pages:
                page.text = clean_text(page.text)

            if not any(is_meaningful_text(p.text) for p in document.pages):
                raise IngestionError(
                    "No readable text was found in this document. If this is a "
                    "scanned PDF (an image of text rather than real text), "
                    "DocMind's OCR fallback is not enabled by default -- "
                    "see README.md 'Limitations'."
                )

            chunks = chunk_document(document)
            if not chunks:
                raise IngestionError("Document produced no usable text chunks.")

        with timer.measure("embedding"):
            if not self.embedder.is_ready:
                raise IngestionError(
                    f"Embedding model is not available: {self.embedder.error_message}"
                )
            texts = [c.text for c in chunks]
            embeddings = self.embedder.encode(texts)

        with timer.measure("indexing"):
            try:
                store = VectorStore(embedding_dim=self.embedder.embedding_dim)
                metadata = [
                    ChunkMetadata(
                        chunk_id=c.chunk_id,
                        text=c.text,
                        page_number=c.page_number,
                        document_name=path.name,
                    )
                    for c in chunks
                ]
                store.add(embeddings, metadata)
            except VectorStoreError as exc:
                raise IngestionError(f"Failed to build search index: {exc}") from exc

        self.vector_store = store
        self.chunks = chunks
        self.document_name = path.name

        metrics = RunMetrics(
            document_processing_seconds=timer.elapsed["document_processing"],
            embedding_seconds=timer.elapsed["embedding"],
            chunk_count=len(chunks),
            active_execution_provider=self.embedder.active_provider,
        )
        self.last_ingest_metrics = metrics
        return metrics

    @staticmethod
    def _load_txt(path: Path) -> ExtractedDocument:
        from core.ingestion.pdf_extractor import ExtractedDocument, PageText

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            raise IngestionError(f"Could not read text file: {exc}") from exc

        if not text.strip():
            raise IngestionError("Text file is empty.")

        return ExtractedDocument(source_path=str(path), pages=[PageText(page_number=1, text=text)])

    # ---------------- Ask (RAG Q&A) ----------------

    def ask(self, question: str, top_k: int = DEFAULT_TOP_K, use_cloud: bool = False):
        """
        Answer a question grounded in the ingested document.

        Returns:
            (answer_text: str, sources: List[dict], metrics: RunMetrics)
        """
        if self.vector_store is None:
            raise IngestionError("No document has been ingested yet.")
        if not question or not question.strip():
            raise IngestionError("Question cannot be empty.")

        timer = Timer()

        with timer.measure("embedding"):
            query_embedding = self.embedder.encode([question])[0]

        with timer.measure("retrieval"):
            # Over-fetch slightly so that, after relevance filtering and
            # de-duplication, we still have a good chance of ending up
            # with a useful number of context chunks.
            raw_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
            results = self._filter_and_dedupe(raw_results, top_k=top_k)

        context_chunks = [
            {
                "text": r.metadata.text,
                "page_number": r.metadata.page_number,
                "score": r.score,
            }
            for r in results
        ]
        prompt = prompts.build_qa_prompt(question, context_chunks)

        with timer.measure("generation"):
            if use_cloud:
                if self.cloud_llm is None or not self.cloud_llm.is_ready:
                    raise CloudLLMUnavailableError(
                        self.cloud_llm.error_message if self.cloud_llm else "Cloud LLM not configured."
                    )
                result = self.cloud_llm.generate(prompt)
                answer_text = result.text
                tokens_per_second = None
                model_name = result.model_name
            else:
                if not self.local_llm.is_ready:
                    raise LocalLLMUnavailableError(self.local_llm.error_message)
                result = self.local_llm.generate(prompt)
                answer_text = result.text
                tokens_per_second = result.tokens_per_second
                model_name = "Llama-3.2-3B-Instruct (local, GGUF)"

        sources = sorted({c["page_number"] for c in context_chunks})
        total = sum(timer.elapsed.values())

        avg_relevance = (
            sum(c["score"] for c in context_chunks) / len(context_chunks)
            if context_chunks
            else None
        )

        metrics = RunMetrics(
            embedding_seconds=timer.elapsed.get("embedding"),
            retrieval_seconds=timer.elapsed.get("retrieval"),
            generation_seconds=timer.elapsed.get("generation"),
            total_seconds=total,
            chunk_count=len(self.chunks),
            tokens_per_second=tokens_per_second,
            active_execution_provider=self.embedder.active_provider,
            model_name=model_name,
            extra={"avg_relevance_score": avg_relevance} if avg_relevance is not None else {},
        )

        return answer_text, sources, metrics

    # ---------------- Summarize ----------------

    def summarize(self, use_cloud: bool = False):
        return self._generate_from_full_document(
            single_pass_builder=prompts.build_summary_prompt,
            reduce_builder=prompts.build_reduce_summary_prompt,
            use_cloud=use_cloud,
        )

    # ---------------- Insights ----------------

    def extract_insights(self, use_cloud: bool = False):
        return self._generate_from_full_document(
            single_pass_builder=prompts.build_insights_prompt,
            reduce_builder=prompts.build_reduce_insights_prompt,
            use_cloud=use_cloud,
        )

    def _generate_from_full_document(self, single_pass_builder, reduce_builder, use_cloud: bool):
        """
        Produces a whole-document result (summary or insights).

        Strategy (see project spec, "Document Summarization" -- do NOT
        just truncate to the first N characters of a large document):
          - If the whole document fits within MAX_SUMMARY_CONTEXT_CHARS,
            do a single generation pass over all of it (cheapest, and
            already covers the entire document -- no need for map-reduce
            overhead on small/medium documents).
          - If it doesn't fit, use map-reduce: split the document into
            budget-sized segments, run a cheap "map" extraction pass over
            EACH segment (so no part of the document is silently dropped),
            then a final "reduce" pass that synthesizes all the partial
            extractions into one coherent whole-document result.
        """
        if not self.chunks:
            raise IngestionError("No document has been ingested yet.")

        total_chars = sum(len(c.text) for c in self.chunks)
        timer = Timer()

        if total_chars <= MAX_SUMMARY_CONTEXT_CHARS:
            context_chunks = [{"text": c.text, "page_number": c.page_number} for c in self.chunks]
            prompt = single_pass_builder(context_chunks)
            with timer.measure("generation"):
                text, tokens_per_second = self._call_llm(prompt, use_cloud, max_tokens=768)
            chunk_count = len(self.chunks)
        else:
            segments = self._split_chunks_into_segments(self.chunks, MAX_SUMMARY_CONTEXT_CHARS)
            partial_summaries = []
            map_tps = []
            with timer.measure("generation"):
                for segment in segments:
                    seg_context = [{"text": c.text, "page_number": c.page_number} for c in segment]
                    map_prompt = prompts.build_map_chunk_prompt(seg_context)
                    partial_text, tps = self._call_llm(map_prompt, use_cloud, max_tokens=300)
                    partial_summaries.append(partial_text)
                    if tps:
                        map_tps.append(tps)

                reduce_prompt = reduce_builder(partial_summaries)
                text, reduce_tps = self._call_llm(reduce_prompt, use_cloud, max_tokens=800)
                if reduce_tps:
                    map_tps.append(reduce_tps)

            tokens_per_second = (sum(map_tps) / len(map_tps)) if map_tps else None
            chunk_count = len(self.chunks)

        metrics = RunMetrics(
            generation_seconds=timer.elapsed.get("generation"),
            total_seconds=timer.elapsed.get("generation"),
            chunk_count=chunk_count,
            tokens_per_second=tokens_per_second,
        )
        return text, metrics

    def _call_llm(self, prompt: str, use_cloud: bool, max_tokens: int = 512):
        """Shared single-call helper for both summarize/insights code paths."""
        if use_cloud:
            if self.cloud_llm is None or not self.cloud_llm.is_ready:
                raise CloudLLMUnavailableError(
                    self.cloud_llm.error_message if self.cloud_llm else "Cloud LLM not configured."
                )
            result = self.cloud_llm.generate(prompt)
            return result.text, None
        else:
            if not self.local_llm.is_ready:
                raise LocalLLMUnavailableError(self.local_llm.error_message)
            result = self.local_llm.generate(prompt, max_tokens=max_tokens)
            return result.text, result.tokens_per_second

    @staticmethod
    def _split_chunks_into_segments(chunks: List[Chunk], max_chars: int) -> List[List[Chunk]]:
        """
        Splits the full ordered list of document chunks into consecutive
        segments, each as large as possible without exceeding max_chars.
        Used by the map-reduce path so every part of a large document is
        covered by some segment (no silent truncation).
        """
        segments: List[List[Chunk]] = []
        current: List[Chunk] = []
        used = 0
        for chunk in chunks:
            if used + len(chunk.text) > max_chars and current:
                segments.append(current)
                current = []
                used = 0
            current.append(chunk)
            used += len(chunk.text)
        if current:
            segments.append(current)
        return segments

    @staticmethod
    def _filter_and_dedupe(results, top_k: int, min_score: float = MIN_RELEVANCE_SCORE):
        """
        Applies a relevance-score floor and removes duplicate/near-duplicate
        chunk text before the results are handed to the LLM as context.

        Rationale (see project spec, "Retrieval / RAG Quality"):
          - A relevance threshold stops barely-related chunks from being
            passed as "context," which is what lets the model correctly
            say "not found in the document" instead of grounding a guess
            in noise.
          - De-duplication matters because overlapping chunk windows
            (see core/ingestion/chunker.py's overlap) can occasionally
            retrieve two chunks whose text is identical or near-identical,
            wasting context budget without adding information.

        Args:
            results: List[SearchResult] from VectorStore.search, already
                sorted by descending similarity.
            top_k: max number of chunks to keep after filtering.
            min_score: minimum cosine similarity to keep a chunk.

        Returns:
            Filtered, de-duplicated list of SearchResult, length <= top_k.
            Falls back to the single best-scoring result even if it is
            below the threshold, so a genuinely relevant document never
            yields a completely empty context due to an overly strict
            cutoff.
        """
        seen_texts = set()
        filtered = []
        for result in results:
            normalized = " ".join(result.metadata.text.split()).lower()
            if normalized in seen_texts:
                continue
            if result.score < min_score:
                continue
            seen_texts.add(normalized)
            filtered.append(result)
            if len(filtered) >= top_k:
                break

        if not filtered and results:
            # Nothing cleared the relevance bar -- keep the single best
            # match anyway so a real (if weak) signal isn't thrown away
            # entirely. The LLM's grounding instruction still applies, so
            # it will say "not found" if this one chunk isn't sufficient.
            filtered = [results[0]]

        return filtered


