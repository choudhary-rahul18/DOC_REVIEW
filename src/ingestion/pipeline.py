import logging
from pathlib import Path
from typing import Callable, Optional

from src.ingestion.chunker import ParsedDocument, chunk_document
from src.ingestion.metadata_extractor import extract_metadata
from src.retrieval.embedder import embed_texts, rebuild_bm25
from src.retrieval.vector_store import ensure_collection, get_client, upsert_chunks

logger = logging.getLogger(__name__)


def ingest_document(
    file_path: Path,
    client=None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Full ingestion pipeline for one document.
    Called from ingest.py (batch) and app.py Tab 1 (UI upload).
    Returns summary stats dict.
    """
    def _progress(msg: str) -> None:
        logger.info("[pipeline] %s", msg)
        if progress_callback:
            progress_callback(msg)

    owns_client = client is None
    if owns_client:
        client = get_client()

    try:
        ensure_collection(client)

        # ── Step 1: Parse ──────────────────────────────────────────────────────
        _progress(f"Parsing {file_path.name}...")
        from src.ingestion.parser import parse_document
        text = parse_document(file_path)
        parsed = ParsedDocument(text=text, source_file=file_path.name)

        if not parsed.text.strip():
            logger.warning("[pipeline] %s: parser returned empty text — skipping", file_path.name)
            return {"source_file": file_path.name, "chunks": 0, "status": "empty"}

        # ── Step 2: Chunk ──────────────────────────────────────────────────────
        _progress("Chunking...")
        chunks = chunk_document(parsed)

        if not chunks:
            logger.warning("[pipeline] %s: chunker returned no chunks — skipping", file_path.name)
            return {"source_file": file_path.name, "chunks": 0, "status": "no_chunks"}

        parse_tier = chunks[0].parse_tier

        # ── Step 3: Metadata extraction ────────────────────────────────────────
        _progress(f"Extracting metadata for {len(chunks)} chunks (Claude Haiku)...")
        enriched = extract_metadata(chunks)

        # ── Step 4: Embed ──────────────────────────────────────────────────────
        _progress("Embedding chunks...")
        texts = [c.text for c in enriched]
        vectors = embed_texts(texts)

        # ── Step 5: Upsert to Qdrant ───────────────────────────────────────────
        _progress("Upserting to Qdrant...")
        upsert_chunks(client, enriched, vectors)

        # ── Step 6: Rebuild BM25 ───────────────────────────────────────────────
        _progress("Rebuilding BM25 index...")
        rebuild_bm25(client)

        doctor_name = enriched[0].doctor_name if enriched else ""
        summary = {
            "source_file": file_path.name,
            "chunks": len(enriched),
            "parse_tier": parse_tier,
            "doctor_name": doctor_name,
            "status": "ok",
        }
        _progress(f"Done: {len(enriched)} chunks ingested ({doctor_name or file_path.stem})")
        return summary

    except Exception as e:
        logger.error("[pipeline] %s: unexpected error: %s", file_path.name, e, exc_info=True)
        return {"source_file": file_path.name, "chunks": 0, "status": f"error: {e}"}
