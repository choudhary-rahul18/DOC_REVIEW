#!/usr/bin/env python3
"""
Chunking review script — runs parser → chunker on PDF/DOCX files and saves results as JSON.
No API calls, no Qdrant. Pure local processing.

Usage:
    python review_chunks.py                        # scans docs/ folder
    python review_chunks.py path/to/folder         # scans given folder
    python review_chunks.py path/to/file.pdf       # single file
"""

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)

from src.ingestion.chunker import ParsedDocument, chunk_document
from src.ingestion.parser import parse_document

DEFAULT_INPUT = Path(__file__).parent / "docs"
OUTPUT_DIR = Path(__file__).parent / "chunk_reviews"

TIER_NAMES = {
    1: "Tier 1 — speaker regex + semantic sub-chunking",
    2: "Tier 2 — semantic chunking on whole doc (fallback)",
    3: "Tier 3 — token window (last resort)",
}

SEP = "─" * 72
THICK = "=" * 72


def review_file(file_path: Path) -> dict:
    """Parse and chunk one file. Returns a dict ready for JSON serialisation."""
    print(f"\n{THICK}")
    print(f"  FILE: {file_path.name}")
    print(THICK)

    # ── Parse ──────────────────────────────────────────────────────────────────
    print("\nParsing...", end=" ", flush=True)
    text = parse_document(file_path)
    print(f"{len(text):,} chars")
    print(f"Preview (first 300 chars):\n  {text[:300].replace(chr(10), ' ')!r}\n")

    # ── Chunk ──────────────────────────────────────────────────────────────────
    print("Chunking (loads sentence-transformers model on first run — ~30s)...", flush=True)
    parsed = ParsedDocument(text=text, source_file=file_path.name)
    chunks = chunk_document(parsed)

    if not chunks:
        print("\nERROR: Chunker returned no chunks.")
        return {"source_file": file_path.name, "error": "no chunks produced", "chunks": []}

    tier = chunks[0].parse_tier
    unique_qa = len({c.qa_index for c in chunks if c.qa_index >= 0})
    avg_len = sum(len(c.text) for c in chunks) // len(chunks)

    print(f"\n{THICK}")
    print(f"  RESULTS")
    print(THICK)
    print(f"  Parse tier : {TIER_NAMES.get(tier, str(tier))}")
    print(f"  Chunks     : {len(chunks)}")
    if unique_qa:
        print(f"  Q&A pairs  : {unique_qa}")
    print(f"  Avg length : {avg_len:,} chars per chunk")
    print()

    # ── Print each chunk ────────────────────────────────────────────────────────
    for chunk in chunks:
        print(SEP)
        meta = f"[chunk {chunk.chunk_index:02d}]  qa={chunk.qa_index}  tier={chunk.parse_tier}  {len(chunk.text):,} chars"
        print(meta)

        if chunk.question_header:
            q = chunk.question_header
            print(f"  Q: {q[:120]}{'...' if len(q) > 120 else ''}")

        body = chunk.text
        if chunk.question_header and "[Answer excerpt]" in body:
            body = body.split("[Answer excerpt]", 1)[-1].strip()

        preview = body[:400]
        if len(body) > 400:
            preview += f"\n  ... [{len(body) - 400} more chars]"
        for line in preview.splitlines():
            print(f"  {line}")
        print()

    print(THICK)
    print(f"  DONE: {len(chunks)} chunks | {unique_qa} Q&A pairs | {TIER_NAMES.get(tier, str(tier))}")
    print(THICK)

    # ── Build JSON-serialisable result ─────────────────────────────────────────
    return {
        "source_file": file_path.name,
        "parse_tier": tier,
        "parse_tier_name": TIER_NAMES.get(tier, str(tier)),
        "total_chars": len(text),
        "total_chunks": len(chunks),
        "unique_qa_pairs": unique_qa,
        "avg_chunk_length_chars": avg_len,
        "chunks": [asdict(c) for c in chunks],
    }


def collect_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(
        f for f in target.iterdir()
        if f.suffix.lower() in {".pdf", ".docx"}
    )


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT

    if not target.exists():
        print(f"Path not found: {target}")
        sys.exit(1)

    files = collect_files(target)
    if not files:
        print(f"No PDF or DOCX files found in: {target}")
        sys.exit(1)

    print(f"\nFound {len(files)} file(s) to review: {[f.name for f in files]}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    all_results = []

    for file_path in files:
        result = review_file(file_path)
        all_results.append(result)

        # Save per-file JSON
        out_file = OUTPUT_DIR / f"{file_path.stem}_chunks.json"
        out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n  Saved → {out_file.relative_to(Path(__file__).parent)}")

    # Save combined JSON
    combined_out = OUTPUT_DIR / "all_chunks.json"
    combined_out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\n  Combined → {combined_out.relative_to(Path(__file__).parent)}")
    print(f"\nDone. Reviewed {len(files)} file(s). JSONs saved to chunk_reviews/\n")


if __name__ == "__main__":
    main()
