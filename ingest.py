"""
CLI batch ingestion script.

Usage:
    python ingest.py               # process all PDFs/DOCXs in data/raw/
    python ingest.py path/to/file  # process a single file
"""

import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Load .env before any Anthropic/config imports
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from config import DATA_DIR
from src.ingestion.pipeline import ingest_document
from src.retrieval.vector_store import get_client

SUPPORTED = {".pdf", ".docx"}


def main() -> None:
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        targets = [p for p in sorted(DATA_DIR.iterdir()) if p.suffix.lower() in SUPPORTED]

    if not targets:
        print(f"No supported files found in {DATA_DIR}. Drop PDFs or DOCXs there and retry.")
        sys.exit(1)

    print(f"\nIngesting {len(targets)} file(s)...\n")
    client = get_client()

    results = []
    for path in targets:
        if not path.exists():
            print(f"  [SKIP] {path} — file not found")
            continue
        result = ingest_document(path, client=client)
        results.append(result)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"{'FILE':<45} {'CHUNKS':>6}  STATUS")
    print("─" * 60)
    total_chunks = 0
    for r in results:
        chunks = r.get("chunks", 0)
        total_chunks += chunks
        tag = r.get("doctor_name", "") or r.get("status", "")
        print(f"  {r['source_file']:<43} {chunks:>6}  {tag}")
    print("─" * 60)
    print(f"  {'TOTAL':<43} {total_chunks:>6}\n")


if __name__ == "__main__":
    main()
