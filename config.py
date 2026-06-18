import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
CONVERSATION_STORE_DIR = BASE_DIR / "conversation_store"
BM25_INDEX_PATH = VECTOR_STORE_DIR / "bm25_index.pkl"
DB_PATH = CONVERSATION_STORE_DIR / "threads.db"

# ── Models ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
METADATA_MODEL = "claude-haiku-4-5-20251001"
GENERATION_MODEL = "claude-sonnet-4-6"

# ── Qdrant ─────────────────────────────────────────────────────────────────────
# Local on-disk (default for demo). Set QDRANT_URL + QDRANT_API_KEY env vars
# to switch to Qdrant Cloud for Render/production deployment.
QDRANT_COLLECTION = "interview_chunks"
QDRANT_PATH = str(VECTOR_STORE_DIR)          # used when QDRANT_URL is not set
QDRANT_URL = os.getenv("QDRANT_URL")         # e.g. "https://xyz.qdrant.tech"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # cloud API key

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_TOKEN_CEILING = 300             # max tokens per sub-chunk
MIN_CHUNK_TOKENS = 100                # sub-chunks below this are merged into their predecessor
SLIDING_WINDOW_OVERLAP = 50           # token overlap for Tier 3 fallback
SPEAKER_DISCOVERY_WINDOW = 2000       # chars scanned for speaker label discovery
SPEAKER_MIN_FREQUENCY = 2             # label must appear this many times to be a speaker
SEMANTIC_SIMILARITY_THRESHOLD = 0.5  # chonkie SemanticChunker boundary sensitivity (0–1)

# ── Retrieval ──────────────────────────────────────────────────────────────────
RRF_K = 60
RELEVANCE_THRESHOLD = 0.018
TOP_K = 5
DENSE_CANDIDATES = 20   # candidates each retriever returns before RRF fusion

# ── Generation ─────────────────────────────────────────────────────────────────
HISTORY_TURNS = 6
MAX_RESPONSE_TOKENS = 1024

# ── CRAG Grader ────────────────────────────────────────────────────────────────
CRAG_GRADER_ENABLED = False   # False → bypass grader, treat all chunks as CORRECT
CRAG_UPPER_TH = 0.7          # chunk score >= this → CORRECT
CRAG_LOWER_TH = 0.3          # all chunks below this → INCORRECT
