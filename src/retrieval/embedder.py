import logging
import pickle
from pathlib import Path

from config import BM25_INDEX_PATH, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("[embedder] Loaded model: %s", EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


def rebuild_bm25(client) -> None:
    """Fetch all chunks from Qdrant and rebuild BM25 index. Called after every ingest."""
    from rank_bm25 import BM25Okapi
    from src.retrieval.vector_store import scroll_all_chunks

    records = scroll_all_chunks(client)
    if not records:
        logger.warning("[embedder] No chunks in Qdrant — BM25 index not built")
        return

    chunk_ids = [r["chunk_id"] for r in records]
    tokenized = [r["text"].lower().split() for r in records]
    bm25 = BM25Okapi(tokenized)

    Path(BM25_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)

    logger.info("[embedder] BM25 index rebuilt: %d chunks indexed", len(chunk_ids))


def load_bm25():
    """Return (BM25Okapi, chunk_ids). Raises FileNotFoundError if index not built yet."""
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunk_ids"]


def bm25_available() -> bool:
    return Path(BM25_INDEX_PATH).exists()
