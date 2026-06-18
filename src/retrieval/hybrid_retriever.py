import logging
from typing import Optional

from qdrant_client.models import Filter

from config import DENSE_CANDIDATES, RELEVANCE_THRESHOLD, RRF_K, TOP_K
from src.retrieval.embedder import bm25_available, embed_query, load_bm25
from src.retrieval.vector_store import search_dense

logger = logging.getLogger(__name__)


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def retrieve(
    query: str,
    client,
    top_k: int = TOP_K,
    qdrant_filter: Optional[Filter] = None,
) -> list[dict]:
    """
    Hybrid retrieval: dense (Qdrant cosine) + BM25 sparse → RRF fusion → top-k.

    Returns [] if best fused score < RELEVANCE_THRESHOLD (relevance gate).
    Each item: {"chunk_id": str, "score": float, "payload": dict}

    BM25 does not respect Qdrant filters. For filtered queries, dense results
    (which do respect filters) dominate the final payload lookup — chunks that
    pass BM25 but fail the filter are dropped when their payload is not found
    in the filtered dense result set.
    """
    query_vec = embed_query(query)

    # ── Dense retrieval ────────────────────────────────────────────────────────
    dense_results = search_dense(client, query_vec, DENSE_CANDIDATES, qdrant_filter)
    payload_lookup = {r["chunk_id"]: r["payload"] for r in dense_results}

    # ── BM25 sparse retrieval ──────────────────────────────────────────────────
    sparse_chunk_ids: list[str] = []
    if bm25_available():
        try:
            bm25, chunk_ids = load_bm25()
            scores = bm25.get_scores(query.lower().split())
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            sparse_chunk_ids = [
                chunk_ids[i]
                for i in top_indices[:DENSE_CANDIDATES]
                if scores[i] > 0
            ]
        except Exception as e:
            logger.warning("[retriever] BM25 load/query failed: %s — using dense only", e)
    else:
        logger.warning("[retriever] BM25 index not found — using dense retrieval only")

    # ── RRF fusion ─────────────────────────────────────────────────────────────
    fused: dict[str, float] = {}

    for rank, result in enumerate(dense_results):
        cid = result["chunk_id"]
        fused[cid] = fused.get(cid, 0.0) + _rrf_score(rank)

    for rank, cid in enumerate(sparse_chunk_ids):
        fused[cid] = fused.get(cid, 0.0) + _rrf_score(rank)

    # Sort by fused score descending
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

    # ── RRF score observability ────────────────────────────────────────────────
    logger.info(
        "[retriever] RRF candidates=%d  top scores: %s",
        len(ranked),
        "  ".join(f"{cid[:8]}…={score:.4f}" for cid, score in ranked[:5]) if ranked else "none",
    )

    # ── Relevance gate ─────────────────────────────────────────────────────────
    best_score = ranked[0][1] if ranked else 0.0
    if not ranked or best_score < RELEVANCE_THRESHOLD:
        logger.info(
            "[retriever] Gate BLOCKED — best=%.4f threshold=%.4f — returning empty",
            best_score, RELEVANCE_THRESHOLD,
        )
        return []

    logger.info(
        "[retriever] Gate PASSED — best=%.4f threshold=%.4f",
        best_score, RELEVANCE_THRESHOLD,
    )

    # ── Build final results (filter to chunks with known payloads) ─────────────
    results: list[dict] = []
    for chunk_id, score in ranked:
        if chunk_id not in payload_lookup:
            # BM25-only result with no dense payload; skip (filter compliance)
            continue
        results.append({"chunk_id": chunk_id, "score": score, "payload": payload_lookup[chunk_id]})
        if len(results) >= top_k:
            break

    logger.info(
        "[retriever] Returning %d/%d chunks to generator (dense_pool=%d  sparse_pool=%d)",
        len(results), len(ranked), len(dense_results), len(sparse_chunk_ids),
    )
    return results
