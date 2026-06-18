# Design Decisions — Retrieval Layer

This document covers methodology decisions for the **retrieval layer**: vector store, embedding, sparse retrieval, hybrid fusion, and the relevance gate.

For ingestion decisions (chunking, metadata extraction), see `DESIGN_DECISIONS.md`.
For generation decisions (response engine, citations), see `GENERATION_DECISIONS.md`.

---

## Decision 1: Qdrant over ChromaDB

**Original choice:** ChromaDB (local persistent).

**Problem:** ChromaDB metadata is flat key-value only. Lists require workarounds (pipe-delimited strings). Filter API is limited for complex cross-doctor queries.

**Qdrant advantages:**
- Native list/array payloads — `topic_tags: ["MIS-C", "staff burnout"]` stored as-is, queried with `MatchAny`
- Full filter API: nested `must`/`should`/`must_not` conditions, `MatchValue`, `MatchText`, `MatchAny`
- Runs fully local in on-disk mode (`QdrantClient(path="./vector_store")`) — no server process needed
- `QDRANT_URL` + `QDRANT_API_KEY` env vars switch to Qdrant Cloud for Render deployment
- Point IDs are UUID5 (deterministic from chunk_id) — upsert is idempotent, safe to re-ingest

**Filter example for cross-doctor queries:**
```python
Filter(must=[
    FieldCondition(key="specialty", match=MatchValue(value="Pediatrician")),
    FieldCondition(key="location", match=MatchText(text="Mumbai")),
    FieldCondition(key="topic_tags", match=MatchAny(any=["vaccination"])),
])
```

---

## Decision 2: Why Ingest-Time Metadata is Non-Negotiable

Hybrid retrieval (dense + BM25) ranks by semantic/keyword similarity. It has no concept of "this chunk is from a pediatrician" or "this chunk is from Mumbai." Pre-filtered metadata in Qdrant narrows the candidate pool *before* retrieval runs.

| Query type | Hybrid retrieval alone | With Qdrant metadata filters |
|---|---|---|
| "What did doctors say about staff burnout?" | ✅ Dense similarity finds it | ✅ Same |
| "What did pediatricians in Mumbai observe?" | ❌ Cannot enforce specialty + city | ✅ Filter then retrieve |
| "Compare Dr. Singh vs Dr. Patel on vaccination" | ❌ Mixes both doctors arbitrarily | ✅ Two filtered retrievals |
| "What happened during the Delta wave?" | ⚠️ BM25 catches "Delta" but misses paraphrases | ✅ `wave_reference contains "Delta"` |

Doc-level metadata (`doctor_name`, `specialty`, `location`, `geographic_region`) is load-bearing infrastructure, not optional enrichment.

---

## Decision 3: BM25 Source of Truth is Qdrant

`rank-bm25` has no incremental update API. Every new document requires a full index rebuild. On each ingest:

1. Fetch all `(chunk_id, text)` pairs from Qdrant via `collection.scroll()` (paginated at 1000 records)
2. Rebuild `BM25Okapi` from all texts
3. Pickle `{"bm25": instance, "chunk_ids": [ordered list]}` to `vector_store/bm25_index.pkl`

The `chunk_ids` list is stored alongside the BM25 instance so BM25 score positions map back to Qdrant chunk IDs. At ~200 chunks (15 docs), rebuild takes < 1 second. Qdrant is the single source of truth — no shadow text store that can drift.

---

## Decision 4: Hybrid Retrieval — Dense + BM25 → RRF

**Why hybrid (not dense-only):**
Dense retrieval excels at semantic similarity but misses exact terms. BM25 catches exact medical terms ("MIS-C", "Omicron", doctor names) that dense embeddings dilute in vector space. Hybrid combines both signals.

**Reciprocal Rank Fusion (RRF) with k=60:**
```
RRF_score(chunk) = Σ 1 / (k + rank_in_retriever)
```
Each retriever votes on its top candidates. Chunks that rank highly in both retrievers get the highest fused scores. k=60 is the standard value (flattens early ranks, prevents one dominator).

**Candidate expansion:**
Both retrievers return `DENSE_CANDIDATES = 20` candidates. RRF is computed over the union of both candidate lists. Final top-K (default 5) is selected from the fused ranking.

**BM25 and Qdrant filters:**
BM25 operates on the full index and does not respect Qdrant metadata filters. When a filter is active (e.g., `specialty="Pediatrician"`), dense retrieval (which does respect filters) dominates the final result set. BM25-only results that are not in the filtered dense candidate pool are dropped. This is acceptable for the demo — at 200 chunks, `DENSE_CANDIDATES=20` covers the most relevant filtered candidates.

---

## Decision 5: Relevance Gate

**Purpose:** Prevent the LLM from generating answers when no relevant chunks exist. Without a gate, the model hallucinates from general knowledge rather than the corpus.

**Implementation:** After RRF fusion, if the best fused score < `RELEVANCE_THRESHOLD`, return an empty list and surface a fallback message in the UI.

**Threshold calibration:**

RRF scores are bounded by the fusion math (`k=60`, two retrievers, `DENSE_CANDIDATES=20`):

| Threshold | Score range | Effect |
|---|---|---|
| 0.013 | Floor (rank #20 in one retriever only) | Passes almost everything |
| **0.018** | Mid-low | Passes documents that rank well in at least one retriever — **current setting** |
| 0.025 | Mid | Requires presence in both retrievers at reasonable rank |
| 0.033 | Ceiling (rank #1 in both) | Only the very top result passes |

- `RELEVANCE_THRESHOLD = 0.018` set in `config.py`
- Initial value of `0.25` was wrong — it exceeded the RRF ceiling and blocked every query
- **Production path:** Tune empirically by running queries with known-irrelevant topics and finding the score below which results are garbage

---

## Decision 6: Embeddings — sentence-transformers all-MiniLM-L6-v2

- Local inference, no API key, no latency from network
- 384-dim vectors — small enough for local Qdrant, large enough for good semantic resolution
- Same model reused by `chonkie.SemanticChunker` during chunking — chunking and retrieval share the same embedding space, so topic-boundary detection and retrieval are semantically consistent
- Normalised embeddings (L2 norm = 1) — cosine similarity reduces to dot product, which Qdrant computes efficiently

---

## Implementation Status

All three retrieval files are implemented and aligned with the decisions above.

| File | Status | Notes |
|---|---|---|
| `src/retrieval/embedder.py` | ✅ Done | Lazy singleton model, `rebuild_bm25` fetches from Qdrant + pickles, `bm25_available` guard |
| `src/retrieval/vector_store.py` | ✅ Done | Dual-mode client, UUID5 point IDs, all 11 metadata fields, paginated scroll |
| `src/retrieval/hybrid_retriever.py` | ✅ Done | Dense + BM25 → RRF fusion, relevance gate, BM25 graceful degradation, filter compliance |

**Known non-issue:** `client.search()` in `vector_store.py` is deprecated in `qdrant-client >= 1.7` (prefer `query_points()`). Left as-is — acceptable for the demo window.

---

## Open Questions

1. **RELEVANCE_THRESHOLD:** Set to `0.018`. Re-tune empirically after first full ingest + query run if needed.
2. **DENSE_CANDIDATES:** 20 is a starting guess. If cross-doctor filtered queries miss chunks, increase to 50.
3. **RRF_K:** Standard value of 60 — no reason to change without benchmarking data.
