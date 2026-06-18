# C-RAG — Corrective RAG Pipeline Design

This document describes the Corrective RAG (CRAG) pipeline that is the core of the response engine.

**Paper:** Yan et al. 2024 — *Corrective Retrieval Augmented Generation* (https://arxiv.org/abs/2401.15884)
**Implementation:** `src/generation/crag_pipeline.py`

---

## Why CRAG

The original pipeline had a single relevance gate: if the best RRF-fused score fell below `0.018`, return a fallback. This is a mechanical threshold with no semantic awareness — it cannot distinguish between "no results found" and "results found but weakly relevant."

CRAG replaces this gate with an **LLM-based retrieval evaluator** that reads each retrieved chunk and the user's question, assigns a relevance score, and routes the pipeline accordingly. This directly satisfies the assignment requirement:

> "The response engine should be robust enough to identify if the retrieved chunks are not helpful."

---

## What CRAG Does

CRAG inserts a **chunk grader** between retrieval and generation. The grader assigns a score `[0.0, 1.0]` to each retrieved chunk. Based on scores, one of three verdicts is issued:

| Verdict | Condition | Action |
|---|---|---|
| `CORRECT` | ≥ 1 chunk scores ≥ 0.7 | Generate from all chunks |
| `AMBIGUOUS` | Some chunks 0.3–0.7, none ≥ 0.7 | Generate from good_chunks only (score ≥ 0.3) |
| `INCORRECT` | All chunks score < 0.3 | Return fallback message, no generation |

### Our Closed-Domain Adaptation

Standard CRAG routes `INCORRECT` to a web search. This system is **closed-domain** (doctor interview corpus only). There is no web search fallback — `INCORRECT` returns a fallback message explaining the limitation.

### CRAG Grader Toggle

The grader can be disabled at runtime via `CRAG_GRADER_ENABLED = False` in `config.py`, or via the **Pipeline settings** toggle in the Streamlit sidebar. When disabled, `bypass_grade_node` sets `overall_verdict=CORRECT` and passes all retrieved chunks straight to generation — useful for A/B comparison against the graded path.

### Removed: Sentence-Level Refinement

An earlier version included `refine_context_node` — a Haiku-powered sentence-by-sentence filter that stripped irrelevant sentences from good chunks. It was removed because:
- It added **70+ seconds of latency** per query (~20–30 serial Haiku API calls)
- Chunk grading already filters irrelevant chunks — sentence filtering provided no meaningful quality gain on top of that
- It stripped doctor attribution from chunk text, causing the generator to incorrectly disclaim knowledge of doctors that were in fact retrieved

---

## Graph Architecture

```
START
  │
  ▼
[classify_node]
  Claude Haiku classifies intent + optionally generates direct response
  Returns: {classification: "direct"|"retrieval", text: str}
  │
  ├── direct ─────────────────────────────────────► [persist_node]
  │   (greetings, "who are you", small talk —                │
  │    Haiku writes the reply in-place, no retrieval)        │
  │                                                          │
  └── retrieval                                              │
          │                                                  │
          ▼                                                  │
  [retrieve_node]                                           │
  Hybrid BM25 + dense + RRF → top-5 chunks                 │
  chunks=[] → INCORRECT (no_chunks shortcut, skip grader)  │
          │                                                  │
          ▼                                                  │
  [grade_chunks_node]  ←── or [bypass_grade_node]          │
  Claude Haiku scores each chunk 0–1                        │
  Derives overall_verdict + good_chunks (score ≥ 0.3)      │
          │                                                  │
          ├── INCORRECT ──────────────────► [fallback_node] │
          │                                        │         │
          └── CORRECT / AMBIGUOUS                  │         │
                  │                                │         │
                  ▼                                │         │
          [load_history_node]                      │         │
          get_history + get_thread_summary         │         │
                  │                                │         │
                  ▼                                │         │
          [generate_node]                          │         │
          Claude Sonnet + summary + history        │         │
          + chunks with doctor headers             │         │
                  │                                │         │
                  └────────────┬───────────────────┘         │
                               │                             │
                               └──────────────┬─────────────┘
                                              ▼
                                      [persist_node]
                                      add_message(user + assistant)
                                      set_thread_title on first turn
                                              │
                                              ▼
                                      [summarize_node]
                                      Haiku folds overflowed messages
                                      into rolling summary (conditional)
                                              │
                                             END
```

---

## State Schema

```python
class RAGState(TypedDict):
    # ── Caller inputs ──────────────────────────────────────────────────────
    query: str               # user's question
    thread_id: str           # SQLite conversation thread
    qdrant_filter: Optional  # metadata filter (e.g., filter by doctor_name)

    # ── Intent ─────────────────────────────────────────────────────────────
    intent: str              # "direct" | "retrieval"

    # ── Retrieval ──────────────────────────────────────────────────────────
    chunks: list[dict]       # [{chunk_id, score, payload}] from hybrid_retriever

    # ── Grading ────────────────────────────────────────────────────────────
    chunk_verdicts: list[dict]   # [{chunk_id, verdict, score, reason}]
    overall_verdict: str         # "CORRECT" | "AMBIGUOUS" | "INCORRECT"
    good_chunks: list[dict]      # chunks where score >= CRAG_LOWER_TH (0.3)

    # ── Generation ─────────────────────────────────────────────────────────
    history: list[dict]      # [{role, content}] last N turns from SQLite
    summary: str             # rolling summary of turns that fell off the window
    answer: str              # final answer (Sonnet for retrieval, Haiku for direct)
    chunks_used: list[dict]  # source payloads shown as sources in UI

    # ── Result ─────────────────────────────────────────────────────────────
    fallback: bool
    fallback_reason: str     # "no_chunks" | "all_irrelevant" | "llm_error"

    # ── Latency ────────────────────────────────────────────────────────────
    latency_retrieve: float  # seconds
    latency_grade: float     # seconds
    latency_generate: float  # seconds
```

---

## Node Details

### `classify_node`
- Model: **Claude Haiku** via `_get_haiku().with_structured_output(IntentResponse)`
- Pydantic schema: `IntentResponse(classification: "direct"|"retrieval", text: str)`
- If `classification == "direct"`: `text` is Haiku's reply (greetings, persona intro, small talk) — sets `answer`, `chunks_used=[]`, `fallback=False`, routes to `persist`
- If `classification == "retrieval"`: `text` is the user's original query — routes to `retrieve`
- On exception → defaults to `"retrieval"` (safe fallback, never drops a real question)

**Persona prompt (Pulse):**
```
You are Pulse, a medical interview insights assistant with deep familiarity with
a curated collection of in-depth conversations with healthcare professionals.

For direct messages: respond naturally and warmly as Pulse.
For retrieval queries: copy the user's message into 'text'.
When in doubt, choose 'retrieval'.
```

### `retrieve_node`
- Calls `src/retrieval/hybrid_retriever.retrieve()` (unchanged)
- If empty list returned → sets `overall_verdict="INCORRECT"`, `fallback_reason="no_chunks"`, skips grader
- Logs latency

### `grade_chunks_node`
- Model: **Claude Haiku** (`claude-haiku-4-5-20251001`) via module-level singleton `_get_haiku()`
- One call per chunk (5 calls for TOP_K=5), serial
- Structured output via Pydantic `ChunkGrade(score, verdict, reason)`
- Logs latency

**Grader system prompt:**
```
You are evaluating a retrieved Q&A chunk from a COVID-19 doctor interview transcript.
Score how well this chunk answers the user's question.

Score guide:
  0.8–1.0 → CORRECT    chunk directly and fully addresses the question
  0.3–0.8 → AMBIGUOUS  partial, tangential, or incomplete information
  0.0–0.3 → INCORRECT  chunk is irrelevant to the question

Output JSON only: {"score": float, "verdict": str, "reason": str}
```

**Verdict derivation:**
- Any score ≥ `CRAG_UPPER_TH` (0.7) → `CORRECT`
- All scores < `CRAG_LOWER_TH` (0.3) → `INCORRECT`
- Otherwise → `AMBIGUOUS`

### `load_history_node`
- Calls `get_history(thread_id, HISTORY_TURNS)` → last 6 messages (raw)
- Calls `get_thread_summary(thread_id)` → existing rolling summary (may be empty)
- Populates both `history` and `summary` fields in state

### `generate_node`
- Model: **Claude Sonnet** (`claude-sonnet-4-6`) via `BaseLLMEngine.complete()`
- `prompt_chunks` = `good_chunks` if AMBIGUOUS, else all `chunks`
- Each chunk formatted via `_format_chunks()` with doctor attribution header:
  ```
  [Dr. Name | Specialty | Location]
  <chunk text>
  ```
- Generation prompt structure (sections omitted if empty):
  ```
  ## Prior Conversation Summary
  <rolling summary of turns older than the window>

  ## Recent Conversation
  User: ...
  Assistant: ...

  ## Source Chunks
  [Dr. Name | Specialty | Location]
  <chunk text>
  ...

  ## Question
  <query>
  ```
- Sonnet speaks as **Pulse** — natural attribution ("Dr. Singh mentioned..."), never references "sources" or "chunks"
- On exception → `fallback=True`, `fallback_reason="llm_error"`
- Logs latency

### `fallback_node`
Three distinct fallback messages by `fallback_reason` (written in Pulse's voice):
- `"no_chunks"` → "I don't have anything on that in the interviews I know about..."
- `"all_irrelevant"` → "I couldn't find a strong enough match for that in the interviews..."
- `"llm_error"` → "Something went wrong on my end. Give it another try."

### `persist_node`
- Checks `get_history(thread_id, 1)` before writing — if empty, sets thread title from query (first 60 chars)
- `add_message(thread_id, "user", query)`
- `add_message(thread_id, "assistant", answer, metadata={chunks_used, fallback, latencies})`
- Metadata is stored as JSON and used to restore the sources panel after restart
- Entire body wrapped in `try/except` — failure is logged but does not crash the pipeline
- Both happy path and fallback path converge here

### `summarize_node`
- Runs after `persist_node` on every path (happy + fallback)
- **Trigger:** only calls Haiku when `total_messages > HISTORY_TURNS` and new messages have fallen off the window since the last summary update
- **Incremental:** tracks `summary_msg_count` in the DB — each run only sends Haiku the newly overflowed messages, not the full history from scratch
- Calls Haiku with `_SUMMARIZER_PROMPT`: existing summary + new turns → updated ≤200-word summary
- Writes updated summary and new `summary_msg_count` back to `threads` table via `set_thread_summary()`
- Non-fatal: failure is logged, pipeline result is unaffected

---

## Thresholds & Flags (config.py)

| Constant | Value | Meaning |
|---|---|---|
| `CRAG_GRADER_ENABLED` | `True` / `False` | Master switch — False bypasses grader entirely |
| `CRAG_UPPER_TH` | `0.7` | Score above this → CORRECT verdict |
| `CRAG_LOWER_TH` | `0.3` | Score below this (all chunks) → INCORRECT verdict |
| `METADATA_MODEL` | `claude-haiku-4-5-20251001` | Used for classify + chunk grader + summarizer |
| `GENERATION_MODEL` | `claude-sonnet-4-6` | Used for final answer generation |

---

## LLM Cost Per Query

| Step | Model | Calls | Est. tokens/call | When |
|---|---|---|---|---|
| Intent classify + direct reply | Haiku | 1 | ~200 in + 50 out | Every query |
| Chunk grading | Haiku | 5 (TOP_K) | ~400 in + 50 out | Retrieval queries only (grader enabled) |
| Generation | Sonnet | 1 | ~2000 in + 500 out | Every non-fallback retrieval query |
| Summarization | Haiku | 1 | ~800 in + 150 out | Only when new messages overflow the window |

Sentence filtering (Haiku, ~20–30 calls/query) was removed — it dominated latency with no quality benefit.

The summarization call is conditional: it fires only after turn 6 and only when the overflow count has grown since the last summary. For most demo sessions (≤6 turns) it never fires.

Direct responses (greetings, "who are you") skip retrieval, grading, and Sonnet entirely — cost is one lightweight Haiku call.

---

## Singletons & Caching

- **Haiku client** (`_haiku`): module-level singleton in `crag_pipeline.py` — initialised once, reused across all queries
- **Qdrant client** (`_client`): module-level singleton in `vector_store.py` — prevents file lock contention on Streamlit hot-reload
- **Sentence-transformers model** (`_model`): module-level singleton in `embedder.py`
- **CRAG graph + Sonnet LLM**: cached via `@st.cache_resource` in `app.py` — compiled once per process

---

## Public API

```python
from src.generation.crag_pipeline import build_crag_graph

# At app startup — compile once, reuse across all queries
graph = build_crag_graph(qdrant_client=client, llm=llm_engine)

# Per query
result = graph.invoke({
    "query": "How did Dr. Singh handle pediatric cases in the first wave?",
    "thread_id": "some-uuid",
    "qdrant_filter": None,
})

result["answer"]          # str — clean answer from Sonnet
result["chunks_used"]     # list[dict] — payloads with doctor/specialty/location/text
result["fallback"]        # bool
result["latency_retrieve"] # float (seconds)
result["latency_grade"]    # float (seconds)
result["latency_generate"] # float (seconds)
```

---

## What Changed vs Original Pipeline

| Aspect | Before (response_engine.py) | After (crag_pipeline.py) |
|---|---|---|
| Persona | None — generic research assistant | **Pulse** — named medical interview insights assistant |
| Intent handling | Every query hits retrieval | `classify_node` handles greetings/small talk directly, skips pipeline |
| Relevance gate | RRF score < 0.018 (mechanical) | LLM scores each chunk semantically |
| Grader toggle | N/A | `CRAG_GRADER_ENABLED` in config + sidebar toggle for A/B comparison |
| Fallback granularity | Binary: yes/no | 3 reasons: no_chunks / all_irrelevant / llm_error |
| Fallback voice | Robotic system messages | Natural Pulse persona voice |
| Context sent to generator | Raw chunk text | Chunks formatted with `[Doctor | Specialty | Location]` headers |
| AMBIGUOUS handling | No concept — all-or-nothing | Generates from good_chunks only |
| Citations | Inline `[CITATION: "..."]` markers + regex parse | Removed — graded chunks shown as sources directly |
| Sentence filtering | `refine_context_node` (70s+, ~30 Haiku calls) | Removed — chunk grading sufficient |
| Orchestration | Linear function | LangGraph stateful graph |
| Latency visibility | None | Per-node timing logged + shown in UI |

---

## Metadata Filtering

`qdrant_filter` is a first-class field in `RAGState` and is wired through the entire pipeline to Qdrant's `query_filter` in `search_dense()`. The infrastructure is complete — Qdrant enforces the filter server-side before returning dense candidates, and BM25 candidates that don't appear in the filtered dense result set are silently dropped at the payload-lookup step.

**Current status: filtering is never activated.** `app.py` always passes `qdrant_filter=None`, so every query searches all 276 chunks across all 20 doctors.

Filterable fields on every chunk: `doctor_name`, `specialty`, `geographic_region`, `location`, `wave_reference`, `topic_tags`.

To activate filtering, build a `Filter` object before invoking the graph:

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

result = graph.invoke({
    "query": "What did she say about pediatric cases?",
    "thread_id": tid,
    "qdrant_filter": Filter(
        must=[FieldCondition(key="doctor_name", match=MatchValue(value="Dr. Anjali Singh"))]
    ),
})
```

The natural next step is a doctor-selector dropdown in the sidebar (populated from the live Qdrant collection) that optionally scopes a conversation to one doctor.

---

## Files

| File | Role |
|---|---|
| `src/generation/crag_pipeline.py` | Graph definition, all nodes, `build_crag_graph()`. Fully self-contained. |
| `src/generation/llm_engine.py` | LLM abstraction — unchanged |
| `src/retrieval/hybrid_retriever.py` | Retrieval — unchanged |
| `src/memory/conversation.py` | SQLite memory — threads + messages + rolling summary |
| `config.py` | `CRAG_UPPER_TH`, `CRAG_LOWER_TH` |
| `app.py` | Streamlit UI: Tab 1 (ingest), Tab 2 (chat + sources panel + latency display + thread management) |
