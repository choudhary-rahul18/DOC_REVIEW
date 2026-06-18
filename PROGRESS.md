# Progress Log — Conversational RAG over Doctor Interview Transcripts

**Project:** Job interview take-home assignment
**Started:** 2026-06-18
**Status:** Complete. All backend layers, CRAG pipeline, Pulse persona, and Streamlit UI done.

---

## Files Status

| File | Status | Notes |
|---|---|---|
| `CLAUDE.md` | ✅ Done | Updated: LangChain allowed, build status current |
| `C-RAG.md` | ✅ Done | Full CRAG design doc: architecture, prompts, state schema, thresholds |
| `DESIGN_DECISIONS.md` | ✅ Done | Ingestion: chunking, chonkie, metadata extraction |
| `RETRIEVAL_DECISIONS.md` | ✅ Done | Qdrant, BM25, hybrid RRF, relevance gate |
| `GENERATION_DECISIONS.md` | ✅ Done | Response engine, citations, memory, fallback, LangGraph plan |
| `config.py` | ✅ Done | All constants; added `CRAG_UPPER_TH=0.7`, `CRAG_LOWER_TH=0.3` |
| `requirements.txt` | ✅ Done | All deps pinned |
| `.env.example` | ✅ Done | |
| `src/ingestion/parser.py` | ✅ Done | PDF + DOCX → raw text; 5-step normalisation |
| `src/ingestion/chunker.py` | ✅ Done | 3-tier cascade: speaker regex + SemanticChunker + TokenChunker |
| `src/ingestion/metadata_extractor.py` | ✅ Done | Claude Haiku → doc metadata + per-chunk topic_tags; 4-layer JSON recovery |
| `src/ingestion/pipeline.py` | ✅ Done | Shared orchestrator (called by ingest.py + app.py Tab 1) |
| `src/retrieval/vector_store.py` | ✅ Done | Qdrant CRUD; fixed `search()` → `query_points()` for qdrant-client v1.18 |
| `src/retrieval/embedder.py` | ✅ Done | sentence-transformers batch embed + BM25 rebuild from Qdrant |
| `src/retrieval/hybrid_retriever.py` | ✅ Done | BM25 + dense + RRF fusion + relevance gate |
| `src/memory/conversation.py` | ✅ Done | SQLite thread CRUD + rolling summary: title, metadata, summary, incremental summarize |
| `src/generation/llm_engine.py` | ✅ Done | Provider-agnostic abstraction: Anthropic / Gemini / Ollama + factory |
| `src/generation/response_engine.py` | 🗑 Deleted | Consolidated into `crag_pipeline.py` — constants + helpers moved, dead imports removed |
| `src/generation/crag_pipeline.py` | ✅ Done | LangGraph CRAG graph: 8 nodes (+ classify_node), Pulse persona, CRAG grader toggle, 3-tier verdict, rolling summary. |
| `ingest.py` | ✅ Done | CLI batch script; auto-loads .env; idempotent upsert |
| `review_chunks.py` | ✅ Done | Dev script: parser → chunker → print chunks. No API/Qdrant required |
| `app.py` | ✅ Done | Streamlit UI: Tab 1 (upload + ingest), Tab 2 (chat + sources + latency + thread titles + persistent metadata) |
| `.streamlit/config.toml` | ✅ Done | Disables file watcher (suppresses torchvision noise from transformers internals) |
| `demo_questions.md` | ✅ Done | Categorised test questions: single-doctor, cross-doctor, topic, fallback, ambiguous |

---

## Known Bugs Fixed

| File | Bug | Fix |
|---|---|---|
| `src/ingestion/pipeline.py` | Called `parse_file` (nonexistent) instead of `parse_document` | Fixed |
| `config.py` | `RELEVANCE_THRESHOLD = 0.25` far too high for RRF scores (max ~0.033) | Fixed: `0.018` |
| `src/retrieval/vector_store.py` | `client.search()` removed in qdrant-client v1.18 | Fixed: `client.query_points()` |
| `requirements.txt` | `streamlit==1.45.1` conflicts with `pdfplumber==0.11.10` over Pillow versioning | Fixed: `streamlit==1.58.0` |
| `src/generation/crag_pipeline.py` | Imported `_build_user_message`, `_format_chunks` from `response_engine.py` — neither existed | Fixed: dead imports removed, constants inlined |

---

## CRAG Pipeline (2026-06-18)

Replaced linear `generate_response()` with a LangGraph CRAG graph (`crag_pipeline.py`).

**Graph:** `retrieve → grade_chunks → load_history → generate → persist → summarize`

**3-tier verdict routing:**
- `CORRECT` (≥1 chunk scores ≥ 0.7) → generate from all chunks
- `AMBIGUOUS` (mixed scores, none ≥ 0.7) → generate from `good_chunks` (score ≥ 0.3) only
- `INCORRECT` (all chunks < 0.3) → fallback message, Sonnet never called

**Smoke test results (2026-06-18):**
| Query | Verdict | Fallback | Citations |
|---|---|---|---|
| "How did Dr. Singh handle pediatric COVID first wave?" | CORRECT | False | 3 |
| "What is the capital of France?" | INCORRECT | True (all_irrelevant) | 0 |
| "What challenges did doctors face during COVID?" | CORRECT | False | 9 |

All three paths confirmed working. Memory persistence confirmed (2 messages per thread in SQLite).

---

## Chunk Quality Review (2026-06-18)

**Finding:** ~40% of chunks were mid-sentence fragments from PDF line-wrap artifacts.
**Fix:** 5-step text normalisation in `parser.py → _normalise()`.

| Doc | Chunks before | Chunks after |
|---|---|---|
| Dr. Singh | 30 | 20 |
| Dr. Johnson | 27 | 12 |

---

## Tech Stack

| Concern | Tool |
|---|---|
| PDF parsing | `pdfplumber` |
| DOCX parsing | `python-docx` |
| Speaker-turn splitting | Custom regex in `chunker.py` |
| Semantic sub-chunking | `chonkie.SemanticChunker` |
| Chunking fallbacks | `chonkie.TokenChunker` (Tier 3) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim) |
| Sparse retrieval | `rank-bm25` (BM25Okapi) |
| Vector DB | `qdrant-client` v1.18 (local on-disk) |
| Metadata extraction | `claude-haiku-4-5-20251001` |
| CRAG chunk grader | `claude-haiku-4-5-20251001` (via `langchain-anthropic`) |
| CRAG sentence filter | Removed (70s+ latency, no quality gain) |
| CRAG rolling summary | `claude-haiku-4-5-20251001` — incremental, fires only when window overflows |
| Answer generation | `claude-sonnet-4-6` |
| Pipeline orchestration | `langgraph` (StateGraph) |
| Prompt templates | `langchain-core` (ChatPromptTemplate) |
| Conversation memory | SQLite (`conversation_store/threads.db`) |
| UI | Streamlit 1.58.0 |

---

## Vector DB State (as of 2026-06-18)

- **276 chunks** across **20 doctor interview documents**
- **India:** Anjali Singh, Arjun Mehta, Kavita Joshi, Neha Reddy, Priya Sharma, Ravi Kumar, Siddharth Rao
- **Germany:** Anna Weber, Clara Hoffmann, Erik Vogel, Felix Schneider, Helena Brandt, Jonas Fischer, Lukas Meier
- **US:** David Thompson, Emily Roberts, James Miller, Michael Anderson, Olivia Davis, Sarah Johnson
- All chunks have full metadata: `doctor_name`, `specialty`, `location`, `hospital`, `topic_tags`, `wave_reference`, `geographic_region`
- Qdrant on-disk: `vector_store/` | BM25 index: `vector_store/bm25_index.pkl`

---

## Environment

- **Venv:** `.venv/` at project root
- **Activate:** `source .venv/bin/activate`
- **Install:** `pip install -r requirements.txt`

---

## Session Log — 2026-06-18 (Evening)

### Changes
- `app.py` — suppressed noisy third-party loggers (`httpx`, `httpcore`, `sentence_transformers`, `huggingface_hub`) to `WARNING`
- `src/retrieval/hybrid_retriever.py` — added RRF score observability: logs top-5 candidate scores, gate PASSED/BLOCKED with actual vs threshold, and final `Returning X/Y chunks to generator (dense_pool=N sparse_pool=M)`

### Grading behaviour clarified
- `CORRECT` verdict (≥1 chunk ≥ `CRAG_UPPER_TH`) → all 5 chunks sent to Sonnet, including the weak one
- `AMBIGUOUS` → only `good_chunks` (score ≥ `CRAG_LOWER_TH`) sent; bad chunks individually dropped
- `INCORRECT` → whole batch rejected, fallback returned

---

## Session Log — 2026-06-18 (Night)

### Memory Layer Upgrade

**`src/memory/conversation.py`**
- Schema: added `title TEXT`, `summary TEXT`, `summary_msg_count INTEGER` to `threads`; added `metadata TEXT` to `messages`; added index on `messages(thread_id)`
- Auto-migration: `ALTER TABLE` on first connect — existing DBs upgrade transparently
- New functions: `set_thread_title`, `get_thread_summary`, `set_thread_summary`, `get_message_count`, `get_messages_slice`
- `add_message` extended with optional `metadata` kwarg (JSON-serialised)
- `get_all_messages` now returns parsed `metadata` dict per message
- Error handling + logging on all public functions; `delete_thread` uses explicit transaction

**`src/generation/crag_pipeline.py`**
- Added `summarize_node`: incremental rolling summary — tracks `summary_msg_count`, calls Haiku only for newly overflowed messages, non-fatal
- `persist_node`: sets thread title on first turn; saves chunks/latencies as assistant message metadata; wrapped in try/except
- `load_history_node`: also loads `summary` from DB; populates `RAGState.summary`
- `generate_node`: injects summary as `## Prior Conversation Summary` before recent turns when non-empty
- Graph wiring: `persist → summarize → END` (was `persist → END`)
- `RAGState`: added `summary: str` field

**`app.py`**
- Sidebar: shows thread title (first question, ≤60 chars) instead of raw UUID
- Tab 2 caption: shows thread title instead of full UUID
- History rendering: tries session cache first, falls back to DB metadata for sources panel + latencies on reload

### Memory architecture
Two-tier design:
- **Tier 1 (short-term):** Last `HISTORY_TURNS=6` messages passed raw to Sonnet every query
- **Tier 2 (rolling summary):** Haiku compresses messages that overflow the window into a ≤200-word summary; injected as `## Prior Conversation Summary` before the raw turns. Incremental — only new overflow is sent to Haiku each turn.

---

## Prompt Instructions for Next Session

Read `PROGRESS.md` first. Run with:
```bash
source .venv/bin/activate
streamlit run app.py
```

The project is complete. Memory layer has two tiers (raw window + rolling summary). Pulse persona added with intent classification. CRAG grader toggle wired through config + UI. All layers done.
