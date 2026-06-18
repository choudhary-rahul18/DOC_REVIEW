# CLAUDE.md — RAG Interview Pipeline

This is a standalone project inside the Athena repo. It is a job assignment submission, not part of Athena's product. Read this before making any changes.

---

## What This Project Is

An end-to-end **Conversational RAG system** over COVID-19 pandemic interview transcripts with healthcare professionals. The corpus is 10–15 PDF/DOCX files, each structured as a Q&A exchange between an interviewer and a doctor.

**Assignment source:** `docs/RAG Pipeline Design.docx`
**Sample data:** `docs/Dr_Anjali_Singh_India_Detailed_Interview.pdf`

This is a job interview take-home assignment. Code quality, design decisions, and clarity of architecture all matter as much as correctness.

---

## Architecture: One Pipeline, Two Entry Points

The ingestion pipeline is shared code (`src/ingestion/pipeline.py`). It is called from two places:

1. **`ingest.py`** (CLI batch script) — run once offline to process all 14-15 docs in `data/raw/` and build the Qdrant vector DB. This is the primary way to bootstrap the knowledge base.
2. **Streamlit Tab 1** (UI ingestion) — user uploads new PDF/DOCX → same pipeline → upserts into the existing Qdrant DB. The DB is additive; re-ingesting the same file is idempotent (upsert by chunk_id).

The Streamlit app (Tab 2: Chat) then queries the Qdrant DB that was built by either path.

**Deployment note:** For local demo, Qdrant runs on-disk. For Render/cloud deployment, swap `QDRANT_PATH` for `QDRANT_URL` + `QDRANT_API_KEY` env vars — `config.py` supports both modes.

---

## Tech Stack (Evolving — upgrade when a better tool exists)

| Layer | Choice | Reason |
|---|---|---|
| PDF parsing | pdfplumber | Clean text, handles multi-column |
| DOCX parsing | python-docx | Standard |
| Chunking | chonkie (SemanticChunker + TokenChunker) | Embedding-based topic-boundary detection; no LLM needed for chunking |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, no API key needed |
| Sparse retrieval | rank-bm25 | Simple, no infra |
| Vector DB | Qdrant (local on-disk mode) | Native list/array payload support; cleaner metadata filters than ChromaDB |
| LLM (generation) | claude-sonnet-4-6 | Best quality, Anthropic access |
| LLM (grading) | claude-haiku-4-5-20251001 | Per-chunk relevance grading (singleton, not recreated per call) |
| LLM (metadata) | claude-haiku-4-5-20251001 | Cheaper per-chunk extraction call at ingest time |
| Pipeline orchestration | LangGraph | Graph-based workflow for retrieval + memory + generation; explicit state, conditional edges |
| Memory | SQLite-backed threads + rolling Haiku summary | Two-tier: raw window (6 turns) + incremental summary for overflow |
| UI | Streamlit | Clean, self-explanatory |

**No LangChain.** The retrieval/generation pipeline is wired explicitly. **LangGraph is allowed** — it is a graph orchestration library (not a chain abstraction) and is used to model the retrieval → memory → generation flow as a stateful graph with conditional edges.

---

## Corpus Format — Critical for Chunking

The interview PDFs are NOT cleanly formatted. Speaker turns appear **inline** in the same paragraph:

```
"Interviewer: What were your first experiences... Dr. Singh: The first wave in 2020..."
```

The Q&A chunker must use regex to split on `"Interviewer:"` and `"Dr. [Name]:"` patterns across continuous text, not newlines. This is the most fragile part of the pipeline — handle it carefully.

Each Q&A pair = one semantic chunk. This is far better than sliding-window for this corpus.

---

## Metadata Per Chunk

Each chunk stored in Qdrant gets:
- `doctor_name` — extracted via LLM at ingest time
- `doctor_specialty` — e.g., "Pediatrician, Neonatologist"
- `location` — e.g., "Lucknow, India"
- `hospital` — e.g., "City Children's Hospital"
- `topic_tags` — list of topics in this Q&A exchange
- `wave_reference` — first wave / Delta / Omicron (if mentioned)
- `geographic_region` — city/state
- `chunk_index` — position in document
- `source_file` — filename

These enable cross-doctor / cross-geo queries ("How did pediatricians in different cities handle vaccination?") — which the assignment explicitly calls out.

---

## Retrieval Design

**Hybrid retrieval:** Dense (sentence-transformers cosine) + BM25 sparse → RRF fusion (k=60).
**Relevance gate:** If best fused score < threshold → return fallback message, not a hallucinated answer.
**Top-K:** 5 chunks returned to the generator.

---

## Response Engine Design (CRAG)

```
classify → retrieve → grade_chunks → load_history → generate → persist → summarize
```

1. **Claude Haiku classifies intent** — greetings/small talk → direct reply (skips pipeline); factual queries → retrieval
2. Hybrid retrieve top-5 chunks (BM25 + dense → RRF)
3. Claude Haiku grades each chunk 0–1; derives CORRECT / AMBIGUOUS / INCORRECT verdict
   - INCORRECT → fallback message, no LLM generation
   - AMBIGUOUS → use only good_chunks (score ≥ 0.3) for generation
   - Grader can be bypassed via `CRAG_GRADER_ENABLED=False` in `config.py` or sidebar toggle
4. Load last N conversation turns + rolling summary from SQLite
5. Claude Sonnet generates a clean answer as **Pulse** (summary + recent turns + source chunks)
6. Persist messages (with metadata); set thread title on first turn
7. Haiku updates rolling summary if new messages have overflowed the window

**Pulse persona:** The assistant is named Pulse — a medical interview insights assistant. It never references "sources", "chunks", or retrieval internals. Attributes answers naturally to doctors by name.

**Source chunks are the citations.** Each chunk displayed in the UI shows the doctor heading (`[Dr. Name | Specialty | Location]`) and the full raw text passed to the generator. No inline citation markers, no regex parsing.

---

## Conversation Memory

Two-tier design:

- **Tier 1 — Raw window:** Last `HISTORY_TURNS=6` messages passed verbatim to Sonnet on every query as `## Recent Conversation`
- **Tier 2 — Rolling summary:** Haiku compresses messages that fall off the window into a ≤200-word summary, stored in `threads.summary`. Incremental — only newly overflowed messages are sent to Haiku each turn (tracked via `summary_msg_count`). Injected into the prompt as `## Prior Conversation Summary` when non-empty.

SQLite schema: `threads(thread_id, created_at, title, summary, summary_msg_count)` + `messages(id, thread_id, role, content, timestamp, metadata)`. The `metadata` JSON on assistant messages stores `chunks_used` and latencies so the sources panel survives browser refresh.

---

## Folder Structure (✅ = built, 🔲 = pending)

```
rag_interview_pipeline/
├── CLAUDE.md
├── app.py                          ← Streamlit entry point (2 tabs: Ingestion + Chat)
├── ingest.py                       ← CLI batch script: process all docs in data/raw/
├── config.py                       ← Centralized paths, model names, thresholds
├── src/
│   ├── ingestion/
│   │   ├── parser.py               ← PDF + DOCX → raw text
│   │   ├── chunker.py              ← 3-tier cascade: speaker-regex + chonkie semantic + token
│   │   ├── metadata_extractor.py   ← Claude Haiku → structured metadata + topic_tags per doc
│   │   └── pipeline.py             ← Shared orchestrator (called by both ingest.py and app.py)
│   ├── retrieval/
│   │   ├── embedder.py             ← sentence-transformers embed + BM25 rebuild
│   │   ├── vector_store.py         ← Qdrant CRUD (local on-disk or cloud via env vars)
│   │   └── hybrid_retriever.py    ← RRF fusion + relevance gate
│   ├── generation/
│   │   ├── llm_engine.py          ← Provider-agnostic LLM abstraction (Anthropic, Gemini, Ollama)
│   │   └── crag_pipeline.py       ← LangGraph CRAG graph: classify → retrieve → grade → history → generate → persist → summarize; Pulse persona
│   └── memory/
│       └── conversation.py        ← SQLite thread CRUD
├── data/
│   └── raw/                        ← Drop all 14-15 PDFs/DOCXs here for batch ingest
├── docs/                           ← Assignment + sample data (reference only)
│   ├── RAG Pipeline Design.docx
│   └── Dr_Anjali_Singh_India_Detailed_Interview.pdf
├── vector_store/                   ← Qdrant persists here (auto-created, gitignored)
├── conversation_store/             ← SQLite thread storage (auto-created, gitignored)
├── notebook/                       ← Walkthrough notebook (design decisions)
├── test_pipeline.py                ← End-to-end smoke test: 10 queries, all path types, 10/10 pass
├── requirements.txt
└── .env.example
```

---

## What NOT to Do

- Do not use LlamaIndex — it abstracts away the pipeline, hiding the design decisions the assignment evaluates
- LangChain utilities (ChatPromptTemplate, structured output, etc.) are allowed where they reduce boilerplate; LangGraph is the primary orchestration layer
- Do not use OpenAI (no API key available — use sentence-transformers for embeddings)
- Do not add streaming to the response engine yet — non-streaming is cleaner for the demo
- Do not simplify the metadata extraction — structured metadata enables the cross-doctor queries the assignment asks for
- Do not add back `refine_context_node` (sentence-level Haiku filtering) — it was removed because it added 70s+ latency per query with no meaningful quality gain over chunk grading alone
- Do not add back inline `[CITATION: "..."]` markers — citation extraction via LLM instruction + regex was removed; the graded chunks themselves are displayed as sources
- Do not use LLM for chunking — use chonkie's SemanticChunker (embedding-based, local, no API cost)
- Do not use sliding-window as primary chunking — it is only a last-resort Tier 3 fallback

---

## Build Status

| Layer | Files | Status |
|---|---|---|
| Ingestion | `parser.py`, `chunker.py`, `metadata_extractor.py`, `pipeline.py` | ✅ Done |
| Vector DB | `vector_store.py` | ✅ Done |
| Retrieval | `embedder.py`, `hybrid_retriever.py` | ✅ Done |
| Generation | `llm_engine.py`, `crag_pipeline.py` (self-contained; `response_engine.py` deleted) | ✅ Done — no citation markers, no parse_citations_node; chunks are sources |
| Memory | `conversation.py` | ✅ Done — two-tier: raw window + rolling summary; thread titles; persistent metadata |
| CLI | `ingest.py` | ✅ Done |
| UI | `app.py`, `.streamlit/config.toml` | ✅ Done |
| Dependencies | `requirements.txt` | ✅ Done |
| Smoke tests | `test_pipeline.py` | ✅ Done — 10/10 pass (grader disabled) |

---

## How to Run

```bash
cd rag_interview_pipeline
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
streamlit run app.py
```

## How to Test

```bash
source .venv/bin/activate
python test_pipeline.py
```

Covers: direct intent, single-doctor, cross-doctor, topic-based, and out-of-scope queries. All 10 cases should pass. Run this before pushing any changes to the pipeline.
