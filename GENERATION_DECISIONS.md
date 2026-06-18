# Design Decisions — Generation Layer

This document covers methodology decisions for the **generation layer**: response engine, citation strategy, conversation memory, and fallback handling.

For ingestion decisions (chunking, metadata extraction), see `DESIGN_DECISIONS.md`.
For retrieval decisions (Qdrant, BM25, hybrid RRF), see `RETRIEVAL_DECISIONS.md`.

---

## Decision 1: Claude Sonnet for Generation

**Model:** `claude-sonnet-4-6`

**Why Sonnet, not Haiku:**
- Haiku handles structured extraction (metadata) well — it's a pattern-matching task
- Sonnet has significantly better reasoning for synthesis across multiple conflicting sources
- The demo asks "Compare Dr. Singh vs Dr. Patel on vaccination" — multi-source synthesis requires reasoning quality, not just retrieval quality
- Cost difference at demo scale (15 docs, ~50 test queries): negligible

**Why not GPT-4 or Gemini:**
- No OpenAI API key available in this environment
- Anthropic access confirmed

---

## Decision 2: Citation Strategy — Graded Chunks as Sources

**Assignment requirement:** Citations are mandatory. Sources must be attributable to specific doctors and chunks.

**Implementation:**
The graded chunks passed to Claude Sonnet are returned directly as `chunks_used` and displayed in the UI as sources. Each chunk in the UI shows:
- Doctor name, specialty, location (from Qdrant metadata)
- Full raw text of the chunk that was sent to the generator

**Why not inline `[CITATION: "..."]` markers:**
An earlier version instructed Sonnet to embed `[CITATION: "exact phrase"]` markers and parsed them with regex. This was removed because:
- It added a `parse_citations_node` — an extra graph node purely for string post-processing
- The citation extraction gave no additional grounding guarantee; the LLM could still confabulate a quote that looked verbatim
- The graded chunks are the actual source material — showing them directly is more honest and simpler

**Doctor attribution:**
Each chunk is formatted with a header before being sent to Sonnet:
```
[Dr. Name | Specialty | Location]
<chunk text>
```
This ensures the generator can attribute claims to the correct doctor without relying on inline markers.

---

## Decision 3: Conversation History Injection

**Storage:** SQLite in `conversation_store/threads.db`. Thread-per-conversation model. Persists across app restarts.

**Why SQLite, not in-memory:**
- Streamlit reruns the entire script on every user interaction — in-memory state is lost
- SQLite requires no additional infrastructure (no Redis, no external DB)
- Thread model enables future features: "go back to a previous conversation"

**History injection into the generation prompt:**
Last `HISTORY_TURNS = 6` messages (3 user + 3 assistant turns) are injected above the source chunks in the user message. This gives the LLM context for follow-up questions ("What about the second wave?" — requires knowing the first question was about pediatric COVID care).

**Why 6 turns, not more:**
- 6 turns = ~1,500 tokens of history at typical message length
- Beyond 6, older turns are rarely relevant and bloat the prompt
- Configurable via `HISTORY_TURNS` in `config.py`

---

## Decision 4: Fallback Handling

Two failure modes with distinct responses:

**Mode 1 — Relevance gate triggered (no chunks returned):**
```
"I could not find relevant information in the interview transcripts to answer
your question. Please try rephrasing, or ask about topics covered in the
COVID-19 doctor interviews..."
```
The model is never called. No risk of hallucination. The fallback message hints at valid query types.

**Mode 2 — Chunks retrieved but LLM generation fails:**
The error is caught, logged, and a brief error message is returned. The user query is still saved to the thread so the conversation history is intact.

**Why not stream the response:**
- Non-streaming is cleaner for the demo — the full response and sources appear together
- Assignment does not require streaming

---

## Decision 5: Response Structure

The CRAG graph returns:
```python
{
    "answer": str,              # clean answer from Sonnet (no markers)
    "chunks_used": list[dict],  # source payloads (doctor, specialty, location, text)
    "fallback": bool,           # True if no relevant chunks found or LLM error
    "fallback_reason": str,     # "no_chunks" | "all_irrelevant" | "llm_error"
    "latency_retrieve": float,  # seconds
    "latency_grade": float,     # seconds
    "latency_generate": float,  # seconds
}
```

`answer_clean` and `citations` were removed along with `parse_citations_node`. The answer is always clean — no post-processing needed. Latency fields are surfaced in the UI as a small caption line below each response.

---

## Decision 6: Source Attribution in the UI

Each retrieved chunk carries its full payload (`doctor_name`, `specialty`, `location`, `source_file`, `text`). The UI displays a collapsible "Sources used" section under each assistant response showing which doctors' transcripts were referenced. This directly supports the assignment requirement for provenance tracking.

---

## Decision 7: LangGraph as Pipeline Orchestrator ✅ Done

`response_engine.py` (linear function) has been replaced by `crag_pipeline.py` — a **LangGraph stateful graph** implementing Corrective RAG (Yan et al. 2024). See `C-RAG.md` for the full design doc.

**Why LangGraph (not raw function calls):**
- Explicit state object (`RAGState`) makes the data flow inspectable and testable at each node
- Conditional edges replace nested if/else — the relevance gate becomes a router edge, not an inline branch
- Each node (retrieve, load_history, generate, persist) is independently testable in isolation
- Graph structure is a better diagram of the architecture for the assignment evaluation

**Planned graph structure:**
```
START → retrieve → gate_check ──(no results)──→ fallback → persist → END
                        │
                   (results ok)
                        ↓
                 load_history → generate → parse_citations → persist → END
```

**What changes vs current design:**
- `response_engine.generate_response()` becomes a compiled LangGraph graph invoked as `graph.invoke({"query": ..., "thread_id": ...})`
- `BaseLLMEngine` / `build_llm_engine()` from `llm_engine.py` remain unchanged — still passed in at construction time
- Return structure is identical — the graph output matches the current dict schema
- No change to `src/retrieval/hybrid_retriever.py` or `src/memory/conversation.py` — LangGraph wraps them, doesn't replace them

**LangGraph is NOT LangChain** — it is a graph/workflow orchestration library. LangChain utilities (`ChatPromptTemplate`, `langchain-anthropic` structured output) are used inside individual nodes for the grader and sentence filter chains, but there are no chain abstractions or LangChain models used for the main generation path.

---

## Decision 8: CRAG — Corrective RAG (Implemented)

The planned LangGraph refactor was implemented as a full **Corrective RAG (CRAG)** pipeline.

**Reference:** Yan et al. 2024 — *Corrective Retrieval Augmented Generation* (https://arxiv.org/abs/2401.15884)

**What was added:**
- `grade_chunks_node` — Claude Haiku scores each retrieved chunk `[0–1]` against the query. Derives a 3-tier verdict: `CORRECT` / `AMBIGUOUS` / `INCORRECT`. Replaces the single RRF threshold gate entirely.
- `fallback_node` — Three distinct fallback reasons (`no_chunks`, `all_irrelevant`, `llm_error`) with distinct user-facing messages.

**What was tried and removed:**
- `refine_context_node` — Sentence-level Haiku filtering was implemented but removed. It added 70+ seconds latency per query (~20–30 serial API calls) and provided no quality gain over chunk grading alone. It also stripped doctor attribution from chunk text, causing generation failures.
- `parse_citations_node` — Inline `[CITATION: "..."]` marker extraction was implemented but removed. Graded chunks shown directly as sources is simpler and equally attributable.

**Closed-domain adaptation:** Standard CRAG routes `INCORRECT` to web search. This system is corpus-only — `INCORRECT` returns a fallback message instead.

**Full design:** see `C-RAG.md`.

---

## Implementation Status

| File | Status | Notes |
|---|---|---|
| `src/generation/llm_engine.py` | ✅ Done | `BaseLLMEngine` ABC + Anthropic / Gemini / Ollama engines + factory |
| `src/generation/response_engine.py` | ✅ Deleted | Replaced by `crag_pipeline.py` |
| `src/generation/crag_pipeline.py` | ✅ Done | LangGraph CRAG graph: 6 nodes, 3-tier verdict, doctor attribution, `build_crag_graph()` factory |

---

## Open Questions (Resolved + Active)

1. ~~**LangGraph refactor (Decision 7)**~~ — Done as CRAG (Decision 8). `response_engine.generate_response()` is superseded by `crag_pipeline.build_crag_graph()`.
2. **Max response tokens:** `MAX_RESPONSE_TOKENS = 1024`. May need to increase for complex multi-doctor comparison answers — monitor in UI testing.
3. **Grade parallelism:** `grade_chunks_node` makes 5 serial Haiku calls (~13s). These could be parallelised with `asyncio.gather` or `concurrent.futures` to cut grade latency to ~3s. Not done yet — serial is simpler and correctness is confirmed first.
