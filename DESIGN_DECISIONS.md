# Design Decisions — Data Ingestion Layer

This document covers methodology decisions for the **ingestion pipeline only**: parsing, chunking, metadata extraction, and the shared pipeline orchestrator. It is the canonical reference for these decisions and supersedes `PROJECT_SCOPE.md` wherever the two conflict.

For retrieval decisions (Qdrant, BM25, hybrid RRF), see `RETRIEVAL_DECISIONS.md`.
For generation decisions (Claude Sonnet, citations, conversation memory), see `GENERATION_DECISIONS.md`.

---

## Decision 1: Chunking Strategy — 3-Tier Cascade

### The problem with one Q&A = one chunk

The sample data (`Dr_Anjali_Singh_India_Detailed_Interview.pdf`) shows doctor answers frequently spanning multiple **unrelated sub-topics** in a single response. A single answer from Dr. Singh discusses immunization delays AND pediatric mental health — two completely different retrieval targets. Storing the whole Q&A as one chunk pollutes the LLM context and degrades retrieval precision.

### What we do: 3-tier cascade

```
Tier 1 — Speaker-turn regex + chonkie SemanticChunker   (best)
Tier 2 — chonkie SemanticChunker on whole doc           (fallback: no Q&A structure detected)
Tier 3 — chonkie TokenChunker                           (last resort: always succeeds)
```

Every chunk that falls to Tier 2 or Tier 3 is logged as a WARNING during ingest. The `parse_tier` field is stored in Qdrant for post-ingest auditing.

### Tier 1 in detail: two-pass regex + semantic sub-chunking

**Step 1 — Speaker discovery (pass 1):**
Scan the first 2,000 characters of the document. Extract candidate speaker labels (title-cased words before a colon). Keep only those that start with a known honorific (`Dr.`, `Interviewer`, `Prof.`) OR appear 2+ times in the full document. The frequency filter kills false positives like `"PPE:"` or `"COVID-19:"`.

**Step 2 — Regex split (pass 2):**
Build a precise regex from the discovered labels, sorted by length descending (prevents partial match). Split full document text into `(speaker_label, turn_text)` pairs. Pair consecutive interviewer→doctor turns into `(question, answer)` pairs.

**Step 3 — Semantic sub-chunking of doctor answers:**
Each doctor answer is passed to `chonkie.SemanticChunker`. It uses sentence embeddings + Savitzky-Golay smoothing to find where topics shift inside the answer — correctly splits "immunization delays → pediatric mental health → MIS-C" even when written as one flowing paragraph.

**Step 4 — Question header prepended:**
Every sub-chunk gets the interviewer's question prepended:
```
[Question] What challenges did you face with immunization programs?

[Answer excerpt] The first wave severely disrupted our vaccination schedules...
```
This anchors the embedding — "disrupted vaccination schedules" means something different depending on whether the question was about immunization or staffing.

**Why regex for speaker splitting, not LLM:**
The inline format (`"Interviewer: ... Dr. Singh: ..."`) is a known grammar with a finite set of patterns. A regex parser for a known format is correct and production-appropriate. LLMs add API cost and latency with zero quality gain for this structural parsing task.

**Chunk schema:**
```json
{
  "chunk_id": "sha256(source_file + chunk_index)[:16]",
  "parent_qa_id": "sha256(source_file + qa_index)[:16]",
  "chunk_index": 7,
  "qa_index": 2,
  "text": "[Question] ...\n\n[Answer excerpt] ...",
  "question_header": "What challenges...",
  "source_file": "Dr_Anjali_Singh.pdf",
  "parse_tier": 1
}
```

---

## Decision 2: Why Chonkie for Sub-Chunking (Not Handwritten Logic)

**Original approach:** Split doctor answers at paragraph (`\n\n`) boundaries; if paragraph > 300 tokens, split further at sentence boundaries with greedy merge.

**Problem:** This relies on physical formatting (paragraph breaks, punctuation) rather than semantic content. A doctor who writes in one long paragraph with no breaks would produce a 1,000-token chunk.

**Chonkie's approach (`SemanticChunker`):**
1. Split text into sentences
2. Embed each sentence with `all-MiniLM-L6-v2` (already loaded for retrieval — zero extra model weight)
3. Compute cosine similarity between consecutive sentence embeddings
4. Apply Savitzky-Golay smoothing to the similarity curve (reduces noise from individual sentence variations)
5. Find valleys in the smoothed curve → semantic topic boundaries
6. Merge sentences between boundaries into chunks, respecting the token ceiling

**Result:** Chunks that align with topic shifts, not formatting accidents. The same `all-MiniLM-L6-v2` model is reused — chunking and retrieval use the same embedding space.

---

## Decision 3: Metadata and Topic Tag Extraction Strategy

### What we extract per document

**A) Document-level constants** (same across all chunks from one file):
- `doctor_name`, `specialty`, `hospital`, `location`, `geographic_region`

**B) Chunk-level variables** (different per chunk):
- `topic_tags` — list of 3–5 semantic concept phrases per chunk
- `wave_reference` — extracted by pure regex (closed vocabulary: "first wave", "delta", "omicron", etc.)

### One LLM call per document (Claude Haiku)

Sends: doc header (first 500 chars of first chunk) + all chunk texts (indexed by chunk_index) + exact JSON schema.

Returns:
```json
{
  "metadata": {
    "doctor_name": "Dr. Anjali Singh",
    "specialty": "Pediatrician, Neonatologist",
    "hospital": "City Children's Hospital",
    "location": "Lucknow, India",
    "geographic_region": "Uttar Pradesh, India"
  },
  "chunk_tags": {
    "0": ["immunization delay", "COVID first wave", "child vaccination"],
    "1": ["pediatric mental health", "lockdown anxiety", "school closure"],
    "2": ["MIS-C", "multisystem inflammatory syndrome", "post-COVID children"]
  }
}
```

### Why LLM tags, not KeyBERT

KeyBERT extracts the most salient *phrases from the text itself* — if the doctor says "overwhelmed and exhausted," the tag is "overwhelmed," not "staff burnout." An LLM understands that the concept is "staff burnout" regardless of phrasing. At demo scale (15 docs, ~$0.005 total), there is no cost reason to sacrifice tag quality.

### Malformed JSON recovery (4-layer)

1. `json.loads(raw.strip())`
2. Regex extract largest `{...}` block (handles LLM preamble/postamble)
3. Targeted regex to extract `metadata` and `chunk_tags` sub-objects independently
4. Total failure → empty metadata, log ERROR, pipeline continues (never halts on one bad doc)

### Why one call per document (not one per chunk)

Every LLM call carries fixed overhead tokens — the system prompt (instructions, JSON schema). That overhead is billed per call, not per document. Our combined per-doc call is ~2.5× cheaper than per-chunk LLM calls at any scale.

### Production path

| Task | Demo (15 docs) | Production (1M+ docs) |
|---|---|---|
| Doc-level metadata | Claude Haiku (1 call, header ~500 tokens) | Local SLM on GPU (Phi-3, Llama-3.2-3B) |
| Per-chunk topic tags | Claude Haiku (same call, all chunks) | KeyBERT (reuses MiniLM embedder, zero API cost) |

---

## Decision 4: Two Entry Points, One Pipeline

```
ingest.py (CLI batch) ─────────┐
                               ▼
                     src/ingestion/pipeline.py   ← shared orchestrator
                               ▲
app.py Tab 1 (UI upload) ──────┘
```

`ingest.py` processes all docs in `data/raw/` at once to bootstrap the Qdrant DB. The Streamlit UI ingestion tab uses the identical `ingest_document()` function to add new docs to the existing DB. The DB is additive — re-ingesting the same file is idempotent (Qdrant upsert by UUID5 point ID).

---

## Decision 5: Parser Text Normalisation Before Chunking

### Problem discovered during chunk review (2026-06-18)

Running the 3-tier chunker against both sample PDFs (`Dr_Anjali_Singh` and `Dr_Sarah_Johnson`) produced **~40% broken fragment chunks** — sentence tails like `"suits to make them less intimidating"` (15 words) and `"thresholds for escalation"` (3 words). These are not retrievable by any meaningful query.

Root cause: `pdfplumber` preserves the PDF's physical line wraps. A sentence like `"PPE suits"` arrives as `"PPE\nsuits"`. The SemanticChunker's sentence splitter treats `\n` as a sentence boundary, producing two embedding vectors with low similarity → forced split → orphaned tail.

### What we evaluated: Docling

Docling (IBM) is a layout-aware PDF parser that detects visual structure (headings, sections, tables). A multimodal AI reviewed the actual PDF pages and confirmed:

- No visual Q&A separation (questions not bold/headed/sectioned)
- No detectable heading hierarchy
- Speaker turns appear **inline** in flowing prose: `"Interviewer: ... Dr. Singh: ..."`
- Line-wrap extraction artifacts present

**Verdict: Docling would not improve chunk boundaries for this corpus.** The structure is linguistic (speaker label patterns), not visual. Our regex approach is correct for this format.

### Fix: 5-step normalisation in `parser.py → _normalise()`

Applied before any chunking or speaker detection:

1. **Unicode cleanup** — curly quotes/apostrophes → ASCII; em/en dashes → hyphen; non-breaking space → space
2. **Soft hyphen reconstruction** — `"immuniza-\ntion"` → `"immunization"`
3. **Soft line break collapse** — `\n` NOT preceded by `.!?…` and NOT followed by `\n` → single space (PDF word-wrap artifact, not a paragraph boundary)
4. **Whitespace normalisation** — 2+ consecutive non-newline whitespace → single space
5. **Paragraph normalisation** — 3+ newlines → double newline

### Result (after normalisation, before minimum-size merge)

| Doc | Chunks before | Chunks after normalisation | Avg length before | Avg length after |
|---|---|---|---|---|
| Dr. Singh | 30 | 20 | ~240 chars | 445 chars |
| Dr. Johnson | 27 | 12 | ~280 chars | 674 chars |

27 fragment chunks eliminated by normalisation. However, 6 micro-fragment chunks remained in Dr. Singh's data — narrative tail sentences (e.g. "Small things like that mattered more than any medicine." — 55 chars) split off by the SemanticChunker at closing-sentence topic shifts. See Decision 6.

---

## Decision 6: Minimum Chunk Size — 100-Token Merge Pass

### Problem discovered during chunk review (2026-06-18, after normalisation fix)

Even after text normalisation, 6 chunks in Dr. Singh's data were micro-fragments: 1–2 closing sentences (55–157 chars) that the SemanticChunker correctly identified as a semantic shift but which are too short to carry any retrieval signal. A query like "how did Dr. Singh manage PPE?" will never match "Small things like that mattered more than any medicine." — the chunk is orphaned context.

These are a different failure mode from the normalisation artifacts: they arise from narrative style (doctors closing an answer with a one-sentence reflection), not from PDF line-wrap bugs.

### Fix: post-chunking merge pass in `chunker.py → _sub_chunk_answer()`

After `SemanticChunker` produces sub-chunks for a doctor answer, run `_merge_short_chunks(texts, MIN_CHUNK_TOKENS)`:

- Walk left to right through the sub-chunk list
- If either the current chunk **or** its predecessor is below `MIN_CHUNK_TOKENS`, concatenate them
- Never merges across QA boundaries (`parent_qa_id` is preserved)

`MIN_CHUNK_TOKENS = 100` is configured in `config.py`. Token count is approximated as `int(words × 1.3)` — sufficient for a minimum-size guard with no additional dependencies.

### Result (final, after both normalisation and merge pass)

| Doc | Raw chunks | After normalisation | After merge | Avg length (final) |
|---|---|---|---|---|
| Dr. Singh | 30 | 20 | **12** | **671 chars** |
| Dr. Johnson | 27 | 12 | **12** | **674 chars** |

Both docs now produce a perfect 1:1 chunk-to-QA-pair ratio. All chunks are complete, coherent semantic units. No fragments remain.

### Why 100 tokens (not lower)

100 tokens (~75 words) is the practical minimum for a chunk to be retrievable. A 2–3 sentence closing remark is typically 30–60 tokens — well below the threshold and correctly merged. A genuine short-answer QA pair (e.g. a doctor giving a concise yes/no with brief explanation) runs 80–120 tokens — close to the boundary. Validated against both sample docs: no legitimate short answers were incorrectly merged.

---

## Retrieval Reference: RRF Relevance Gate Thresholds

RRF scores are bounded by `k=60`, two retrievers, `DENSE_CANDIDATES=20`. Full decision rationale in `RETRIEVAL_DECISIONS.md` — Decision 5.

| Threshold | Effect |
|---|---|
| 0.013 | Passes almost everything |
| **0.018** | Passes documents that rank well in at least one retriever — **current setting** |
| 0.025 | Requires presence in both retrievers at reasonable rank |
| 0.033 | Only the top result in both passes (ceiling) |

---

## Open Questions

*(None — all open questions from prior sessions resolved as of 2026-06-18.)*

- **Semantic similarity threshold** (`SEMANTIC_SIMILARITY_THRESHOLD = 0.5`): validated at 0.5 — produces clean splits on both sample docs after the merge pass absorbs any resulting micro-fragments. No change needed.
- **Chunk count per doc**: confirmed at **12 chunks per document** on both sample docs. Use this for Haiku prompt sizing estimates on the full corpus.
