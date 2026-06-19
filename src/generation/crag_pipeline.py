"""
Corrective RAG (CRAG) pipeline — LangGraph implementation.

Graph:
  classify ──(direct)──────────────────────────────► persist → summarize → END
      │
   (retrieval)
      ↓
  retrieve → grade_chunks ──(INCORRECT)──► fallback ──► persist → summarize → END
                 │
            (CORRECT/AMBIGUOUS)
                 ↓
         load_history → generate → persist → summarize → END

Adaptation from Yan et al. 2024 (https://arxiv.org/abs/2401.15884):
- No web search (closed-domain corpus). INCORRECT verdict → fallback message.
- AMBIGUOUS verdict → generate from good_chunks only (score > LOWER_TH).
- chunks_used are returned as-is and displayed as sources in the UI.
"""

import logging
import time
from typing import Any, Literal, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from config import (
    CRAG_LOWER_TH,
    CRAG_UPPER_TH,
    HISTORY_TURNS,
    MAX_RESPONSE_TOKENS,
    METADATA_MODEL,
)
from src.memory.conversation import (
    add_message,
    get_history,
    get_message_count,
    get_messages_slice,
    get_thread_summary,
    set_thread_summary,
    set_thread_title,
)
from src.retrieval.hybrid_retriever import retrieve

_SYSTEM_PROMPT = """\
You are Pulse, a medical interview insights assistant. You have deep familiarity with \
a curated collection of in-depth conversations with healthcare professionals — doctors, \
specialists, and practitioners who shared their experiences, challenges, and observations \
in personal interviews.

Your job is to help users understand these professionals — their backgrounds, perspectives, \
clinical experiences, and the stories they told.

When answering:
- Speak naturally and confidently, as if you personally know these professionals
- Attribute information to the specific person by name (e.g., "Dr. Singh mentioned..." \
or "Dr. Weber spoke about...") — always make it feel personal and direct
- Never use phrases like "based on the provided sources", "according to the chunks", \
"the retrieved text says", "based on my knowledge base", or any language that reveals \
you are reading from documents — it breaks the experience
- If you genuinely don't have enough information to answer something, say so honestly \
and naturally, without technical jargon
- Do not speculate or invent details beyond what these professionals actually shared\
"""


_FALLBACK_NO_RESULTS = (
    "I don't have anything on that in the interviews I know about. "
    "Try asking about a specific doctor, their experiences, clinical work, "
    "or topics the professionals spoke about — that's where I can really help."
)

_FALLBACK_LLM_ERROR = (
    "Something went wrong on my end. Give it another try."
)

_FALLBACK_ALL_IRRELEVANT = (
    "I couldn't find a strong enough match for that in the interviews. "
    "Try rephrasing, or ask about a specific doctor or topic the professionals spoke about."
)

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    # Caller inputs
    query: str
    thread_id: str
    qdrant_filter: Optional[Any]

    # Retrieval
    chunks: list[dict]

    # Grading
    chunk_verdicts: list[dict]       # [{chunk_id, verdict, score, reason}]
    overall_verdict: str             # "CORRECT" | "AMBIGUOUS" | "INCORRECT"
    good_chunks: list[dict]          # chunks with grade score > CRAG_LOWER_TH

    # Generation
    history: list[dict]
    summary: str           # rolling summary of turns that fell off the history window
    answer: str
    chunks_used: list[dict]

    # Result
    fallback: bool
    fallback_reason: str             # "no_chunks" | "all_irrelevant" | "llm_error"

    # Intent classification
    intent: str                          # "direct" | "retrieval"

    # Latency (seconds per stage)
    latency_retrieve: float
    latency_grade: float
    latency_generate: float


# ── Pydantic schemas for structured LLM output ───────────────────────────────

class ChunkGrade(BaseModel):
    score: float
    verdict: Literal["CORRECT", "AMBIGUOUS", "INCORRECT"]
    reason: str


class IntentResponse(BaseModel):
    classification: Literal["direct", "retrieval"]
    text: str   # if "direct": the response to send; if "retrieval": the user's query


# ── Prompt templates ─────────────────────────────────────────────────────────

_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are Pulse, a medical interview insights assistant. "
        "You have deep familiarity with a curated collection of in-depth conversations "
        "with healthcare professionals — doctors, specialists, and practitioners.\n\n"
        "Recent conversation history (use this for context when classifying and responding):\n"
        "{history}\n\n"
        "For the new user message, decide:\n\n"
        "1. classification = 'direct'  — ONLY for pure greetings, farewells, or "
        "one-word acknowledgements with no question behind them (e.g. 'hello', 'hi', "
        "'thanks', 'bye', 'ok'). "
        "In this case, write a short, warm, natural response in 'text'. "
        "Speak as Pulse. You may reference the conversation history above if relevant. "
        "Never mention documents, sources, or retrieval systems.\n\n"
        "2. classification = 'retrieval'  — for everything else, including:\n"
        "   - questions about a specific doctor, their experiences, or clinical topics\n"
        "   - questions about the current conversation (e.g. 'what did I ask earlier?', "
        "'what was my first question?', 'can you summarize what we covered?')\n"
        "   - follow-up questions, clarifications, or any substantive request\n"
        "   - questions about who you are or what you can do\n"
        "   In this case, copy the user's original message exactly into 'text'.\n\n"
        "When in doubt, choose 'retrieval'.\n\n"
        "Output JSON only: {{\"classification\": \"direct\" | \"retrieval\", \"text\": \"...\"}}"
        " — no prose, no markdown fences.",
    ),
    ("human", "{query}"),
])

_SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You maintain a running summary of a conversation handled by Pulse, a medical interview insights assistant.\n"
        "Given the existing summary (may be empty) and new conversation turns that have scrolled out of the "
        "active context window, produce an updated summary.\n\n"
        "Rules:\n"
        "- Keep the summary under 200 words\n"
        "- Preserve: doctors mentioned, topics covered, key findings, any follow-up questions the user asked\n"
        "- Write in past tense, third-person neutral (e.g. 'The user asked about... Pulse explained...')\n"
        "- Output only the updated summary text — no preamble or labels",
    ),
    (
        "human",
        "Existing summary:\n{existing_summary}\n\nNew turns to incorporate:\n{new_turns}",
    ),
])

_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are evaluating a retrieved Q&A chunk from a COVID-19 doctor interview transcript.\n"
        "Score how well this chunk contributes to answering the user's question.\n\n"
        "Score guide:\n"
        "  0.8–1.0 → CORRECT    chunk directly and fully addresses the question\n"
        "  0.3–0.8 → AMBIGUOUS  partial, tangential, or one perspective of a multi-part question\n"
        "  0.0–0.3 → INCORRECT  chunk is irrelevant or completely off-topic\n\n"
        "Important: for comparative or cross-doctor questions (e.g. 'How did doctors in India vs Germany differ...'), "
        "a chunk covering just one side of the comparison is still AMBIGUOUS (0.3–0.8), not INCORRECT. "
        "Only score INCORRECT if the chunk has no bearing on the question topic at all.\n\n"
        "Output JSON only with keys: score (float), verdict (str), reason (str).",
    ),
    ("human", "Question: {question}\n\nChunk:\n{chunk}"),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

_haiku: ChatAnthropic | None = None

def _get_haiku() -> ChatAnthropic:
    global _haiku
    if _haiku is None:
        _haiku = ChatAnthropic(model=METADATA_MODEL, temperature=0)
        logger.info("[crag] Haiku client initialised — model=%s", METADATA_MODEL)
    return _haiku


def _format_chunks(chunks: list[dict]) -> str:
    """Format chunks with doctor attribution for the generation prompt."""
    blocks: list[str] = []
    for chunk in chunks:
        p = chunk["payload"]
        header_parts = [p.get("doctor_name") or "Unknown Doctor"]
        if p.get("specialty"):
            header_parts.append(p["specialty"])
        if p.get("location"):
            header_parts.append(p["location"])
        header = " | ".join(header_parts)
        blocks.append(f"[{header}]\n{p.get('text', '').strip()}")
    return "\n\n---\n\n".join(blocks)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def classify_node(state: RAGState) -> dict:
    try:
        recent = get_history(state["thread_id"], HISTORY_TURNS)
        history_ctx = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Pulse'}: {m['content'][:300]}"
            for m in recent
        ) if recent else "(no prior messages in this conversation)"
    except Exception:
        history_ctx = "(no prior messages in this conversation)"

    chain = _CLASSIFY_PROMPT | _get_haiku().with_structured_output(IntentResponse)
    try:
        result: IntentResponse = chain.invoke({"query": state["query"], "history": history_ctx})
    except Exception as exc:
        logger.warning("[crag] classify failed: %s — defaulting to retrieval", exc)
        return {"intent": "retrieval"}

    logger.info("[crag] classify: %s", result.classification)
    if result.classification == "direct":
        return {"intent": "direct", "answer": result.text, "chunks_used": [], "fallback": False}
    return {"intent": "retrieval"}


def retrieve_node(state: RAGState, *, qdrant_client) -> dict:
    t0 = time.perf_counter()
    chunks = retrieve(state["query"], qdrant_client, qdrant_filter=state.get("qdrant_filter"))
    elapsed = time.perf_counter() - t0
    if not chunks:
        logger.info("[crag] retrieve: no chunks — skipping to fallback  (%.2fs)", elapsed)
        return {
            "chunks": [],
            "overall_verdict": "INCORRECT",
            "fallback_reason": "no_chunks",
            "latency_retrieve": elapsed,
        }
    logger.info("[crag] retrieve: %d chunks  (%.2fs)", len(chunks), elapsed)
    return {"chunks": chunks, "latency_retrieve": elapsed}


def grade_chunks_node(state: RAGState) -> dict:
    t0 = time.perf_counter()
    grader_chain = _GRADER_PROMPT | _get_haiku().with_structured_output(ChunkGrade)

    query = state["query"]
    verdicts: list[dict] = []
    scores: list[float] = []

    for chunk in state["chunks"]:
        text = chunk["payload"].get("text", "")
        try:
            grade: ChunkGrade = grader_chain.invoke({"question": query, "chunk": text})
        except Exception as exc:
            logger.warning("[crag] grader failed for chunk %s: %s — defaulting AMBIGUOUS", chunk["chunk_id"], exc)
            grade = ChunkGrade(score=0.5, verdict="AMBIGUOUS", reason="grader error")

        verdicts.append({
            "chunk_id": chunk["chunk_id"],
            "verdict": grade.verdict,
            "score": grade.score,
            "reason": grade.reason,
        })
        scores.append(grade.score)
        logger.debug("[crag] chunk %s → %.2f %s", chunk["chunk_id"], grade.score, grade.verdict)

    if any(s >= CRAG_UPPER_TH for s in scores):
        overall = "CORRECT"
    elif all(s < CRAG_LOWER_TH for s in scores):
        overall = "INCORRECT"
    else:
        overall = "AMBIGUOUS"

    good_chunks = [
        c for c, v in zip(state["chunks"], verdicts)
        if v["score"] >= CRAG_LOWER_TH
    ]

    elapsed = time.perf_counter() - t0
    logger.info("[crag] grade: overall=%s good=%d/%d  (%.2fs)", overall, len(good_chunks), len(state["chunks"]), elapsed)
    return {
        "chunk_verdicts": verdicts,
        "overall_verdict": overall,
        "good_chunks": good_chunks,
        "latency_grade": elapsed,
        **({"fallback_reason": "all_irrelevant"} if overall == "INCORRECT" else {}),
    }


def load_history_node(state: RAGState) -> dict:
    thread_id = state["thread_id"]
    history = get_history(thread_id, HISTORY_TURNS)
    summary_data = get_thread_summary(thread_id)
    return {"history": history, "summary": summary_data["summary"] or ""}


def generate_node(state: RAGState, *, llm) -> dict:
    t0 = time.perf_counter()
    prompt_chunks = state["good_chunks"] if state["overall_verdict"] == "AMBIGUOUS" else state["chunks"]

    history = state.get("history", [])
    parts = []
    if state.get("summary"):
        parts.append(f"## Prior Conversation Summary\n{state['summary']}")
    if history:
        conv = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history
        )
        parts.append(f"## Recent Conversation\n{conv}")
    parts.append(f"## Source Chunks\n{_format_chunks(prompt_chunks)}")
    parts.append(f"## Question\n{state['query']}")
    user_message = "\n\n".join(parts)

    try:
        answer = llm.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=MAX_RESPONSE_TOKENS,
        )
    except Exception as exc:
        logger.error("[crag] generation failed: %s", exc)
        return {
            "answer": _FALLBACK_LLM_ERROR,
            "chunks_used": [c["payload"] for c in prompt_chunks],
            "fallback": True,
            "fallback_reason": "llm_error",
        }

    elapsed = time.perf_counter() - t0
    logger.info("[crag] generate: done  (%.2fs)", elapsed)
    return {
        "answer": answer,
        "chunks_used": [c["payload"] for c in prompt_chunks],
        "fallback": False,
        "latency_generate": elapsed,
    }


def fallback_node(state: RAGState) -> dict:
    reason = state.get("fallback_reason", "no_chunks")
    if reason == "all_irrelevant":
        msg = _FALLBACK_ALL_IRRELEVANT
    elif reason == "llm_error":
        msg = _FALLBACK_LLM_ERROR
    else:
        msg = _FALLBACK_NO_RESULTS

    return {
        "answer": msg,
        "chunks_used": [],
        "fallback": True,
    }


def persist_node(state: RAGState) -> dict:
    thread_id = state["thread_id"]
    try:
        is_first = not get_history(thread_id, 1)
        add_message(thread_id, "user", state["query"])
        if is_first:
            title = state["query"][:60].strip()
            set_thread_title(thread_id, title)
        add_message(
            thread_id,
            "assistant",
            state.get("answer", ""),
            metadata={
                "chunks_used": state.get("chunks_used", []),
                "fallback": state.get("fallback", False),
                "fallback_reason": state.get("fallback_reason"),
                "latency_retrieve": state.get("latency_retrieve"),
                "latency_grade": state.get("latency_grade"),
                "latency_generate": state.get("latency_generate"),
            },
        )
    except Exception as exc:
        logger.error("[crag] persist_node failed (answer already generated): %s", exc)
    return {}


def summarize_node(state: RAGState) -> dict:
    """
    Incrementally fold overflowed messages into the rolling summary.
    Runs after persist on every turn; only calls Haiku when new messages
    have fallen off the HISTORY_TURNS window since the last summary update.
    Non-fatal — a failure here does not affect the already-delivered answer.
    """
    thread_id = state["thread_id"]
    try:
        total = get_message_count(thread_id)
        overflow = total - HISTORY_TURNS          # messages no longer in the raw window
        if overflow <= 0:
            return {}                             # window not yet full — nothing to summarize

        summary_data = get_thread_summary(thread_id)
        already_summarized = summary_data["summary_msg_count"]
        new_overflow = overflow - already_summarized
        if new_overflow <= 0:
            return {}                             # summary already covers all overflowed messages

        new_msgs = get_messages_slice(thread_id, already_summarized, new_overflow)
        new_turns_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in new_msgs
        )

        chain = _SUMMARIZER_PROMPT | _get_haiku()
        result = chain.invoke({
            "existing_summary": summary_data["summary"] or "(none yet)",
            "new_turns": new_turns_text,
        })
        new_summary = result.content.strip()
        set_thread_summary(thread_id, new_summary, overflow)
        logger.info(
            "[crag] summary updated: %d messages now summarized (added %d)",
            overflow, new_overflow,
        )
    except Exception as exc:
        logger.error("[crag] summarize_node failed (non-fatal): %s", exc)
    return {}


# ── Bypass node (used when CRAG grader is disabled) ──────────────────────────

def bypass_grade_node(state: RAGState) -> dict:
    """Skip grading — treat all retrieved chunks as CORRECT."""
    logger.info("[crag] grade: BYPASSED — all %d chunks accepted as CORRECT", len(state["chunks"]))
    return {
        "chunk_verdicts": [],
        "overall_verdict": "CORRECT",
        "good_chunks": state["chunks"],
        "latency_grade": 0.0,
    }


# ── Routers ───────────────────────────────────────────────────────────────────

def _route_after_classify(state: RAGState) -> str:
    return "persist" if state.get("intent") == "direct" else "retrieve"


def _route_after_retrieve(state: RAGState, *, use_crag_grader: bool) -> str:
    if not state.get("chunks"):
        return "fallback"
    return "grade_chunks" if use_crag_grader else "bypass_grade"


def _route_after_grade(state: RAGState) -> str:
    return "fallback" if state.get("overall_verdict") == "INCORRECT" else "load_history"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_crag_graph(qdrant_client, llm, *, use_crag_grader: bool = True):
    """
    Compile and return the CRAG LangGraph.

    Call once at app startup; reuse the compiled graph across queries.

    Args:
        use_crag_grader: When False, skip the Haiku chunk grader and treat all
                         retrieved chunks as CORRECT (useful for A/B comparison).

    Usage:
        graph = build_crag_graph(qdrant_client=client, llm=llm_engine)
        result = graph.invoke({"query": "...", "thread_id": "...", "qdrant_filter": None})
    """
    import functools

    builder = StateGraph(RAGState)

    builder.add_node("classify", classify_node)
    builder.add_node("retrieve", functools.partial(retrieve_node, qdrant_client=qdrant_client))
    builder.add_node("grade_chunks", grade_chunks_node)
    builder.add_node("bypass_grade", bypass_grade_node)
    builder.add_node("load_history", load_history_node)
    builder.add_node("generate", functools.partial(generate_node, llm=llm))
    builder.add_node("fallback", fallback_node)
    builder.add_node("persist", persist_node)
    builder.add_node("summarize", summarize_node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", _route_after_classify, {
        "persist": "persist",
        "retrieve": "retrieve",
    })
    builder.add_conditional_edges(
        "retrieve",
        functools.partial(_route_after_retrieve, use_crag_grader=use_crag_grader),
        {"grade_chunks": "grade_chunks", "bypass_grade": "bypass_grade", "fallback": "fallback"},
    )
    builder.add_conditional_edges("grade_chunks", _route_after_grade, {
        "load_history": "load_history",
        "fallback": "fallback",
    })
    builder.add_edge("bypass_grade", "load_history")
    builder.add_edge("load_history", "generate")
    builder.add_edge("generate", "persist")
    builder.add_edge("fallback", "persist")
    builder.add_edge("persist", "summarize")
    builder.add_edge("summarize", END)

    return builder.compile()
