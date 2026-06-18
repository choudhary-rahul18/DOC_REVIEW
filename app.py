import logging
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# Suppress noisy third-party loggers
for _noisy in ("httpx", "httpcore", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from config import CRAG_GRADER_ENABLED, GENERATION_MODEL
from src.generation.crag_pipeline import build_crag_graph
from src.generation.llm_engine import build_llm_engine
from src.ingestion.pipeline import ingest_document
from src.memory.conversation import (
    create_thread,
    delete_thread,
    get_all_messages,
    list_threads,
)
from src.retrieval.vector_store import get_client

st.set_page_config(page_title="COVID-19 Interview RAG", layout="wide", page_icon="🏥")


# ── Cached resources (built once per (process × grader flag) combination) ──────

@st.cache_resource
def _get_resources(use_crag_grader: bool = True):
    qdrant = get_client()
    llm = build_llm_engine("anthropic", GENERATION_MODEL)
    graph = build_crag_graph(qdrant_client=qdrant, llm=llm, use_crag_grader=use_crag_grader)
    return qdrant, graph


# ── Helpers ────────────────────────────────────────────────────────────────────

def _render_latencies(latencies: dict) -> None:
    parts = []
    for key, label in [("retrieve", "retrieve"), ("grade", "grade"), ("generate", "generate")]:
        val = latencies.get(key)
        if val is not None:
            parts.append(f"{label} {val:.2f}s")
    if parts:
        st.caption("⏱ " + "  ·  ".join(parts))


def _render_assistant_message(answer: str, chunks_used: list[dict]) -> None:
    st.markdown(answer)

    if chunks_used:
        with st.expander(f"Sources ({len(chunks_used)} chunks)"):
            for i, src in enumerate(chunks_used, 1):
                doctor = src.get("doctor_name") or "Unknown"
                meta_parts = [p for p in [src.get("specialty"), src.get("location"), src.get("hospital")] if p]
                st.markdown(f"**[{i}] {doctor}**" + (f"  ·  {' · '.join(meta_parts)}" if meta_parts else ""))
                if src.get("text"):
                    st.markdown(src["text"])
                if i < len(chunks_used):
                    st.divider()


# ── Session state ──────────────────────────────────────────────────────────────

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# Stores rich result data keyed by (thread_id, answer_clean) for re-render
if "result_cache" not in st.session_state:
    st.session_state.result_cache: dict[tuple, dict] = {}


# ── Sidebar: conversation management ──────────────────────────────────────────

with st.sidebar:
    st.title("Conversations")

    with st.expander("Pipeline settings", expanded=False):
        use_crag_grader = st.toggle(
            "CRAG chunk grader",
            value=CRAG_GRADER_ENABLED,
            help="When ON: Haiku grades each chunk and routes CORRECT/AMBIGUOUS/INCORRECT. "
                 "When OFF: all retrieved chunks pass straight to generation.",
        )

    qdrant_client, crag_graph = _get_resources(use_crag_grader=use_crag_grader)

    if st.button("＋  New conversation", use_container_width=True, type="primary"):
        tid = create_thread()
        st.session_state.thread_id = tid
        st.rerun()

    st.divider()

    threads = list_threads()
    if threads:
        for t in threads:
            tid = t["thread_id"]
            n_msgs = t["message_count"] or 0
            title = t.get("title") or f"Thread {tid[:8]}"
            label = f"{title}  ({n_msgs} msg{'s' if n_msgs != 1 else ''})"
            active = tid == st.session_state.thread_id
            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                if st.button(
                    label,
                    key=f"sel_{tid}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    st.session_state.thread_id = tid
                    st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{tid}", help="Delete thread"):
                    delete_thread(tid)
                    if st.session_state.thread_id == tid:
                        st.session_state.thread_id = None
                    st.rerun()
    else:
        st.caption("No conversations yet.")


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_ingest, tab_chat = st.tabs(["📂  Ingest Documents", "💬  Chat"])


# ── Tab 1: Ingest ──────────────────────────────────────────────────────────────

with tab_ingest:
    st.header("Ingest Interview Documents")
    st.caption(
        "Upload PDF or DOCX interview transcripts. "
        "Each file is parsed, chunked, metadata-extracted (Claude Haiku), embedded, "
        "and upserted into the Qdrant vector store. Re-uploading the same file is safe — upsert is idempotent."
    )

    uploaded_files = st.file_uploader(
        "Drop interview transcripts here",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Ingest", type="primary", use_container_width=False):
            for uploaded in uploaded_files:
                st.markdown(f"#### {uploaded.name}")
                log_box = st.empty()
                log_lines: list[str] = []

                def _make_cb(lines: list[str], box: st.empty):  # noqa: E306
                    def _cb(msg: str) -> None:
                        lines.append(msg)
                        box.code("\n".join(lines), language=None)
                    return _cb

                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = Path(tmp.name)

                result = ingest_document(
                    tmp_path,
                    client=qdrant_client,
                    progress_callback=_make_cb(log_lines, log_box),
                )
                tmp_path.unlink(missing_ok=True)

                if result["status"] == "ok":
                    st.success(
                        f"✓ {result['chunks']} chunks ingested"
                        + (f" — {result['doctor_name']}" if result.get("doctor_name") else "")
                    )
                else:
                    st.error(f"✗ {result['status']}")


# ── Tab 2: Chat ────────────────────────────────────────────────────────────────

with tab_chat:
    if st.session_state.thread_id is None:
        st.info("Create a new conversation from the sidebar to get started.")
    else:
        thread_id = st.session_state.thread_id
        _active_threads = {t["thread_id"]: t for t in list_threads()}
        _active_title = (_active_threads.get(thread_id) or {}).get("title") or f"Thread {thread_id[:8]}"
        st.caption(_active_title)

        # Render conversation history from SQLite
        history = get_all_messages(thread_id)
        for msg in history:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    # Prefer in-session cache; fall back to persisted metadata from DB
                    cache_key = (thread_id, msg["content"])
                    rich = st.session_state.result_cache.get(cache_key)
                    db_meta = msg.get("metadata")
                    if rich and not rich["fallback"]:
                        _render_assistant_message(msg["content"], rich["chunks_used"])
                        _render_latencies(rich.get("latencies", {}))
                    elif db_meta and not db_meta.get("fallback"):
                        _render_assistant_message(msg["content"], db_meta.get("chunks_used") or [])
                        _render_latencies({
                            "retrieve": db_meta.get("latency_retrieve"),
                            "grade": db_meta.get("latency_grade"),
                            "generate": db_meta.get("latency_generate"),
                        })
                    else:
                        st.markdown(msg["content"])
                else:
                    st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask about the COVID-19 doctor interviews…"):
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving and generating…"):
                    _t0 = time.perf_counter()
                    result = crag_graph.invoke({
                        "query": prompt,
                        "thread_id": thread_id,
                        "qdrant_filter": None,
                    })
                    logging.getLogger(__name__).info(
                        "[app] total query latency: %.2fs", time.perf_counter() - _t0
                    )

                answer: str = result.get("answer", "")
                chunks_used: list[dict] = result.get("chunks_used") or []
                fallback: bool = result.get("fallback", False)
                latencies: dict = {
                    "retrieve": result.get("latency_retrieve"),
                    "grade": result.get("latency_grade"),
                    "generate": result.get("latency_generate"),
                }

                if not fallback:
                    _render_assistant_message(answer, chunks_used)
                else:
                    st.markdown(answer)

                _render_latencies(latencies)

                # Cache rich data for subsequent re-renders
                st.session_state.result_cache[(thread_id, answer)] = {
                    "chunks_used": chunks_used,
                    "fallback": fallback,
                    "latencies": latencies,
                }

            st.rerun()
