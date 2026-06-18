import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from config import (
    CHUNK_TOKEN_CEILING,
    EMBEDDING_MODEL,
    MIN_CHUNK_TOKENS,
    SEMANTIC_SIMILARITY_THRESHOLD,
    SLIDING_WINDOW_OVERLAP,
    SPEAKER_DISCOVERY_WINDOW,
    SPEAKER_MIN_FREQUENCY,
)

logger = logging.getLogger(__name__)


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class ParsedDocument:
    text: str
    source_file: str  # filename only, not full path


@dataclass
class Chunk:
    chunk_id: str         # sha256(source_file + str(chunk_index))[:16]
    parent_qa_id: str     # sha256(source_file + str(qa_index))[:16]; equals chunk_id when no Q&A structure
    chunk_index: int      # global sequential index across all sub-chunks in this doc
    qa_index: int         # which Q&A pair this came from; -1 when no Q&A structure
    text: str             # full chunk text: "[Question] ...\n\n[Answer excerpt] ..."
    question_header: str  # the interviewer question alone (for metadata prompt); "" when no Q&A structure
    source_file: str
    parse_tier: int       # 1 = speaker regex, 2 = semantic fallback, 3 = token fallback


# ── Chunker singletons (lazy init, shared across documents) ────────────────────

_semantic_chunker = None
_token_chunker = None


def _get_chunkers():
    global _semantic_chunker, _token_chunker
    if _semantic_chunker is None:
        from chonkie import SemanticChunker, TokenChunker
        _semantic_chunker = SemanticChunker(
            embedding_model=EMBEDDING_MODEL,
            chunk_size=CHUNK_TOKEN_CEILING,
            threshold=SEMANTIC_SIMILARITY_THRESHOLD,
        )
        _token_chunker = TokenChunker(
            chunk_size=CHUNK_TOKEN_CEILING,
            chunk_overlap=SLIDING_WINDOW_OVERLAP,
        )
    return _semantic_chunker, _token_chunker


# ── Public entry point ─────────────────────────────────────────────────────────

def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """
    3-tier cascade:
      Tier 1 — speaker-turn regex + chonkie SemanticChunker on each answer
      Tier 2 — chonkie SemanticChunker on whole doc (no Q&A structure detected)
      Tier 3 — chonkie TokenChunker (always succeeds, last resort)
    Returns empty list only on a complete failure (logged as ERROR).
    """
    semantic_chunker, token_chunker = _get_chunkers()

    chunks = _tier1_speaker_split(doc, semantic_chunker)
    if chunks:
        return chunks

    logger.warning("[chunker] %s: Tier 1 (speaker regex) failed — falling back to Tier 2 (semantic)", doc.source_file)
    chunks = _tier2_semantic(doc, semantic_chunker)
    if chunks:
        return chunks

    logger.warning("[chunker] %s: Tier 2 (semantic) failed — falling back to Tier 3 (token window)", doc.source_file)
    chunks = _tier3_token(doc, token_chunker)
    if chunks:
        return chunks

    logger.error("[chunker] %s: All tiers failed — returning empty chunk list", doc.source_file)
    return []


# ── Tier 1: speaker-turn regex ─────────────────────────────────────────────────

def _discover_speakers(text: str, window: int = SPEAKER_DISCOVERY_WINDOW) -> set[str]:
    """
    Scan the first `window` chars to find candidate speaker labels.
    Pattern: "Word Word Word:" — up to 4 title-cased words before a colon.
    Filters by:
      (a) starts with a known honorific (Dr., Interviewer, Prof., Mr., Ms., Dr)
      (b) OR appears SPEAKER_MIN_FREQUENCY+ times in the full text
    This kills false positives like "PPE:" or "COVID-19:" which appear rarely
    and don't start with an honorific.
    """
    candidate_pattern = re.compile(
        r'\b([A-Z][a-zA-Z]*(?:\.)?(?:\s+[A-Z][a-zA-Z]*){0,3})\s*:'
    )
    HONORIFICS = {"Dr", "Dr.", "Interviewer", "Prof", "Prof.", "Mr", "Mr.", "Ms", "Ms."}

    candidates = candidate_pattern.findall(text[:window])
    speaker_set: set[str] = set()

    for label in candidates:
        label = label.strip()
        first_word = label.split()[0] if label.split() else ""
        if first_word in HONORIFICS:
            speaker_set.add(label)
        elif text.count(label + ":") >= SPEAKER_MIN_FREQUENCY:
            speaker_set.add(label)

    return speaker_set


def _build_speaker_regex(speakers: set[str]) -> re.Pattern:
    """
    Build a precise split regex from discovered speaker labels.
    Sort by length descending so longer labels match before partial overlaps.
    Example: "Dr. Anjali Singh" before "Dr. Anjali".
    """
    escaped = [re.escape(s) for s in sorted(speakers, key=len, reverse=True)]
    pattern = "(" + "|".join(escaped) + r")\s*:"
    return re.compile(pattern)


def _tier1_speaker_split(doc: ParsedDocument, semantic_chunker) -> list[Chunk]:
    """
    Two-pass approach:
      1. Discover speaker labels in the first SPEAKER_DISCOVERY_WINDOW chars
      2. Build a precise regex from those labels and split the full text
      3. Sub-chunk each doctor answer with chonkie SemanticChunker
    Returns [] if fewer than 2 speaker turns found.
    """
    speakers = _discover_speakers(doc.text)
    if not speakers:
        return []

    speaker_regex = _build_speaker_regex(speakers)

    # re.split with a capturing group gives: [pre_text, label1, seg1, label2, seg2, ...]
    parts = speaker_regex.split(doc.text)
    if len(parts) < 3:  # need at least one (label, segment) pair
        return []

    # Pair up (label, text) tuples — parts[0] is pre-text (discard if empty)
    turns: list[tuple[str, str]] = []
    i = 1
    while i + 1 < len(parts):
        label = parts[i].strip()
        segment = parts[i + 1].strip()
        if segment:
            turns.append((label, segment))
        i += 2

    if len(turns) < 2:
        return []

    # Identify which speaker is the interviewer (heuristic: contains "Interviewer")
    interviewer_labels = {l for l, _ in turns if "interviewer" in l.lower()}

    # Pair consecutive (question, answer) turns
    # A "question" turn is one from the interviewer, "answer" from the doctor.
    # We scan for alternating interviewer → doctor patterns.
    qa_pairs: list[tuple[str, str, str]] = []  # (question_text, answer_text, answer_speaker)
    i = 0
    while i < len(turns):
        label, text = turns[i]
        if label in interviewer_labels and i + 1 < len(turns):
            next_label, next_text = turns[i + 1]
            if next_label not in interviewer_labels:
                qa_pairs.append((text, next_text, next_label))
                i += 2
                continue
        i += 1

    if not qa_pairs:
        # All turns identified but no alternating Q&A pairs — treat each turn as its own chunk
        return _turns_to_chunks(turns, doc.source_file, semantic_chunker)

    return _qa_pairs_to_chunks(qa_pairs, doc.source_file, semantic_chunker)


def _qa_pairs_to_chunks(
    qa_pairs: list[tuple[str, str, str]],
    source_file: str,
    semantic_chunker,
) -> list[Chunk]:
    """Convert (question, answer, speaker) triples into Chunks with semantic sub-chunking."""
    all_chunks: list[Chunk] = []
    chunk_index = 0

    for qa_index, (question, answer, _) in enumerate(qa_pairs):
        qa_id = _make_id(source_file, qa_index, prefix="qa")
        sub_chunks = _sub_chunk_answer(
            question=question,
            answer=answer,
            source_file=source_file,
            qa_index=qa_index,
            qa_id=qa_id,
            start_chunk_index=chunk_index,
            semantic_chunker=semantic_chunker,
        )
        all_chunks.extend(sub_chunks)
        chunk_index += len(sub_chunks)

    return all_chunks


def _turns_to_chunks(
    turns: list[tuple[str, str]],
    source_file: str,
    semantic_chunker,
) -> list[Chunk]:
    """Fallback within Tier 1: Q&A pairing failed but turns were found. Each turn → chunks."""
    all_chunks: list[Chunk] = []
    chunk_index = 0

    for qa_index, (label, text) in enumerate(turns):
        qa_id = _make_id(source_file, qa_index, prefix="qa")
        sub_chunks = _sub_chunk_answer(
            question="",
            answer=text,
            source_file=source_file,
            qa_index=qa_index,
            qa_id=qa_id,
            start_chunk_index=chunk_index,
            semantic_chunker=semantic_chunker,
        )
        all_chunks.extend(sub_chunks)
        chunk_index += len(sub_chunks)

    return all_chunks


def _sub_chunk_answer(
    question: str,
    answer: str,
    source_file: str,
    qa_index: int,
    qa_id: str,
    start_chunk_index: int,
    semantic_chunker,
) -> list[Chunk]:
    """
    Semantically sub-chunk a single doctor answer using chonkie SemanticChunker.
    Prepends the question as a header to every resulting sub-chunk for embedding context.
    """
    if not answer.strip():
        return []

    try:
        chonkie_chunks = semantic_chunker.chunk(answer)
        texts = [c.text for c in chonkie_chunks if c.text.strip()]
    except Exception as e:
        logger.warning("[chunker] SemanticChunker failed on answer in %s: %s — using whole answer", source_file, e)
        texts = [answer]

    if not texts:
        texts = [answer]

    texts = _merge_short_chunks(texts, MIN_CHUNK_TOKENS)

    chunks: list[Chunk] = []
    for i, text in enumerate(texts):
        chunk_index = start_chunk_index + i
        if question:
            full_text = f"[Question] {question}\n\n[Answer excerpt] {text}"
        else:
            full_text = text

        chunks.append(Chunk(
            chunk_id=_make_id(source_file, chunk_index),
            parent_qa_id=qa_id,
            chunk_index=chunk_index,
            qa_index=qa_index,
            text=full_text,
            question_header=question,
            source_file=source_file,
            parse_tier=1,
        ))

    return chunks


# ── Tier 2: semantic chunking on whole doc ─────────────────────────────────────

def _tier2_semantic(doc: ParsedDocument, semantic_chunker) -> list[Chunk]:
    """
    No Q&A structure detected. Run chonkie SemanticChunker on the full document text.
    Returns [] if fewer than 2 chunks produced (document too short or chunker failed).
    """
    try:
        chonkie_chunks = semantic_chunker.chunk(doc.text)
        texts = [c.text for c in chonkie_chunks if c.text.strip()]
    except Exception as e:
        logger.warning("[chunker] Tier 2 SemanticChunker failed on %s: %s", doc.source_file, e)
        return []

    if len(texts) < 2:
        return []

    return [
        Chunk(
            chunk_id=_make_id(doc.source_file, i),
            parent_qa_id=_make_id(doc.source_file, i),
            chunk_index=i,
            qa_index=-1,
            text=text,
            question_header="",
            source_file=doc.source_file,
            parse_tier=2,
        )
        for i, text in enumerate(texts)
    ]


# ── Tier 3: token window (last resort) ────────────────────────────────────────

def _tier3_token(doc: ParsedDocument, token_chunker) -> list[Chunk]:
    """
    Fixed-size token windows with overlap. Always produces at least 1 chunk.
    """
    try:
        chonkie_chunks = token_chunker.chunk(doc.text)
        texts = [c.text for c in chonkie_chunks if c.text.strip()]
    except Exception as e:
        logger.error("[chunker] Tier 3 TokenChunker failed on %s: %s", doc.source_file, e)
        texts = [doc.text]  # absolute last resort: whole doc as one chunk

    return [
        Chunk(
            chunk_id=_make_id(doc.source_file, i),
            parent_qa_id=_make_id(doc.source_file, i),
            chunk_index=i,
            qa_index=-1,
            text=text,
            question_header="",
            source_file=doc.source_file,
            parse_tier=3,
        )
        for i, text in enumerate(texts)
    ]


# ── Utility ────────────────────────────────────────────────────────────────────

def _make_id(source_file: str, index: int, prefix: str = "chunk") -> str:
    raw = f"{prefix}::{source_file}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _approx_tokens(text: str) -> int:
    # 1 word ≈ 1.3 tokens for English; good enough for a minimum-size guard
    return int(len(text.split()) * 1.3)


def _merge_short_chunks(texts: list[str], min_tokens: int) -> list[str]:
    """
    Merge any sub-chunk below min_tokens into its predecessor.
    Runs left-to-right: if the predecessor is short, the next chunk absorbs into it;
    if the current chunk is short, it folds into the predecessor.
    """
    if len(texts) <= 1:
        return texts

    result = [texts[0]]
    for text in texts[1:]:
        if _approx_tokens(result[-1]) < min_tokens or _approx_tokens(text) < min_tokens:
            result[-1] = result[-1].rstrip() + " " + text.lstrip()
        else:
            result.append(text)
    return result
