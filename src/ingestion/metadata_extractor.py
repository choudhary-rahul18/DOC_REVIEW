import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from pydantic import BaseModel

from config import METADATA_MODEL
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


# ── Output type ────────────────────────────────────────────────────────────────

@dataclass
class EnrichedChunk:
    chunk_id: str
    parent_qa_id: str
    chunk_index: int
    qa_index: int
    text: str
    question_header: str
    source_file: str
    parse_tier: int
    doctor_name: str = ""
    specialty: str = ""
    hospital: str = ""
    location: str = ""
    geographic_region: str = ""
    topic_tags: list = field(default_factory=list)
    wave_reference: str = ""


# ── Pydantic schema for partial validation ─────────────────────────────────────

class _DocMetadata(BaseModel):
    doctor_name: Optional[str] = None
    specialty: Optional[str] = None
    hospital: Optional[str] = None
    location: Optional[str] = None
    geographic_region: Optional[str] = None


class _LLMResponse(BaseModel):
    metadata: Optional[_DocMetadata] = None
    chunk_tags: Optional[dict] = None


# ── Wave reference regex (closed vocabulary) ───────────────────────────────────

_WAVE_PATTERN = re.compile(
    r'\b('
    r'first\s+wave|second\s+wave|third\s+wave|fourth\s+wave'
    r'|wave\s+[1234]'
    r'|delta(?:\s+wave)?|omicron(?:\s+wave)?|alpha(?:\s+wave)?'
    r'|beta(?:\s+wave)?|gamma(?:\s+wave)?'
    r'|2020\s+wave|2021\s+wave|2022\s+wave'
    r')',
    re.IGNORECASE,
)


def _extract_wave_reference(text: str) -> str:
    match = _WAVE_PATTERN.search(text)
    return match.group(0).lower() if match else ""


# ── LLM prompt ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a medical interview metadata extractor. Extract structured information \
from COVID-19 doctor interview transcripts.

Return ONLY valid JSON with this exact schema — no explanation, no markdown, no preamble:
{
  "metadata": {
    "doctor_name": "full name with title e.g. Dr. Anjali Singh",
    "specialty": "medical specialty e.g. Pediatrician, Neonatologist",
    "hospital": "hospital or clinic name",
    "location": "city, country e.g. Lucknow, India",
    "geographic_region": "state/province, country e.g. Uttar Pradesh, India"
  },
  "chunk_tags": {
    "0": ["tag1", "tag2", "tag3"],
    "1": ["tag1", "tag2", "tag3"]
  }
}

Rules:
- topic_tags: 3-5 semantic concept phrases per chunk. Identify the underlying medical/public \
health concept — not lexical keywords. "overwhelmed and exhausted" → "staff burnout".
- Use null for any metadata field you cannot determine.
- chunk_tags keys are the chunk_index values as strings."""


def _build_prompt(chunks: list[Chunk]) -> str:
    header_text = chunks[0].text[:500] if chunks else ""
    parts = [f"DOCUMENT HEADER:\n{header_text}\n\nCHUNKS:"]
    for chunk in chunks:
        parts.append(f"\n[chunk_index={chunk.chunk_index}]\n{chunk.text}")
    return "\n".join(parts)


# ── 4-layer JSON recovery ──────────────────────────────────────────────────────

def _parse_llm_response(raw: str) -> Optional[_LLMResponse]:
    # Layer 1: direct parse
    try:
        return _LLMResponse(**json.loads(raw.strip()))
    except Exception:
        pass

    # Layer 2: extract largest {...} block (handles LLM preamble/postamble)
    matches = re.findall(r'\{.*\}', raw, re.DOTALL)
    if matches:
        largest = max(matches, key=len)
        try:
            return _LLMResponse(**json.loads(largest))
        except Exception:
            pass

    # Layer 3: extract partial fields with targeted regex
    try:
        partial: dict = {}
        meta_match = re.search(r'"metadata"\s*:\s*(\{[^}]+\})', raw, re.DOTALL)
        tags_match = re.search(r'"chunk_tags"\s*:\s*(\{.*?\})\s*[,}]', raw, re.DOTALL)
        if meta_match:
            partial['metadata'] = json.loads(meta_match.group(1))
        if tags_match:
            partial['chunk_tags'] = json.loads(tags_match.group(1))
        if partial:
            return _LLMResponse(**partial)
    except Exception:
        pass

    return None


# ── Public entry point ─────────────────────────────────────────────────────────

def extract_metadata(chunks: list[Chunk]) -> list[EnrichedChunk]:
    """
    One Claude Haiku call per document.
    Returns EnrichedChunk list with doc-level metadata + per-chunk topic_tags + wave_reference.
    Never raises — on total failure, returns chunks with empty metadata fields.
    """
    if not chunks:
        return []

    source_file = chunks[0].source_file
    client = anthropic.Anthropic()
    llm_response: Optional[_LLMResponse] = None

    try:
        message = client.messages.create(
            model=METADATA_MODEL,
            max_tokens=1500,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_prompt(chunks)}],
        )
        raw = message.content[0].text
        llm_response = _parse_llm_response(raw)
        if llm_response is None:
            logger.error("[metadata_extractor] %s: all JSON recovery layers failed. Raw: %.200s", source_file, raw)
    except Exception as e:
        logger.error("[metadata_extractor] %s: LLM call failed: %s", source_file, e)

    meta = llm_response.metadata if llm_response and llm_response.metadata else None
    doctor_name = (meta.doctor_name or "") if meta else ""
    specialty = (meta.specialty or "") if meta else ""
    hospital = (meta.hospital or "") if meta else ""
    location = (meta.location or "") if meta else ""
    geographic_region = (meta.geographic_region or "") if meta else ""
    chunk_tags_raw: dict = (llm_response.chunk_tags or {}) if llm_response else {}

    enriched: list[EnrichedChunk] = []
    for chunk in chunks:
        tags = chunk_tags_raw.get(str(chunk.chunk_index), [])
        if not isinstance(tags, list):
            tags = []

        enriched.append(EnrichedChunk(
            chunk_id=chunk.chunk_id,
            parent_qa_id=chunk.parent_qa_id,
            chunk_index=chunk.chunk_index,
            qa_index=chunk.qa_index,
            text=chunk.text,
            question_header=chunk.question_header,
            source_file=chunk.source_file,
            parse_tier=chunk.parse_tier,
            doctor_name=doctor_name,
            specialty=specialty,
            hospital=hospital,
            location=location,
            geographic_region=geographic_region,
            topic_tags=tags,
            wave_reference=_extract_wave_reference(chunk.text),
        ))

    return enriched
