import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchAny,
    MatchText,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import EMBEDDING_DIM, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_PATH, QDRANT_URL
from src.ingestion.metadata_extractor import EnrichedChunk

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            _client = QdrantClient(path=QDRANT_PATH)
        logger.info("[vector_store] Qdrant client initialised")
    return _client


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("[vector_store] Created collection '%s'", QDRANT_COLLECTION)


def chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def upsert_chunks(
    client: QdrantClient,
    enriched_chunks: list[EnrichedChunk],
    vectors: list[list[float]],
) -> None:
    if not enriched_chunks:
        return

    points = [
        PointStruct(
            id=chunk_id_to_point_id(chunk.chunk_id),
            vector=vector,
            payload={
                "chunk_id": chunk.chunk_id,
                "parent_qa_id": chunk.parent_qa_id,
                "chunk_index": chunk.chunk_index,
                "qa_index": chunk.qa_index,
                "text": chunk.text,
                "question_header": chunk.question_header,
                "source_file": chunk.source_file,
                "parse_tier": chunk.parse_tier,
                "doctor_name": chunk.doctor_name,
                "specialty": chunk.specialty,
                "hospital": chunk.hospital,
                "location": chunk.location,
                "geographic_region": chunk.geographic_region,
                "topic_tags": chunk.topic_tags,
                "wave_reference": chunk.wave_reference,
            },
        )
        for chunk, vector in zip(enriched_chunks, vectors)
    ]

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    logger.info("[vector_store] Upserted %d points", len(points))


def scroll_all_chunks(client: QdrantClient) -> list[dict]:
    """Fetch all (chunk_id, text) pairs for BM25 rebuild. Handles pagination."""
    all_points = []
    offset = None

    while True:
        results, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            with_payload=["chunk_id", "text"],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        all_points.extend(results)
        if offset is None:
            break

    return [{"chunk_id": p.payload["chunk_id"], "text": p.payload["text"]} for p in all_points]


def search_dense(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int,
    qdrant_filter: Optional[Filter] = None,
) -> list[dict]:
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    ).points
    return [
        {"chunk_id": r.payload["chunk_id"], "score": r.score, "payload": r.payload}
        for r in results
    ]


def collection_count(client: QdrantClient) -> int:
    try:
        info = client.get_collection(QDRANT_COLLECTION)
        return info.points_count or 0
    except Exception:
        return 0
