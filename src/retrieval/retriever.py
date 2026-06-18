"""Semantic retrieval over the vector store.

Embeds the user query with the same embedding model used at ingestion time,
runs a cosine-similarity search, converts Chroma distances into similarity
scores and filters by ``SIMILARITY_THRESHOLD``.

Each returned chunk is a dict::

    {"text": "...", "metadata": {...}, "score": 0.57}
"""

from __future__ import annotations

from src.ingestion.embedder import embed_query
from src.retrieval.vector_store import VectorStore
from src.utils.config import Config, get_logger

logger = get_logger(__name__)


def retrieve(
    query: str,
    vector_store: VectorStore,
    user_id,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[dict]:
    """Return the most relevant chunks for a query, scoped to ``user_id``.

    Returns an empty list when nothing clears the similarity threshold (the
    caller then surfaces the "no relevant information" message).
    """
    top_k = top_k or Config.TOP_K
    threshold = Config.SIMILARITY_THRESHOLD if threshold is None else threshold

    query_embedding = embed_query(query)
    raw_results = vector_store.query(user_id, query_embedding, top_k=top_k)

    relevant: list[dict] = []
    for item in raw_results:
        # cosine distance in [0, 2]; similarity = 1 - distance
        similarity = 1.0 - float(item["distance"])
        if similarity >= threshold:
            relevant.append(
                {
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "score": similarity,
                }
            )

    logger.info(
        "Retrieved %d/%d chunks above threshold %.2f",
        len(relevant),
        len(raw_results),
        threshold,
    )
    return relevant
