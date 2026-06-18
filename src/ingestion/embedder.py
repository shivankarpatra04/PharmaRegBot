"""Embedding generation (OpenAI or NVIDIA).

The embedding provider is selected via ``EMBEDDING_PROVIDER``:

  * ``openai``  -> ``text-embedding-3-small`` (default).
  * ``nvidia``  -> an NVIDIA API-catalog embedding model (OpenAI-compatible
    endpoint). NVIDIA retrieval embedders require an ``input_type`` of
    ``passage`` (for stored documents) or ``query`` (for searches), which this
    module sets automatically.

Both providers use the OpenAI Python SDK — NVIDIA simply points it at a
different ``base_url``. Clients are created lazily and cached so a missing key
only errors when embeddings are actually requested.
"""

from __future__ import annotations

from openai import OpenAI

from src.utils.config import Config, get_logger

logger = get_logger(__name__)

# provider name -> cached OpenAI-compatible client
_clients: dict[str, OpenAI] = {}

# NVIDIA's hosted embedding endpoints prefer smaller batches than OpenAI.
_BATCH_SIZE = {"openai": 100, "nvidia": 50}


def _get_client(provider: str) -> OpenAI:
    if provider in _clients:
        return _clients[provider]

    if provider == "nvidia":
        if not Config.NVIDIA_API_KEY:
            raise ValueError(
                "NVIDIA_API_KEY is not set. NVIDIA embeddings require it. "
                "Add it to your .env file."
            )
        client = OpenAI(api_key=Config.NVIDIA_API_KEY, base_url=Config.NVIDIA_BASE_URL)
    else:  # openai
        if not Config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. OpenAI embeddings require it. "
                "Add it to your .env file."
            )
        client = OpenAI(api_key=Config.OPENAI_API_KEY)

    _clients[provider] = client
    return client


def _embedding_model(provider: str) -> str:
    return Config.NVIDIA_EMBEDDING_MODEL if provider == "nvidia" else Config.EMBEDDING_MODEL


def _embed_batch(
    client: OpenAI, provider: str, model: str, batch: list[str], input_type: str
) -> list[list[float]]:
    if provider == "nvidia":
        # NVIDIA retrieval embedders need input_type; truncate long inputs safely.
        response = client.embeddings.create(
            model=model,
            input=batch,
            extra_body={"input_type": input_type, "truncate": "END"},
        )
    else:
        response = client.embeddings.create(model=model, input=batch)
    return [item.embedding for item in response.data]


def embed_texts(
    texts: list[str],
    input_type: str = "passage",
    batch_size: int | None = None,
) -> list[list[float]]:
    """Return embedding vectors for a list of texts, batching requests.

    ``input_type`` is ``passage`` for stored documents and ``query`` for search
    queries (used only by NVIDIA models; ignored by OpenAI).
    """
    if not texts:
        return []

    provider = Config.EMBEDDING_PROVIDER
    client = _get_client(provider)
    model = _embedding_model(provider)
    batch_size = batch_size or _BATCH_SIZE.get(provider, 100)

    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings.extend(_embed_batch(client, provider, model, batch, input_type))
        logger.info(
            "Embedded %d/%d chunks via %s (%s)",
            min(start + batch_size, len(texts)),
            len(texts),
            provider,
            model,
        )

    return embeddings


def embed_query(text: str) -> list[float]:
    """Return the embedding vector for a single query string."""
    return embed_texts([text], input_type="query")[0]
