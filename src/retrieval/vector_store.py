"""ChromaDB persistent vector store wrapper.

Wraps a single Chroma collection (``pharma_regulations``) configured for cosine
similarity and stored on disk at ``CHROMA_DB_PATH``. Provides the operations the
rest of the app needs: add, query, dedup-check, list and delete.
"""

from __future__ import annotations

import logging

import chromadb
from chromadb.config import Settings

from src.utils.config import Config, get_logger

logger = get_logger(__name__)

# Telemetry is disabled below, but some ChromaDB / posthog version pairings still
# emit noisy "Failed to send telemetry event" warnings. Silence that logger.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# Chroma limits the number of records per add() call; batch to stay safe.
_ADD_BATCH_SIZE = 1000


class VectorStore:
    """A thin, purpose-built wrapper around a persistent Chroma collection."""

    def __init__(
        self,
        path: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.path = path or Config.CHROMA_DB_PATH
        self.collection_name = collection_name or Config.COLLECTION_NAME

        self.client = chromadb.PersistentClient(
            path=self.path,
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )
        # cosine distance => similarity = 1 - distance
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Vector store ready (collection='%s', path='%s', count=%d)",
            self.collection_name,
            self.path,
            self.count(),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _user_filter(user_id, **extra) -> dict:
        """Build a Chroma ``where`` clause scoped to a user (plus extra equalities)."""
        conditions = [{"user_id": {"$eq": user_id}}]
        for key, value in extra.items():
            conditions.append({key: {"$eq": value}})
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    # ------------------------------------------------------------------
    def count(self) -> int:
        """Total number of stored chunks across all users."""
        return self.collection.count()

    def has_documents(self, user_id) -> bool:
        """Return True if the given user has any indexed chunks."""
        result = self.collection.get(where=self._user_filter(user_id), limit=1)
        return bool(result and result.get("ids"))

    def file_exists(self, user_id, file_hash: str) -> bool:
        """Return True if this user already has a document with this content hash."""
        result = self.collection.get(
            where=self._user_filter(user_id, file_hash=file_hash), limit=1
        )
        return bool(result and result.get("ids"))

    # ------------------------------------------------------------------
    def add(self, user_id, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Add a user's chunks (with embeddings), stamping each with the user_id.

        IDs are namespaced by user so that two users uploading the same file
        (same content hash) don't collide.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        ids = [f"{user_id}_{c['id']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{**c["metadata"], "user_id": user_id} for c in chunks]

        for start in range(0, len(ids), _ADD_BATCH_SIZE):
            end = start + _ADD_BATCH_SIZE
            self.collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
        logger.info("Added %d chunks for user %s", len(ids), user_id)

    # ------------------------------------------------------------------
    def query(self, user_id, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Return up to ``top_k`` of the user's nearest chunks (text/meta/distance)."""
        if self.count() == 0:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            where=self._user_filter(user_id),
            include=["documents", "metadatas", "distances"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    # ------------------------------------------------------------------
    def list_documents(self, user_id) -> list[dict]:
        """Return one entry per source file owned by the user, with chunk counts."""
        result = self.collection.get(
            where=self._user_filter(user_id), include=["metadatas"]
        )
        metadatas = result.get("metadatas") or []

        aggregated: dict[str, dict] = {}
        for meta in metadatas:
            source = meta.get("source_file", "unknown")
            entry = aggregated.setdefault(
                source,
                {
                    "source_file": source,
                    "doc_type": meta.get("doc_type", ""),
                    "file_type": meta.get("file_type", ""),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1

        return sorted(aggregated.values(), key=lambda e: e["source_file"].lower())

    def delete_document(self, user_id, source_file: str) -> None:
        """Remove every chunk of a source file — only if it belongs to the user."""
        self.collection.delete(
            where=self._user_filter(user_id, source_file=source_file)
        )
        logger.info("Deleted '%s' for user %s", source_file, user_id)
