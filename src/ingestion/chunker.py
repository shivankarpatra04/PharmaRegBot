"""Recursive, token-aware text chunking.

Uses LangChain's ``RecursiveCharacterTextSplitter`` with a tiktoken encoder so
that ``chunk_size`` / ``chunk_overlap`` are measured in tokens (1000 / 200),
matching the embedding model's tokenisation. If tiktoken is unavailable for any
reason the splitter falls back to character-based splitting.

``chunk_document`` turns a loaded document (see ``loader.load_document``) into a
flat list of chunk dicts ready for embedding and storage::

    {
        "id": "<file_hash>_<chunk_index>",
        "text": "...",
        "metadata": {
            "source_file": "ICH_Q7.pdf",
            "page_number": 12,
            "doc_type": "ICH",
            "file_type": "pdf",
            "chunk_index": 7,
            "file_hash": "ab12...",
        },
    }
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import Config, get_logger

logger = get_logger(__name__)


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Create a token-aware splitter, falling back to character counts."""
    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "tiktoken-based splitting unavailable (%s); "
            "falling back to character-based splitting.",
            exc,
        )
        return RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )


def chunk_document(document: dict, file_hash: str) -> list[dict]:
    """Split a loaded document into embedding-ready chunks with metadata."""
    splitter = _build_splitter()

    source_file = document["source_file"]
    doc_type = document["doc_type"]
    file_type = document["file_type"]

    chunks: list[dict] = []
    chunk_index = 0

    for page in document["pages"]:
        page_number = int(page["page_number"])
        for piece in splitter.split_text(page["text"]):
            if not piece.strip():
                continue
            chunks.append(
                {
                    "id": f"{file_hash}_{chunk_index}",
                    "text": piece,
                    "metadata": {
                        "source_file": source_file,
                        "page_number": page_number,
                        "doc_type": doc_type,
                        "file_type": file_type,
                        "chunk_index": chunk_index,
                        "file_hash": file_hash,
                    },
                }
            )
            chunk_index += 1

    logger.info("Split '%s' into %d chunks", source_file, len(chunks))
    return chunks
