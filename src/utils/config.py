"""Central configuration.

Loads every environment variable used by PharmaRegBot from a local ``.env``
file (via python-dotenv) and exposes them through a single ``Config`` object.
Also provides a shared, timestamped logger factory.

No API keys are ever hardcoded — they are read exclusively from the
environment.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

# Load variables from a .env file in the project root (if present).
# Existing real environment variables always win over the file.
load_dotenv(override=False)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Config:
    """Application configuration sourced entirely from the environment."""

    # --- API keys ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "").strip()

    # NVIDIA's API catalog is OpenAI-compatible; reuse the OpenAI SDK with this
    # base URL for both chat completions and embeddings.
    NVIDIA_BASE_URL: str = os.getenv(
        "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
    ).strip()

    # --- LLM provider / models ---
    # provider: "openai" | "anthropic" | "nvidia"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o").strip()

    # --- Embeddings ---
    # provider: "openai" | "nvidia"
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
    # 8192-token context (fits our ~1000-token chunks without truncation) and
    # supports the required query/passage input types.
    NVIDIA_EMBEDDING_MODEL: str = os.getenv(
        "NVIDIA_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-1b-v2"
    ).strip()

    # Sensible default chat models per provider when LLM_MODEL doesn't match.
    ANTHROPIC_DEFAULT_MODEL: str = "claude-3-5-sonnet-latest"
    NVIDIA_DEFAULT_MODEL: str = "meta/llama-3.3-70b-instruct"

    # --- Storage ---
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db").strip()
    DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "./data/documents").strip()
    COLLECTION_NAME: str = "pharma_regulations"

    # --- Supabase (user-account database / authentication) ---
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "").strip()
    # Use the service_role key (server-side, bypasses RLS). Keep it secret in .env.
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "").strip()
    SUPABASE_USERS_TABLE: str = os.getenv("SUPABASE_USERS_TABLE", "users").strip()

    # --- Retrieval / chunking ---
    SIMILARITY_THRESHOLD: float = _get_float("SIMILARITY_THRESHOLD", 0.4)
    TOP_K: int = 5
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # --- Bilingual behaviour ---
    # "translate" -> generate in English then translate to Hindi (spec default)
    # "native"    -> answer directly in Hindi using the Hindi system prompt
    HINDI_RESPONSE_MODE: str = os.getenv("HINDI_RESPONSE_MODE", "translate").strip().lower()

    # --- Selectable chat models exposed in the UI ---
    MODEL_OPTIONS: dict[str, list[str]] = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "anthropic": [
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
        ],
        "nvidia": [
            "meta/llama-3.3-70b-instruct",
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-8b-instruct",
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "mistralai/mixtral-8x7b-instruct-v0.1",
        ],
    }

    # Providers that speak the OpenAI chat/embeddings API (handled identically).
    OPENAI_COMPATIBLE_PROVIDERS = {"openai", "nvidia"}

    # ------------------------------------------------------------------
    @classmethod
    def _provider_default(cls, provider: str) -> str:
        return {
            "anthropic": cls.ANTHROPIC_DEFAULT_MODEL,
            "nvidia": cls.NVIDIA_DEFAULT_MODEL,
        }.get(provider, cls.MODEL_OPTIONS["openai"][0])

    @classmethod
    def default_model_for(cls, provider: str) -> str:
        """Return a reasonable default chat model for the given provider."""
        provider = (provider or "").lower()
        options = cls.MODEL_OPTIONS.get(provider, cls.MODEL_OPTIONS["openai"])
        # Honour an explicitly configured LLM_MODEL only if it fits the provider.
        if cls.LLM_MODEL in options:
            return cls.LLM_MODEL
        return cls._provider_default(provider)

    @classmethod
    def _key_for_provider(cls, provider: str) -> tuple[str, str]:
        """Return (env_var_name, value) of the API key a provider needs."""
        return {
            "openai": ("OPENAI_API_KEY", cls.OPENAI_API_KEY),
            "anthropic": ("ANTHROPIC_API_KEY", cls.ANTHROPIC_API_KEY),
            "nvidia": ("NVIDIA_API_KEY", cls.NVIDIA_API_KEY),
        }.get(provider, ("OPENAI_API_KEY", cls.OPENAI_API_KEY))

    @classmethod
    def missing_keys(cls, provider: str | None = None) -> list[str]:
        """Return missing-but-required API key names for the active config.

        Accounts for both the embedding provider and the (chat) LLM provider.
        """
        provider = (provider or cls.LLM_PROVIDER).lower()
        missing: list[str] = []
        for prov in (cls.EMBEDDING_PROVIDER, provider):
            name, value = cls._key_for_provider(prov)
            if not value:
                missing.append(name)
        # Preserve order while de-duplicating.
        return list(dict.fromkeys(missing))


# ----------------------------------------------------------------------
# Shared logger
# ----------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes timestamped lines to the console."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
