"""LLM access layer.

A single ``chat`` helper talks to either OpenAI or Anthropic (selected via
config or an explicit argument) and returns the response text plus token usage.
``generate_answer`` builds the grounded RAG prompt and calls ``chat``.

Clients are created lazily and cached, so a missing key only raises when that
provider is actually used. All callers are expected to wrap these functions in
try/except for user-friendly error reporting.
"""

from __future__ import annotations

from anthropic import Anthropic
from openai import OpenAI

from src.generation.prompt import build_user_prompt, get_system_prompt
from src.utils.config import Config, get_logger

logger = get_logger(__name__)

_openai_client: OpenAI | None = None
_nvidia_client: OpenAI | None = None
_anthropic_client: Anthropic | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not Config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file to use OpenAI."
            )
        _openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
    return _openai_client


def _get_nvidia_client() -> OpenAI:
    """NVIDIA's API catalog is OpenAI-compatible — reuse the SDK with its base URL."""
    global _nvidia_client
    if _nvidia_client is None:
        if not Config.NVIDIA_API_KEY:
            raise ValueError(
                "NVIDIA_API_KEY is not set. Add it to your .env file to use NVIDIA."
            )
        _nvidia_client = OpenAI(
            api_key=Config.NVIDIA_API_KEY, base_url=Config.NVIDIA_BASE_URL
        )
    return _nvidia_client


def _get_anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        if not Config.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file to use Anthropic."
            )
        _anthropic_client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    return _anthropic_client


def chat(
    system_prompt: str,
    user_prompt: str,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> tuple[str, dict]:
    """Send a single-turn chat request and return (text, usage).

    ``usage`` is a dict with ``input_tokens``, ``output_tokens`` and
    ``total_tokens``.
    """
    provider = (provider or Config.LLM_PROVIDER).lower()

    # OpenAI and NVIDIA share the OpenAI chat-completions API.
    if provider in Config.OPENAI_COMPATIBLE_PROVIDERS:
        client = _get_nvidia_client() if provider == "nvidia" else _get_openai_client()
        model = model or Config.default_model_for(provider)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        # Be defensive: some OpenAI-compatible servers may omit usage fields.
        u = response.usage
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(u, "completion_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
        }
        return text, usage

    if provider == "anthropic":
        client = _get_anthropic_client()
        model = model or Config.default_model_for("anthropic")
        response = client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        return text, usage

    raise ValueError(
        f"Unknown LLM provider: '{provider}'. Use 'openai', 'anthropic', or 'nvidia'."
    )


def generate_answer(
    question: str,
    chunks: list[dict],
    language: str = "en",
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    """Generate a grounded, cited answer from retrieved chunks.

    Returns ``{"answer": str, "usage": dict}``.
    """
    system_prompt = get_system_prompt(language)
    user_prompt = build_user_prompt(question, chunks, language)

    text, usage = chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider,
        model=model,
        max_tokens=1024,
        temperature=0.0,
    )
    return {"answer": text.strip(), "usage": usage}
