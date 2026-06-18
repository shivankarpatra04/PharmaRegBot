"""Bilingual (English <-> Hindi) helpers.

Responsibilities:
  * detect whether a query is written in Hindi
  * translate a Hindi query into English so the RAG pipeline (which runs over
    English regulatory documents) can process it
  * translate an English answer back into Hindi

Translation uses deep-translator's Google backend. Language detection uses a
fast Devanagari character check first, falling back to langdetect.

Every function degrades gracefully: if detection or translation fails for any
reason, the original text is returned and the error is logged rather than
crashing the pipeline.
"""

from __future__ import annotations

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

from deep_translator import GoogleTranslator

from src.utils.config import get_logger

# Make langdetect deterministic across runs.
DetectorFactory.seed = 0

logger = get_logger(__name__)

# Google Translate rejects very long single requests; stay well under the limit.
_MAX_CHARS = 4500


def _contains_devanagari(text: str) -> bool:
    """Return True if the text contains any Devanagari (Hindi) character."""
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def detect_language(text: str) -> str:
    """Return an ISO language code ("hi", "en", ...) for the given text."""
    if not text or not text.strip():
        return "en"
    if _contains_devanagari(text):
        return "hi"
    try:
        return detect(text)
    except LangDetectException:
        return "en"
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Language detection failed: %s", exc)
        return "en"


def is_hindi(text: str) -> bool:
    """Convenience wrapper: True when the text is (predominantly) Hindi."""
    if not text:
        return False
    if _contains_devanagari(text):
        return True
    return detect_language(text) == "hi"


def _split_for_translation(text: str) -> list[str]:
    """Split long text into <= _MAX_CHARS pieces on paragraph/sentence breaks."""
    if len(text) <= _MAX_CHARS:
        return [text]

    pieces: list[str] = []
    current = ""
    # Split on newlines first, then fall back to sentence-ish boundaries.
    for segment in text.replace(". ", ".\n").split("\n"):
        if len(current) + len(segment) + 1 > _MAX_CHARS:
            if current:
                pieces.append(current)
            # A single oversized segment is hard-split.
            while len(segment) > _MAX_CHARS:
                pieces.append(segment[:_MAX_CHARS])
                segment = segment[_MAX_CHARS:]
            current = segment
        else:
            current = f"{current}\n{segment}" if current else segment
    if current:
        pieces.append(current)
    return pieces


def _translate(text: str, target: str, source: str = "auto") -> str:
    """Translate text to the target language, chunking long inputs."""
    if not text or not text.strip():
        return text
    try:
        translator = GoogleTranslator(source=source, target=target)
        parts = _split_for_translation(text)
        translated = [translator.translate(part) or part for part in parts]
        return "\n".join(translated)
    except Exception as exc:
        logger.warning("Translation to '%s' failed: %s", target, exc)
        return text  # fail open — return original text


def translate_to_english(text: str) -> str:
    """Translate arbitrary (likely Hindi) text into English."""
    return _translate(text, target="en", source="auto")


def translate_to_hindi(text: str) -> str:
    """Translate English text into Hindi."""
    return _translate(text, target="hi", source="en")
