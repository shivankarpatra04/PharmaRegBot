"""Scope detection — keep PharmaRegBot on-topic.

Two-stage check, applied to the (English) query before retrieval:
  1. A fast keyword check — if any pharma/regulatory term is present, the query
     is in scope and no LLM call is made.
  2. Otherwise, a lightweight LLM classifier decides YES/NO.

The check fails open: if the classifier errors (e.g. transient API issue), the
query is allowed through and the retrieval similarity threshold acts as a
secondary guard.
"""

from __future__ import annotations

from src.generation.llm import chat
from src.utils.config import get_logger

logger = get_logger(__name__)

PHARMA_KEYWORDS = {
    "gmp", "good manufacturing", "ich", "q7", "q9", "q10", "q11", "q12",
    "fda", "who", "cdsco", "schedule m", "ema", "anvisa", "anda", "dmf",
    "api", "active pharmaceutical", "drug", "pharmaceutical", "pharma",
    "manufacturing", "regulatory", "regulation", "guideline", "guidance",
    "validation", "qualification", "calibration", "cleaning", "sop",
    "batch", "batch release", "documentation", "record", "specification",
    "quality", "quality control", "quality assurance", "qa", "qc",
    "stability", "impurity", "excipient", "sterile", "aseptic",
    "water system", "hvac", "deviation", "capa", "change control",
    "audit", "inspection", "raw material", "formulation", "dossier",
    "pharmacopoeia", "potency", "assay", "dissolution", "contamination",
    "annexure", "annex", "facility", "equipment",
    # A few Hindi terms in case scope is checked before translation.
    "दवा", "निर्माण", "सत्यापन", "गुणवत्ता", "सफाई", "नियामक",
}

_SCOPE_SYSTEM_PROMPT = (
    "You are a strict binary classifier for a pharmaceutical regulatory "
    "assistant. Decide whether the user's question relates to pharmaceutical "
    "manufacturing, regulatory affairs, GMP, ICH/FDA/WHO/CDSCO guidelines, "
    "Schedule M, SOPs, quality systems, or related topics. "
    "Respond with exactly one word: YES or NO."
)


def keyword_in_scope(query: str) -> bool:
    """Return True if the query contains an obvious pharma/regulatory term."""
    low = (query or "").lower()
    return any(keyword in low for keyword in PHARMA_KEYWORDS)


def is_in_scope(query: str, provider: str | None = None, model: str | None = None) -> bool:
    """Return True if the query is within the pharma/regulatory domain."""
    if not query or not query.strip():
        return False

    if keyword_in_scope(query):
        return True

    try:
        text, _ = chat(
            system_prompt=_SCOPE_SYSTEM_PROMPT,
            user_prompt=f"Question: {query}",
            provider=provider,
            model=model,
            max_tokens=5,
            temperature=0.0,
        )
        decision = "yes" in text.strip().lower()
        logger.info("LLM scope classifier -> %s", "in scope" if decision else "out of scope")
        return decision
    except Exception as exc:
        # Fail open: don't block a possibly-valid query on a transient error.
        logger.warning("Scope classifier failed (%s); allowing query through.", exc)
        return True
