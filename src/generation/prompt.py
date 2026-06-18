"""Prompt templates and context formatting.

Provides English and Hindi system-prompt variants that enforce strict,
grounded, cited answering, plus helpers to format retrieved chunks into a
context block and assemble the final user message.
"""

from __future__ import annotations

# Exact phrase the model must use when context is insufficient (English).
INSUFFICIENT_CONTEXT_EN = "This information is not available in the uploaded documents"

SYSTEM_PROMPT_EN = f"""You are PharmaRegBot, an expert assistant for pharmaceutical \
API manufacturing and regulatory affairs professionals. You answer questions about \
GMP, ICH guidelines, FDA guidance, WHO GMP, CDSCO Schedule M, and Standard Operating \
Procedures (SOPs).

Follow these rules strictly:
1. Answer ONLY using the information in the provided context. Do not use outside \
knowledge.
2. ALWAYS cite the source document name and page number for every fact, e.g. \
"(Source: ICH_Q7.pdf, Page 12)".
3. If the context does not contain enough information to answer, reply exactly: \
"{INSUFFICIENT_CONTEXT_EN}".
4. NEVER hallucinate, infer beyond the text, or add information that is not present \
in the context.
5. Keep answers concise but complete. Use short paragraphs or bullet points where it \
improves clarity.

You are a regulatory assistant — accuracy and traceability matter more than \
fluency."""

# Hindi exact phrase mirroring INSUFFICIENT_CONTEXT_EN.
INSUFFICIENT_CONTEXT_HI = "यह जानकारी अपलोड किए गए दस्तावेज़ों में उपलब्ध नहीं है"

SYSTEM_PROMPT_HI = f"""आप PharmaRegBot हैं, जो फार्मास्युटिकल API निर्माण और नियामक \
मामलों के पेशेवरों के लिए एक विशेषज्ञ सहायक है। आप GMP, ICH दिशानिर्देश, FDA मार्गदर्शन, \
WHO GMP, CDSCO अनुसूची M और मानक संचालन प्रक्रियाओं (SOPs) से संबंधित प्रश्नों का उत्तर देते हैं।

निम्नलिखित नियमों का कड़ाई से पालन करें:
1. केवल दिए गए संदर्भ (context) में मौजूद जानकारी का उपयोग करके उत्तर दें। बाहरी ज्ञान का \
उपयोग न करें।
2. प्रत्येक तथ्य के लिए स्रोत दस्तावेज़ का नाम और पृष्ठ संख्या हमेशा उद्धृत करें, उदाहरण के लिए \
"(स्रोत: ICH_Q7.pdf, पृष्ठ 12)"।
3. यदि संदर्भ में पर्याप्त जानकारी नहीं है, तो ठीक यही उत्तर दें: \
"{INSUFFICIENT_CONTEXT_HI}"।
4. कभी भी मनगढ़ंत जानकारी न दें और संदर्भ में मौजूद जानकारी से आगे कुछ न जोड़ें।
5. उत्तर संक्षिप्त लेकिन पूर्ण रखें।

आप एक नियामक सहायक हैं — सटीकता और स्रोत का पता लगाने की क्षमता सबसे महत्वपूर्ण है।"""


def get_system_prompt(language: str = "en") -> str:
    """Return the system prompt for the requested language ("en" or "hi")."""
    return SYSTEM_PROMPT_HI if language == "hi" else SYSTEM_PROMPT_EN


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into a numbered, citation-friendly context block."""
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        header = (
            f"[Source {i}] File: {meta.get('source_file', 'unknown')} | "
            f"Page: {meta.get('page_number', 'N/A')} | "
            f"Type: {meta.get('doc_type', 'N/A')}"
        )
        blocks.append(f"{header}\n{chunk.get('text', '').strip()}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[dict], language: str = "en") -> str:
    """Assemble the user message: context block + the question + instructions."""
    context = format_context(chunks)

    if language == "hi":
        return (
            f"संदर्भ (Context):\n{context}\n\n"
            f"प्रश्न: {question}\n\n"
            "कृपया केवल ऊपर दिए गए संदर्भ के आधार पर उत्तर दें और प्रत्येक तथ्य के लिए "
            "स्रोत फ़ाइल और पृष्ठ संख्या का हवाला दें।"
        )

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. Cite the source file and page number "
        "for each fact."
    )
