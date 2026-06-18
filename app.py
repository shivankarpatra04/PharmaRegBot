"""PharmaRegBot — Streamlit application.

A bilingual (English + Hindi) RAG chatbot for pharmaceutical API manufacturing
and regulatory professionals. Upload regulatory documents, then ask grounded,
cited questions answered strictly from those documents.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import os

import streamlit as st

from src.auth.authenticator import current_user, ensure_db_initialised, logout, render_auth_ui
from src.generation.llm import generate_answer
from src.generation.scope_checker import is_in_scope
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import embed_texts
from src.ingestion.loader import SUPPORTED_EXTENSIONS, load_document
from src.retrieval.retriever import retrieve
from src.retrieval.vector_store import VectorStore
from src.utils.config import Config, get_logger
from src.utils.translator import is_hindi, translate_to_english, translate_to_hindi

logger = get_logger("pharmaregbot.app")

# ----------------------------------------------------------------------
# Localised UI strings
# ----------------------------------------------------------------------
UI_TEXT = {
    "en": {
        "page_title": "PharmaRegBot",
        "title": "PharmaRegBot 🧪",
        "subtitle": "Regulatory Document Intelligence System",
        "upload_label": "Upload regulatory documents",
        "upload_help": "PDF, DOCX, TXT or MD. Multiple files allowed.",
        "process_btn": "⚙️ Process Documents",
        "indexed_header": "📚 Indexed Documents",
        "no_docs_indexed": "No documents indexed yet.",
        "delete": "Delete",
        "language_label": "Language",
        "provider_label": "LLM Provider",
        "model_label": "LLM Model",
        "settings_header": "⚙️ Settings",
        "account_header": "👤 Account",
        "logged_in_as": "Signed in as **{name}**",
        "logout": "🚪 Log out",
        "profile": "📋 My Profile",
        "username_label": "Username",
        "fullname_label": "Full name",
        "email_label": "Email",
        "member_since": "Member since",
        "not_provided": "—",
        "chat_header": "💬 Ask a question",
        "chat_placeholder": "Ask about GMP, ICH, FDA, Schedule M, SOPs...",
        "clear_chat": "🗑️ Clear Chat",
        "sources": "📄 Sources",
        "thinking": "Searching documents and generating answer...",
        "processing": "Processing documents...",
        "done": "✅ Processing complete.",
        "already_indexed": "already indexed — skipped",
        "indexed_ok": "indexed",
        "no_text": "no extractable text — skipped",
        "token_usage": "🔢 Tokens used this session",
        "chunks": "chunks",
        "score": "score",
        "page": "Page",
        "welcome": (
            "👋 **Welcome to PharmaRegBot.** Upload your regulatory documents in "
            "the sidebar, click **Process Documents**, then ask a question below."
        ),
        "missing_keys": (
            "⚠️ Missing required API key(s): {keys}. Add them to your `.env` file."
        ),
        "no_docs_warning": "Please upload and process documents before asking questions.",
    },
    "hi": {
        "page_title": "PharmaRegBot",
        "title": "PharmaRegBot 🧪",
        "subtitle": "नियामक दस्तावेज़ बुद्धिमत्ता प्रणाली",
        "upload_label": "नियामक दस्तावेज़ अपलोड करें",
        "upload_help": "PDF, DOCX, TXT या MD। एक से अधिक फ़ाइलें स्वीकार्य हैं।",
        "process_btn": "⚙️ दस्तावेज़ संसाधित करें",
        "indexed_header": "📚 अनुक्रमित दस्तावेज़",
        "no_docs_indexed": "अभी तक कोई दस्तावेज़ अनुक्रमित नहीं है।",
        "delete": "हटाएँ",
        "language_label": "भाषा",
        "provider_label": "LLM प्रदाता",
        "model_label": "LLM मॉडल",
        "settings_header": "⚙️ सेटिंग्स",
        "account_header": "👤 खाता",
        "logged_in_as": "**{name}** के रूप में साइन इन",
        "logout": "🚪 लॉग आउट",
        "profile": "📋 मेरी प्रोफ़ाइल",
        "username_label": "उपयोगकर्ता नाम",
        "fullname_label": "पूरा नाम",
        "email_label": "ईमेल",
        "member_since": "सदस्य बने",
        "not_provided": "—",
        "chat_header": "💬 प्रश्न पूछें",
        "chat_placeholder": "GMP, ICH, FDA, अनुसूची M, SOPs के बारे में पूछें...",
        "clear_chat": "🗑️ चैट साफ़ करें",
        "sources": "📄 स्रोत",
        "thinking": "दस्तावेज़ खोजे जा रहे हैं और उत्तर तैयार किया जा रहा है...",
        "processing": "दस्तावेज़ संसाधित किए जा रहे हैं...",
        "done": "✅ संसाधन पूर्ण हुआ।",
        "already_indexed": "पहले से अनुक्रमित — छोड़ा गया",
        "indexed_ok": "अनुक्रमित",
        "no_text": "कोई पाठ नहीं मिला — छोड़ा गया",
        "token_usage": "🔢 इस सत्र में उपयोग किए गए टोकन",
        "chunks": "खंड",
        "score": "स्कोर",
        "page": "पृष्ठ",
        "welcome": (
            "👋 **PharmaRegBot में आपका स्वागत है।** साइडबार में अपने नियामक दस्तावेज़ "
            "अपलोड करें, **दस्तावेज़ संसाधित करें** पर क्लिक करें, फिर नीचे प्रश्न पूछें।"
        ),
        "missing_keys": (
            "⚠️ आवश्यक API कुंजी अनुपलब्ध है: {keys}. इसे अपनी `.env` फ़ाइल में जोड़ें।"
        ),
        "no_docs_warning": "प्रश्न पूछने से पहले कृपया दस्तावेज़ अपलोड करें और संसाधित करें।",
    },
}

# Canned answers (deterministic translations, not machine-translated at runtime).
MESSAGES = {
    "en": {
        "out_of_scope": (
            "This question is outside the scope of pharmaceutical regulatory "
            "documents. Please ask questions related to GMP, ICH guidelines, FDA "
            "regulations, SOPs, or manufacturing procedures."
        ),
        "no_info": "No relevant information found in uploaded documents",
        "no_docs": "Please upload and process documents before asking questions.",
        "error": "An error occurred while processing your question: {error}",
    },
    "hi": {
        "out_of_scope": (
            "यह प्रश्न फार्मास्युटिकल नियामक दस्तावेज़ों के दायरे से बाहर है। कृपया GMP, "
            "ICH दिशानिर्देश, FDA नियम, SOPs, या निर्माण प्रक्रियाओं से संबंधित प्रश्न पूछें।"
        ),
        "no_info": "अपलोड किए गए दस्तावेज़ों में कोई प्रासंगिक जानकारी नहीं मिली",
        "no_docs": "प्रश्न पूछने से पहले कृपया दस्तावेज़ अपलोड करें और संसाधित करें।",
        "error": "आपके प्रश्न को संसाधित करते समय एक त्रुटि हुई: {error}",
    },
}


# ----------------------------------------------------------------------
# Cached resources / helpers
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStore:
    """Create (once) and reuse the persistent vector store across reruns."""
    return VectorStore(Config.CHROMA_DB_PATH, Config.COLLECTION_NAME)


def file_hash(data: bytes) -> str:
    """Content hash used to avoid re-ingesting identical documents."""
    return hashlib.md5(data).hexdigest()


def user_documents_dir(user_id) -> str:
    """Return (and create) the per-user document storage directory."""
    path = os.path.join(Config.DOCUMENTS_PATH, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


def save_uploaded_file(data: bytes, filename: str, user_id) -> str:
    """Persist an uploaded file under the user's folder and return its path."""
    path = os.path.join(user_documents_dir(user_id), filename)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def delete_document_file(user_id, filename: str) -> None:
    """Best-effort removal of a user's document file from disk."""
    path = os.path.join(user_documents_dir(user_id), filename)
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:  # pragma: no cover - non-fatal
        logger.warning("Could not delete file %s: %s", path, exc)


def process_uploaded_files(uploaded_files, vector_store: VectorStore, ui: dict, user_id) -> None:
    """Ingest a user's uploaded files with a progress bar, skipping duplicates."""
    total = len(uploaded_files)
    progress = st.progress(0.0, text=ui["processing"])
    status = st.empty()

    for index, uploaded in enumerate(uploaded_files):
        data = uploaded.getvalue()
        digest = file_hash(data)
        name = uploaded.name

        try:
            if vector_store.file_exists(user_id, digest):
                status.info(f"📄 {name} — {ui['already_indexed']}")
            else:
                path = save_uploaded_file(data, name, user_id)
                document = load_document(path)
                chunks = chunk_document(document, digest)

                if not chunks:
                    status.warning(f"📄 {name} — {ui['no_text']}")
                else:
                    embeddings = embed_texts([c["text"] for c in chunks])
                    vector_store.add(user_id, chunks, embeddings)
                    status.success(
                        f"📄 {name} — {len(chunks)} {ui['chunks']} {ui['indexed_ok']}"
                    )
        except Exception as exc:  # noqa: BLE001 - surface a friendly message
            logger.exception("Failed to process %s", name)
            status.error(f"📄 {name} — error: {exc}")

        progress.progress((index + 1) / total, text=ui["processing"])

    progress.empty()
    st.toast(ui["done"], icon="✅")


def process_query(
    query: str,
    vector_store: VectorStore,
    provider: str,
    model: str,
    user_id,
) -> dict:
    """Run the full bilingual RAG pipeline for a single question (user-scoped).

    Returns ``{"answer": str, "sources": list, "usage": dict}``.
    """
    respond_lang = "hi" if is_hindi(query) else "en"
    english_query = translate_to_english(query) if respond_lang == "hi" else query

    empty_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # No documents for this user -> ask them to upload first.
    if not vector_store.has_documents(user_id):
        return {
            "answer": MESSAGES[respond_lang]["no_docs"],
            "sources": [],
            "usage": empty_usage,
        }

    # Scope check (on the English form of the query).
    if not is_in_scope(english_query, provider=provider, model=model):
        return {
            "answer": MESSAGES[respond_lang]["out_of_scope"],
            "sources": [],
            "usage": empty_usage,
        }

    # Retrieve relevant context (scoped to this user's documents).
    chunks = retrieve(english_query, vector_store, user_id)
    if not chunks:
        return {
            "answer": MESSAGES[respond_lang]["no_info"],
            "sources": [],
            "usage": empty_usage,
        }

    # Generate the grounded answer.
    if respond_lang == "hi" and Config.HINDI_RESPONSE_MODE == "native":
        # Generate directly in Hindi using the Hindi system prompt.
        result = generate_answer(
            english_query, chunks, language="hi", provider=provider, model=model
        )
        answer = result["answer"]
    else:
        # Spec default: generate in English, then translate back to Hindi.
        result = generate_answer(
            english_query, chunks, language="en", provider=provider, model=model
        )
        answer = result["answer"]
        if respond_lang == "hi":
            answer = translate_to_hindi(answer)

    return {"answer": answer, "sources": chunks, "usage": result["usage"]}


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
def render_sidebar(vector_store: VectorStore, user: dict) -> tuple[dict, str, str]:
    """Render the sidebar and return (ui_text, provider, model)."""
    user_id = user["id"]
    # Language toggle drives all UI labels.
    with st.sidebar:
        lang_choice = st.radio(
            "🌐 Language / भाषा",
            options=["English", "हिंदी"],
            horizontal=True,
            key="ui_language",
        )
        ui_lang = "hi" if lang_choice == "हिंदी" else "en"
        ui = UI_TEXT[ui_lang]

        st.title(ui["title"])
        st.caption(ui["subtitle"])

        # --- Account ---
        display_name = user.get("full_name") or user["username"]
        st.markdown(ui["logged_in_as"].format(name=display_name))

        na = ui["not_provided"]
        with st.expander(ui["profile"]):
            joined = (user.get("created_at") or "")[:10] or na
            st.markdown(
                f"- **{ui['username_label']}:** {user.get('username', na)}\n"
                f"- **{ui['fullname_label']}:** {user.get('full_name') or na}\n"
                f"- **{ui['email_label']}:** {user.get('email') or na}\n"
                f"- **{ui['member_since']}:** {joined}"
            )

        if st.button(ui["logout"], use_container_width=True):
            logout()
            st.rerun()

        st.divider()

        # --- File upload + processing ---
        accepted = [ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)]
        uploaded_files = st.file_uploader(
            ui["upload_label"],
            type=accepted,
            accept_multiple_files=True,
            help=ui["upload_help"],
        )
        if st.button(ui["process_btn"], use_container_width=True, type="primary"):
            if uploaded_files:
                process_uploaded_files(uploaded_files, vector_store, ui, user_id)
            else:
                st.warning(ui["no_docs_warning"])

        st.divider()

        # --- Indexed documents with delete option (this user's documents only) ---
        st.subheader(ui["indexed_header"])
        documents = vector_store.list_documents(user_id)
        if not documents:
            st.caption(ui["no_docs_indexed"])
        else:
            for doc in documents:
                col_info, col_del = st.columns([4, 1])
                with col_info:
                    st.markdown(
                        f"**{doc['source_file']}**  \n"
                        f"`{doc['doc_type']}` · {doc['chunks']} {ui['chunks']}"
                    )
                with col_del:
                    if st.button(
                        "🗑️",
                        key=f"del_{doc['source_file']}",
                        help=ui["delete"],
                    ):
                        vector_store.delete_document(user_id, doc["source_file"])
                        delete_document_file(user_id, doc["source_file"])
                        st.rerun()

        st.divider()

        # --- Settings: provider + model ---
        st.subheader(ui["settings_header"])
        providers = list(Config.MODEL_OPTIONS.keys())
        default_provider_index = (
            providers.index(Config.LLM_PROVIDER)
            if Config.LLM_PROVIDER in providers
            else 0
        )
        provider = st.selectbox(
            ui["provider_label"],
            options=providers,
            index=default_provider_index,
        )

        model_options = Config.MODEL_OPTIONS[provider]
        default_model = Config.default_model_for(provider)
        model_index = (
            model_options.index(default_model)
            if default_model in model_options
            else 0
        )
        model = st.selectbox(
            ui["model_label"],
            options=model_options,
            index=model_index,
        )

        st.divider()

        # --- Token usage counter ---
        st.metric(ui["token_usage"], st.session_state.get("total_tokens", 0))

        # --- API key warnings (covers both embedding + chat providers) ---
        missing = Config.missing_keys(provider)
        if missing:
            st.error(ui["missing_keys"].format(keys=", ".join(missing)))

    return ui, provider, model


# ----------------------------------------------------------------------
# Main chat area
# ----------------------------------------------------------------------
def render_chat(
    ui: dict, vector_store: VectorStore, provider: str, model: str, user: dict
) -> None:
    """Render the chat header, history, sources and input box."""
    header_col, clear_col = st.columns([4, 1])
    with header_col:
        st.subheader(ui["chat_header"])
    with clear_col:
        if st.button(ui["clear_chat"], use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.rerun()

    if not st.session_state.messages:
        st.info(ui["welcome"])

    # Render conversation history.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander(ui["sources"]):
                    for source in message["sources"]:
                        meta = source["metadata"]
                        st.markdown(
                            f"**{meta.get('source_file', 'unknown')}** — "
                            f"{ui['page']} {meta.get('page_number', 'N/A')} — "
                            f"`{meta.get('doc_type', '')}` "
                            f"({ui['score']}: {source.get('score', 0):.2f})"
                        )
                        snippet = source["text"].strip()
                        if len(snippet) > 800:
                            snippet = snippet[:800] + "…"
                        st.caption(snippet)
                        st.divider()

    # Input box (anchored at the bottom by Streamlit).
    prompt = st.chat_input(ui["chat_placeholder"])
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(ui["thinking"]):
                try:
                    result = process_query(
                        prompt, vector_store, provider, model, user["id"]
                    )
                except Exception as exc:  # noqa: BLE001 - friendly error surface
                    logger.exception("Query processing failed")
                    lang = "hi" if is_hindi(prompt) else "en"
                    result = {
                        "answer": MESSAGES[lang]["error"].format(error=exc),
                        "sources": [],
                        "usage": {"total_tokens": 0},
                    }

        st.session_state.total_tokens += result.get("usage", {}).get("total_tokens", 0)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            }
        )
        st.rerun()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="PharmaRegBot",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Session state defaults.
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("total_tokens", 0)

    # Verify the Supabase user backend is reachable, then gate behind auth.
    try:
        ensure_db_initialised()
    except Exception as exc:  # noqa: BLE001 - friendly configuration error
        st.error(f"⚠️ Authentication backend unavailable.\n\n{exc}")
        st.stop()

    user = current_user()
    if not user:
        render_auth_ui()
        st.stop()

    vector_store = get_vector_store()

    ui, provider, model = render_sidebar(vector_store, user)
    render_chat(ui, vector_store, provider, model, user)


if __name__ == "__main__":
    main()
