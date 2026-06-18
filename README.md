# PharmaRegBot 🧪 — Regulatory Document Intelligence System

PharmaRegBot is a bilingual (English + Hindi) Retrieval-Augmented Generation (RAG)
chatbot built for pharmaceutical **API manufacturing** and **regulatory affairs**
professionals. Upload regulatory documents — ICH guidelines, FDA guidance, WHO GMP,
CDSCO Schedule M, and internal SOPs — and ask natural-language questions. PharmaRegBot
retrieves the most relevant passages from *your* documents and generates concise,
**grounded, source-cited** answers. It never answers from outside knowledge, and it
politely refuses questions that fall outside the pharmaceutical regulatory domain.

---

## ✨ Features

- 🔐 **Multi-user accounts** — sign-up / login with hashed passwords stored in
  **Supabase** (Postgres); each user's documents are **private and isolated**.
- 📄 **Multi-format ingestion** — PDF, DOCX, TXT and MD files.
- 🔍 **Grounded RAG** — answers come only from your uploaded documents.
- 🧾 **Source citations** — every answer links back to the source file and page number.
- 🌐 **Bilingual** — ask in English *or* Hindi; the UI and answers follow your language.
- 🛡️ **Scope guardrails** — non-pharma questions are politely rejected.
- 🚫 **Anti-hallucination** — strict system prompt + similarity threshold; says
  "not available in the uploaded documents" when context is insufficient.
- 🧠 **Configurable LLM** — switch between **NVIDIA** (Llama / Nemotron / Mixtral),
  **OpenAI GPT-4o**, and **Anthropic Claude** from the UI or `.env`.
- 💾 **Persistent vector store** — local ChromaDB; documents stay indexed between runs.
- ♻️ **Deduplication** — identical files are detected by content hash and skipped.
- 📊 **Token usage counter** and a clean two-column Streamlit UI.
- 🐳 **Deployment-ready** — Dockerfile + pinned `requirements.txt`.

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/RAG-LangChain-1C3C3C?logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FFCD00)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI%20GPT--4o-412991?logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/LLM-Anthropic%20Claude-D97757?logo=anthropic&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

| Layer            | Technology                                              |
|------------------|---------------------------------------------------------|
| Language         | Python 3.10+                                            |
| UI               | Streamlit                                               |
| LLM              | NVIDIA / OpenAI GPT-4o / Anthropic Claude (configurable) |
| Embeddings       | NVIDIA `llama-3.2-nv-embedqa-1b-v2` **or** OpenAI `text-embedding-3-small` |
| Vector DB        | ChromaDB (local, persistent)                            |
| RAG framework    | LangChain                                               |
| Chunking         | `RecursiveCharacterTextSplitter` (1000 / 200, token-aware) |
| Translation      | deep-translator (Google) + langdetect                   |
| File parsing     | PyMuPDF (PDF), python-docx (DOCX), plain text (TXT/MD)   |
| Config / secrets | python-dotenv                                           |

---

## 📁 Project Structure

```
PharmaRegBot/
├── data/documents/<user_id>/    # uploaded docs, isolated per user
├── supabase_schema.sql          # one-time Supabase users-table setup
├── src/
│   ├── auth/
│   │   ├── database.py          # Supabase user store + PBKDF2 password hashing
│   │   └── authenticator.py     # Streamlit login / register flow
│   ├── ingestion/
│   │   ├── loader.py            # load PDF, DOCX, TXT, MD
│   │   ├── chunker.py           # recursive, token-aware splitting
│   │   └── embedder.py          # OpenAI embeddings
│   ├── retrieval/
│   │   ├── vector_store.py      # ChromaDB init / add / query / list / delete
│   │   └── retriever.py         # semantic search, top-5 + threshold
│   ├── generation/
│   │   ├── prompt.py            # English + Hindi system prompts
│   │   ├── llm.py               # OpenAI / Anthropic chat with context injection
│   │   └── scope_checker.py     # reject non-pharma queries
│   └── utils/
│       ├── translator.py        # Hindi detection + translation
│       └── config.py            # env loading + logging
├── app.py                       # Streamlit application
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Setup

### 1. Clone

```bash
git clone <your-repo-url> PharmaRegBot
cd PharmaRegBot
```

### 2. Create a virtual environment & install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure your environment

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Then edit `.env`. **NVIDIA-only setup (default)** — a single key powers both
generation and embeddings:

```env
NVIDIA_API_KEY=nvapi-...
LLM_PROVIDER=nvidia
LLM_MODEL=meta/llama-3.3-70b-instruct
EMBEDDING_PROVIDER=nvidia
NVIDIA_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-1b-v2
SIMILARITY_THRESHOLD=0.4
HINDI_RESPONSE_MODE=translate   # or "native"
```

Get a free NVIDIA API key from the [NVIDIA API Catalog](https://build.nvidia.com/).

**Other providers** — set `LLM_PROVIDER` to `openai` or `anthropic` and provide
the matching key (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`). The embedding provider
is independent: `EMBEDDING_PROVIDER=openai` uses `text-embedding-3-small` (needs
`OPENAI_API_KEY`).

> **Note:** The chosen embedding provider's key is always required (it indexes and
> searches your documents). Don't switch embedding providers/models after ingesting
> documents — re-process them, since different models produce incompatible vectors.

### 4. Set up Supabase (user accounts)

1. Create a free project at [supabase.com](https://supabase.com/).
2. In the dashboard, open **SQL Editor → New query**, paste the contents of
   [`supabase_schema.sql`](supabase_schema.sql), and **Run** it (creates the
   `users` table with Row Level Security).
3. In **Project Settings → API**, copy the **Project URL** and the
   **`service_role`** key into `.env`:

```env
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_KEY=your_service_role_key_here
SUPABASE_USERS_TABLE=users
```

> The `service_role` key is secret — it's only used server-side and is covered by
> `.gitignore` (it lives in `.env`).

### 5. Run

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

## 🐳 Run with Docker

```bash
docker build -t pharmaregbot .
docker run -p 8501:8501 --env-file .env \
  -v "$(pwd)/chroma_db:/app/chroma_db" \
  -v "$(pwd)/data/documents:/app/data/documents" \
  pharmaregbot
```

---

## 💡 Example Questions

**English**
1. "What are the cleaning validation requirements as per ICH Q7?"
2. "What does Schedule M say about water system qualification?"
3. "List documentation requirements for API batch release."
4. "What is the requirement for change control in GMP?"

**Hindi (हिंदी)**
5. "आईसीएच Q7 के अनुसार सफाई सत्यापन की क्या आवश्यकताएं हैं?"

*(Tip: upload the corresponding ICH Q7 / Schedule M / SOP documents first so the
answers are grounded in real content.)*

---

## 🧠 How It Works

1. **Ingest** — files are parsed page-by-page, split into ~1000-token chunks (200
   overlap), embedded with `text-embedding-3-small`, and stored in ChromaDB with
   metadata `{source_file, page_number, doc_type, chunk_index, file_hash}`.
2. **Query** — your question is language-detected; Hindi queries are translated to
   English. A scope check (keywords + lightweight LLM classifier) blocks off-topic
   questions.
3. **Retrieve** — the query is embedded and the top-5 most similar chunks (cosine
   similarity ≥ `SIMILARITY_THRESHOLD`) are fetched. If nothing clears the threshold,
   the bot replies *"No relevant information found in uploaded documents."*
4. **Generate** — the retrieved context + a strict system prompt are sent to the LLM,
   which answers only from that context and cites sources. Hindi answers are produced
   per `HINDI_RESPONSE_MODE` (translate-back by default, or native Hindi generation).

---

## 🖼️ Screenshot

> _Screenshot placeholder — add `docs/screenshot.png` and reference it here._
>
> `![PharmaRegBot UI](docs/screenshot.png)`

---

## 👥 Multi-user & data isolation

PharmaRegBot is built for teams of pharma professionals — each person gets their
own private workspace:

- **Accounts** are stored in a **Supabase** Postgres `users` table. Passwords are
  hashed with **PBKDF2-HMAC-SHA256** + a per-user random salt (never plaintext).
  The table is protected by Row Level Security; the app uses the `service_role`
  key server-side.
- **Register / Login** on the landing screen; the app is fully gated behind auth.
- **Isolation** — every document chunk is tagged with its owner's `user_id`, and
  every search, listing, and deletion is filtered by it. You only ever see, query,
  and answer from **your own** documents. Two users can even upload the same file
  independently without collision.
- **Persistence & ownership** — once you process a document it stays indexed
  (ChromaDB is persistent) until **you** delete it from the sidebar, which removes
  both its vectors and the stored file.

> **Note:** Login state lasts for the browser session (re-login after a refresh).
> For cross-session "remember me" cookies, add a cookie component — a natural next
> step. Documents uploaded *before* auth was added won't be attributed to any
> account; re-upload them after logging in (or clear `chroma_db/` for a clean start).

## 🔐 Security

- All API keys are read from `.env` via `python-dotenv` — **never hardcoded**.
- `.env`, `chroma_db/` and `data/documents/` are git-ignored.
- LLM and embedding calls are wrapped in error handling with user-friendly messages.

---

## 📜 License

Released under the **MIT License**. See [LICENSE](LICENSE) for details.
