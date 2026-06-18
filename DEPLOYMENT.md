# Deploying PharmaRegBot

PharmaRegBot is a **Streamlit** app. Streamlit needs a persistent, long-running
server (WebSockets) and writable disk for its ChromaDB index and uploaded files.

> **Why not Vercel?** Vercel only runs *stateless serverless functions* with an
> ephemeral, read-only runtime filesystem and short execution limits. Streamlit
> cannot run there. Use one of the platforms below instead.

The recommended (and free) host is **Streamlit Community Cloud**.

---

## Option A — Streamlit Community Cloud (recommended, free)

### 1. Prerequisites
- A GitHub account with this repo pushed (see the main README / steps below).
- API key(s) for your chosen LLM + embedding provider (NVIDIA, OpenAI, or Anthropic).
- A Supabase project (URL + `service_role` key) for user accounts.

### 2. Deploy
1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **"Create app"** → **"Deploy a public app from GitHub"**.
3. Select:
   - **Repository:** `shivankarpatra04/PharmaRegBot` (or your repo)
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Advanced settings"**:
   - **Python version:** select **3.11** (most compatible with the pinned wheels).
   - **Secrets:** paste the contents of `.streamlit/secrets.toml.example`,
     filling in your real values. (See that file for the full list.)
5. Click **Deploy**. First build takes a few minutes while dependencies install.

### 3. Notes / limitations on the free tier
- **Ephemeral filesystem:** `chroma_db/` and `data/documents/` reset on every
  restart or redeploy. Re-upload your documents after a restart. This is normal.
- Secrets set in the dashboard are exposed to the app as environment variables,
  so `src/utils/config.py` (which uses `os.getenv`) reads them with no code change.
- `packages.txt` installs `build-essential` so native wheels (chroma-hnswlib,
  pymupdf) can build if a prebuilt wheel isn't available.

---

## Option B — Render (persistent server, free web-service tier)

Render can run the included `Dockerfile`, and you can attach a **persistent disk**
(paid) so the ChromaDB index and uploads survive restarts.

1. Push this repo to GitHub.
2. On https://render.com → **New** → **Web Service** → connect the repo.
3. **Runtime:** Docker (it auto-detects the `Dockerfile`).
4. Set environment variables (same keys as `.env.example`).
5. (Optional) Add a **Disk** mounted at `/app/chroma_db` for persistence.
6. Deploy. Render gives you a public `*.onrender.com` URL.

---

## Option C — Hugging Face Spaces (free, native Streamlit)

1. Create a new **Space** → SDK: **Streamlit**.
2. Push this repo's contents to the Space's Git remote.
3. Add your keys under **Settings → Variables and secrets**.
4. The Space builds and serves automatically. Filesystem is ephemeral on free tier.

---

## Local run (for reference)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in your keys
streamlit run app.py
```
