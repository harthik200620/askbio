# DEPLOY.md — Ship AskBio to a Live Public URL

This walks you from a working local repo to a **public Streamlit URL** backed by **Qdrant Cloud** and **OpenAI**, and it's honest about the two non-obvious parts: the cloud app *queries* an index it does not build, and the BM25 index is a local file that has to ship with the app.

> Mental model: **you build the indexes locally → push vectors to Qdrant Cloud → commit the small BM25 pickle + corpus → Streamlit Cloud runs only fast read queries.** The hosted app never ingests or embeds the corpus itself.

---

## Keys checklist (create these first)

| Account | What you need | Where |
| ------- | ------------- | ----- |
| **OpenAI** | API key, with **~$5 credit** added (embeddings + answers + the ragas judge) | https://platform.openai.com/api-keys |
| **Qdrant Cloud** | A **free cluster**, then its **cluster URL** + **API key** | https://cloud.qdrant.io |
| **Hugging Face** | A **read token** (to download `MedRAG/pubmed` + `PubMedQA`) | https://huggingface.co/settings/tokens |
| **GitHub** | An account + a repo to hold the code | https://github.com |
| **Streamlit Community Cloud** | Sign in **with GitHub** (no separate key) | https://share.streamlit.io |

---

## Step 1 — Push the repo to GitHub

```bash
cd askbio   # or your repo root
git init
git add .
git commit -m "AskBio: biomedical RAG"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Confirm `.env` is **not** committed (it's gitignored — only `.env.example` should be in the repo). Your real keys go into Streamlit Secrets later, never into git.

---

## Step 2 — Build the index into Qdrant Cloud, LOCALLY

The deployed app **queries** Qdrant Cloud; it does **not** build it. So you populate the cloud cluster from your own machine first.

In your local `.env`, point at the cloud cluster (leave `QDRANT_LOCAL` unset/0) and use the OpenAI embedding backend so the deployed app's query embeddings match the stored ones:

```env
OPENAI_API_KEY=sk-...
QDRANT_URL=https://xxxxxxxx.cloud.qdrant.io:6333
QDRANT_API_KEY=...
HF_TOKEN=hf_...
EMBED_BACKEND=openai
LLM_BACKEND=openai
# IMPORTANT: pick a focused corpus size for the live demo (see Step 3)
CORPUS_SUBSET_SIZE=15000
```

Then build:

```bash
python ingest.py        # stream + clean PubMed → data/corpus.jsonl
python embed_index.py   # OpenAI embeddings → Qdrant Cloud + data/bm25.pkl  (resumable)
```

When this finishes, your Qdrant **Cloud** collection holds the vectors, and `data/bm25.pkl` + `data/corpus.jsonl` exist locally. (`embed_index.py` is resumable — if it's interrupted, re-run it and it continues from a bookmark.)

> The embedding backend **must** be the same in deploy as it was at index time. You embedded the corpus with OpenAI `text-embedding-3-small` @ 768d, so the deployed app must also set `EMBED_BACKEND=openai` — otherwise query vectors won't match the stored ones (and dimensions won't even agree).

---

## Step 3 — Handle the BM25 pickle (the important nuance)

AskBio's retrieval is hybrid, so the deployed app needs **both** indexes:

- **Dense / Qdrant** — lives in Qdrant Cloud, queried over the network. ✅ Already handled by Step 2.
- **BM25** — a **local pickle** (`data/bm25.pkl`) loaded from disk by `retrieve.py`. The hosted app can only read it if it **ships in the repo**.

There's no free managed BM25 service here, so the practical answer is: **use a focused, smaller corpus for the live demo** so the artifacts are small enough to commit. A **~10–25k-row** corpus keeps `corpus.jsonl` + `bm25.pkl` to a reasonable size; a full 100k index is large and not worth hosting for a free demo.

Two ways to get the pickle into the deployed app:

**Option A — commit the artifacts (simplest).** Un-ignore the data files for deploy and commit them:

```gitignore
# .gitignore — for deploy, allow the small demo index to be committed:
# data/corpus.jsonl     ← un-ignore (comment out / remove the ignore)
# data/bm25.pkl         ← un-ignore
data/qdrant_local/      ← keep ignoring (local on-disk DB, not used in cloud)
```

```bash
git add -f data/corpus.jsonl data/bm25.pkl
git commit -m "Add focused demo corpus + BM25 index for deploy"
git push
```

**Option B — rebuild BM25 on first load from a committed corpus.** Commit only `data/corpus.jsonl` (smaller than the pickle) and have the app build the BM25 index once on startup from it, cached with `st.cache_resource`. This keeps the repo lighter but adds a one-time build on the first cold start. (Option A is simpler; use B if the pickle is the size problem.)

> Be honest in interviews about this trade-off: hosting a full 100k hybrid index isn't free, so the **live demo deliberately uses a smaller, topic-focused corpus**. The architecture is identical to the full system — only the corpus size differs.

---

## Step 4 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and **sign in with GitHub**.
2. **New app** → pick your repo and the `main` branch.
3. **Main file path:** set it to where `app.py` lives:
   - `askbio/app.py` if the repo has an `askbio/` subfolder, **or**
   - `app.py` if the repo root *is* the `askbio` folder.
4. Make sure **`requirements.txt`** is at the repo root (or alongside the main file) so Streamlit installs the dependencies. Click **Deploy**.

First build takes a few minutes (it installs `torch`, `sentence-transformers`, etc.).

---

## Step 5 — Add Secrets (TOML) in the Streamlit Cloud UI

Open the app's **Settings → Secrets** and paste this TOML. These become environment variables that `config.py` reads via `python-dotenv`/`os.getenv`, so **no `.env` file is needed in the cloud**:

```toml
OPENAI_API_KEY = "sk-..."
QDRANT_URL = "https://xxxxxxxx.cloud.qdrant.io:6333"
QDRANT_API_KEY = "..."
EMBED_BACKEND = "openai"
LLM_BACKEND = "openai"
```

Notes:
- **Do not** set `QDRANT_LOCAL` (or set it to `0`) so the app connects to Qdrant **Cloud**, not an on-disk DB that doesn't exist on the host.
- `HF_TOKEN` is only needed for *downloading datasets*; the deployed app just queries an already-built index, so it's optional in cloud secrets (include it only if you also run `evaluate.py` somewhere that pulls PubMedQA).
- Save — Streamlit reboots the app with the new secrets. Then open the public URL and ask a question; you should get a grounded answer with clickable PubMed citations.

---

## Step 6 — Free-tier idle sleep (and a keep-alive)

Both free tiers sleep when idle, which can make a "live" demo look broken to a recruiter who clicks a cold link:

- **Qdrant Cloud (free):** an idle free cluster is **terminated after ~1 week** of inactivity — if that happens you'd need to recreate the cluster and re-run Step 2. Log in periodically (or run a query) to keep it alive.
- **Streamlit Community Cloud:** apps **go to sleep** after a period of inactivity and cold-start on the next visit (slow first load while it reinstalls/warms up).

**Mitigation — a keep-alive ping.** Schedule a lightweight periodic request (e.g. a GitHub Actions cron, an UptimeRobot monitor, or any scheduler) that hits the app's public URL — and ideally issues a trivial Qdrant query — every few days. That keeps both the Streamlit app warm and the Qdrant cluster from being reclaimed. Keep the interval modest so you stay within free-tier limits.

---

## Quick deploy checklist

- [ ] Repo pushed to GitHub; `.env` **not** committed (only `.env.example`).
- [ ] Local `.env` points at Qdrant Cloud with `EMBED_BACKEND=openai`; `CORPUS_SUBSET_SIZE` set to a focused value.
- [ ] `python ingest.py` then `python embed_index.py` run successfully → Qdrant Cloud populated, `bm25.pkl` built.
- [ ] `data/corpus.jsonl` (+ `data/bm25.pkl` for Option A) committed for the deploy.
- [ ] Streamlit app created from the repo; main file path = `askbio/app.py` (or `app.py`).
- [ ] Secrets TOML added (OpenAI + Qdrant + `EMBED_BACKEND="openai"` + `LLM_BACKEND="openai"`); `QDRANT_LOCAL` unset.
- [ ] Public URL returns a grounded, cited answer.
- [ ] Keep-alive ping scheduled.
