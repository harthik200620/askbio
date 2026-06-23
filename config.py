"""Central config. Everything imports settings from here; secrets come from .env.

Set EMBED_BACKEND=local, LLM_BACKEND=none, QDRANT_LOCAL=1 to run the whole thing
free/offline (see .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load the .env next to this file so streamlit/pytest/CLIs all agree regardless of cwd.
load_dotenv(Path(__file__).resolve().parent / ".env")

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CORPUS_PATH = DATA_DIR / "corpus.jsonl"             # ingest.py output
BM25_PATH = DATA_DIR / "bm25.pkl"                   # embed_index.py output
EMBED_PROGRESS_PATH = DATA_DIR / "embed_progress.json"  # resume bookmark for embedding
EVAL_RESULTS_PATH = DATA_DIR / "eval_results.csv"
EVAL_CHART_PATH = DATA_DIR / "eval_chart.png"

# Data sources
HF_CORPUS = "MedRAG/pubmed"          # pre-chunked PubMed snippets (text + PMID)
CORPUS_SUBSET_SIZE = int(os.getenv("CORPUS_SUBSET_SIZE", "100000"))

HF_EVAL = "qiaojin/PubMedQA"
EVAL_CONFIG = "pqa_labeled"          # expert-labeled split (yes/no/maybe)
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "50"))

# Embeddings. EMBED_BACKEND: "openai" (default) or "local" (free CPU).
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "openai").lower()
OPENAI_EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 768                      # 3-small truncated to 768
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_EMBED_DIM = 384
EMBED_BATCH_SIZE = 128


def embed_dim() -> int:
    # Qdrant needs the right vector size and the two backends differ.
    return EMBED_DIM if EMBED_BACKEND == "openai" else LOCAL_EMBED_DIM


# Vector store (Qdrant)
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "askbio_pubmed")
# QDRANT_LOCAL=1 -> on-disk Qdrant, no cloud account.
QDRANT_LOCAL = os.getenv("QDRANT_LOCAL", "0") == "1"
QDRANT_LOCAL_PATH = str(DATA_DIR / "qdrant_local")

# Retrieval
DENSE_TOP_K = 20      # from vector search
BM25_TOP_K = 20       # from keyword search
RRF_TOP_K = 20        # kept after fusion
RRF_K = 60            # RRF constant
RERANK_TOP_K = 5      # after cross-encoder rerank
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Generation. LLM_BACKEND: "openai" | "anthropic" | "gemini" | "none".
# gemini is free via Google AI Studio; none = extractive answer, no LLM call.
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower()
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
ANTHROPIC_LLM_MODEL = os.getenv("ANTHROPIC_LLM_MODEL", "claude-haiku-4-5")
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# accept either env var name for the Google key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2", "")
GEMINI_API_KEY_3 = os.getenv("GEMINI_API_KEY_3", "")
# all configured keys, in order, for round-robin when one hits its rate limit
GEMINI_API_KEYS = [k for k in (GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3) if k]

ABSTAIN_MESSAGE = "I don't have enough information in the literature to answer that."
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

# Topic focus + relevance threshold
# When set (space/comma-separated keywords), ingest keeps only snippets mentioning
# any keyword. Empty = just take the first CORPUS_SUBSET_SIZE.
CORPUS_TOPIC = os.getenv("CORPUS_TOPIC", "")
# Cap on raw rows to stream while hunting for topic matches. 0 = no cap.
CORPUS_SCAN_LIMIT = int(os.getenv("CORPUS_SCAN_LIMIT", "0"))

# Abstain if the best reranked passage scores below this. Off by default; demo .env raises it.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "-1e9"))


# Misc
HF_TOKEN = os.getenv("HF_TOKEN", "")
RANDOM_SEED = 42
