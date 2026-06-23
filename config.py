"""
AskBio - central configuration.

Every module imports its settings from here so the whole pipeline shares ONE
source of truth: model names, vector size, collection name, retrieval sizes and
file paths. Secrets are read from environment variables (loaded from .env) and
are NEVER hard-coded.

Free local-test switches (so you can run the real pipeline at $0 before adding
paid keys) are documented in .env.example:
    EMBED_BACKEND=local   LLM_BACKEND=none   QDRANT_LOCAL=1   CORPUS_SUBSET_SIZE=200
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # read .env into the environment if the file exists

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CORPUS_PATH = DATA_DIR / "corpus.jsonl"             # cleaned snippets (ingest.py output)
BM25_PATH = DATA_DIR / "bm25.pkl"                   # saved BM25 index (embed_index.py output)
EMBED_PROGRESS_PATH = DATA_DIR / "embed_progress.json"  # resumable-embedding bookmark
EVAL_RESULTS_PATH = DATA_DIR / "eval_results.csv"   # eval table (evaluate.py output)
EVAL_CHART_PATH = DATA_DIR / "eval_chart.png"       # eval bar chart (evaluate.py output)

# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #
HF_CORPUS = "MedRAG/pubmed"          # pre-chunked PubMed snippets (text + PMID)
CORPUS_SUBSET_SIZE = int(os.getenv("CORPUS_SUBSET_SIZE", "100000"))

HF_EVAL = "qiaojin/PubMedQA"         # expert Q&A used for evaluation
EVAL_CONFIG = "pqa_labeled"          # the expert-labeled split (yes/no/maybe)
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "50"))

# --------------------------------------------------------------------------- #
# Embeddings   (EMBED_BACKEND = "openai" [spec default] | "local" [free, CPU])
# --------------------------------------------------------------------------- #
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "openai").lower()
OPENAI_EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 768                      # text-embedding-3-small truncated to 768 dims
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, free, CPU
LOCAL_EMBED_DIM = 384
EMBED_BATCH_SIZE = 128


def embed_dim() -> int:
    """Vector size for the active embedding backend (keeps Qdrant in sync)."""
    return EMBED_DIM if EMBED_BACKEND == "openai" else LOCAL_EMBED_DIM


# --------------------------------------------------------------------------- #
# Vector store (Qdrant)
# --------------------------------------------------------------------------- #
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "askbio_pubmed")
# Set QDRANT_LOCAL=1 to use an on-disk Qdrant (no cloud account needed) for testing.
QDRANT_LOCAL = os.getenv("QDRANT_LOCAL", "0") == "1"
QDRANT_LOCAL_PATH = str(DATA_DIR / "qdrant_local")

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
DENSE_TOP_K = 20      # candidates pulled from vector (dense) search
BM25_TOP_K = 20       # candidates pulled from keyword (BM25) search
RRF_TOP_K = 20        # kept after fusing the two ranked lists
RRF_K = 60            # Reciprocal Rank Fusion constant (standard default)
RERANK_TOP_K = 5      # final passages after cross-encoder reranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --------------------------------------------------------------------------- #
# Generation   (LLM_BACKEND = "openai" [spec default] | "anthropic" | "none")
#   "none" => extractive demo answer with NO LLM call (free, for testing)
# --------------------------------------------------------------------------- #
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower()
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
ANTHROPIC_LLM_MODEL = os.getenv("ANTHROPIC_LLM_MODEL", "claude-haiku-4-5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

ABSTAIN_MESSAGE = "I don't have enough information in the literature to answer that."
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
HF_TOKEN = os.getenv("HF_TOKEN", "")
RANDOM_SEED = 42
