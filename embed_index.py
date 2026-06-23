"""
AskBio - Phase 2: build the search indexes.

This module turns the cleaned PubMed corpus (produced by ingest.py) into the two
indexes the retriever needs:

1. A DENSE index in Qdrant. Every snippet's text is turned into a vector (an
   "embedding") so we can find passages by *meaning*, not just keywords. The
   embeddings come from one of two backends, chosen in config:
       - "openai" -> OpenAI's text-embedding-3-small (768 dims), a paid API.
       - "local"  -> a small sentence-transformers model (384 dims) that runs
                     free on your CPU.
2. A KEYWORD index (BM25) saved to disk. BM25 is a classic word-overlap ranker;
   it catches exact terms (gene names, acronyms) that embeddings sometimes miss.

Embedding a big corpus is slow, so build_index() is RESUMABLE: it writes a small
bookmark file recording how many rows are already in Qdrant, and on a re-run it
skips those and continues where it left off.

Heavy / optional libraries (sentence_transformers, openai, qdrant_client,
rank_bm25) are imported LAZILY inside the functions that use them. That keeps this
module importable - and its pure helpers unit-testable - even when those big
libraries are not installed.
"""
from __future__ import annotations

import json
import pickle
import re
from typing import Iterable, Iterator, List

import config

# A module-level cache for the local embedding model. Loading a
# sentence-transformers model is expensive, so we load it once and reuse it.
_LOCAL_MODEL = None


# --------------------------------------------------------------------------- #
# Pure helpers (no heavy imports - safe to unit-test on their own)
# --------------------------------------------------------------------------- #
# Matches runs of non-alphanumeric characters; used as the token separator.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase `text` and split it into alphanumeric tokens for BM25.

    Splitting on every run of non-alphanumeric characters means punctuation,
    whitespace and symbols all act as separators. Empty tokens (which appear at
    the string ends after a leading/trailing separator) are dropped.
    """
    if not text:
        return []
    lowered = text.lower()
    # re.split can yield "" at the boundaries; keep only the real tokens.
    return [tok for tok in _NON_ALNUM.split(lowered) if tok]


def _point_id(snippet_id: str) -> int:
    """Map a snippet's string id to a STABLE non-negative integer for Qdrant.

    Qdrant point ids must be ints (or UUIDs). Python's built-in hash() is salted
    per process, so it is NOT stable across runs - that would break resuming and
    re-indexing. We use a small deterministic FNV-1a hash instead, which always
    returns the same number for the same id.
    """
    # FNV-1a 64-bit: a tiny, well-known, deterministic string hash.
    h = 0xCBF29CE484222325                     # FNV offset basis
    for byte in snippet_id.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF  # FNV prime, kept 64-bit
    return h


def _batched(iterable: Iterable, n: int) -> Iterator[list]:
    """Yield successive lists of up to `n` items from `iterable`.

    The final batch holds the remainder and may be shorter than `n`. Works on any
    iterable (including a generator) without loading it all into memory at once.
    """
    if n < 1:
        raise ValueError("batch size must be >= 1")
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:               # flush whatever is left over
        yield batch


# --------------------------------------------------------------------------- #
# Resume bookmark (how many corpus rows have already been embedded + upserted)
# --------------------------------------------------------------------------- #
def _read_progress() -> int:
    """Return the number of rows already indexed (0 if no bookmark yet)."""
    path = config.EMBED_PROGRESS_PATH
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("done", 0))
    except (json.JSONDecodeError, ValueError, OSError):
        # A corrupt bookmark should not crash the build; just start over.
        return 0


def _write_progress(done: int) -> None:
    """Persist how many rows are done so a re-run can skip them."""
    config.EMBED_PROGRESS_PATH.write_text(
        json.dumps({"done": done}), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed `texts` into vectors, dispatching on `config.EMBED_BACKEND`.

    Returns one float vector per input text. Backends:
      - "openai": call the OpenAI embeddings API (model config.OPENAI_EMBED_MODEL)
                  truncated to config.EMBED_DIM dimensions.
      - "local":  use a cached sentence-transformers model, L2-normalized.
    """
    if not texts:
        return []
    backend = config.EMBED_BACKEND
    if backend == "openai":
        return _embed_openai(texts)
    if backend == "local":
        return _embed_local(texts)
    raise ValueError(f"unknown EMBED_BACKEND: {backend!r}")


def _embed_openai(texts: List[str]) -> List[List[float]]:
    """Embed via OpenAI, batching requests to stay well within API limits."""
    from openai import OpenAI  # lazy: only needed for the openai backend

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    vectors: List[List[float]] = []
    # Send the texts in chunks rather than one giant request.
    for chunk in _batched(texts, config.EMBED_BATCH_SIZE):
        resp = client.embeddings.create(
            model=config.OPENAI_EMBED_MODEL,
            input=chunk,
            dimensions=config.EMBED_DIM,   # truncate to 768 dims
        )
        # resp.data preserves input order; pull out each embedding vector.
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def _get_local_model():
    """Load (once) and return the cached local sentence-transformers model."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from sentence_transformers import SentenceTransformer  # lazy import
        _LOCAL_MODEL = SentenceTransformer(config.LOCAL_EMBED_MODEL)
    return _LOCAL_MODEL


def _embed_local(texts: List[str]) -> List[List[float]]:
    """Embed with the local model, normalized, returned as plain Python lists."""
    model = _get_local_model()
    # normalize_embeddings=True gives unit vectors so cosine similarity is clean.
    embeddings = model.encode(texts, normalize_embeddings=True)
    # encode() returns a numpy array; .tolist() makes JSON/Qdrant-friendly lists.
    return embeddings.tolist()


# --------------------------------------------------------------------------- #
# Qdrant client + collection
# --------------------------------------------------------------------------- #
def get_qdrant_client():
    """Return a Qdrant client, honoring config.QDRANT_LOCAL.

    Local mode uses an on-disk database (no cloud account needed); otherwise we
    connect to a Qdrant server via URL + API key.
    """
    from qdrant_client import QdrantClient  # lazy: avoid import unless indexing

    if config.QDRANT_LOCAL:
        return QdrantClient(path=config.QDRANT_LOCAL_PATH)
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def _recreate_collection(client) -> None:
    """Create the collection fresh (dropping any old one) with the right vectors."""
    from qdrant_client.models import Distance, VectorParams  # lazy import

    # recreate_collection is deprecated; do an explicit drop-then-create so the
    # collection's vector size always matches the active embedding backend.
    if client.collection_exists(config.QDRANT_COLLECTION):
        client.delete_collection(config.QDRANT_COLLECTION)
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=config.embed_dim(),     # size must match the active backend
            distance=Distance.COSINE,
        ),
    )


# --------------------------------------------------------------------------- #
# Build the indexes
# --------------------------------------------------------------------------- #
def build_index() -> None:
    """Embed the corpus into Qdrant and build a BM25 index beside it.

    Steps:
      1. Load every snippet from the cached corpus (ingest.iter_corpus()).
      2. Ensure the Qdrant collection exists. If we are resuming (bookmark > 0)
         we keep the existing collection; otherwise we (re)create it fresh.
      3. Embed the not-yet-done snippets in batches and upsert them, updating the
         resume bookmark after each batch and printing progress.
      4. Build a BM25 keyword index over the whole corpus and pickle it (together
         with the id list + per-id metadata) for retrieve.py.
    """
    import ingest  # lazy: keep embed_index importable even before ingest exists

    # 1. Materialize the corpus once. We need random access (for the resume
    #    slice) and a full pass for BM25, so a list is the simplest choice.
    snippets = list(ingest.iter_corpus())
    total = len(snippets)
    print(f"corpus: {total} snippets")

    # 2. Connect and figure out where to resume from.
    client = get_qdrant_client()
    already_done = _read_progress()
    if already_done >= total:
        already_done = total            # nothing left to embed
    if already_done == 0:
        _recreate_collection(client)    # fresh build -> start the collection clean
        print(f"created collection {config.QDRANT_COLLECTION!r}")
    else:
        print(f"resuming: {already_done} / {total} already embedded")

    # 3. Embed + upsert the remaining snippets in batches.
    remaining = snippets[already_done:]
    done = already_done
    for batch in _batched(remaining, config.EMBED_BATCH_SIZE):
        vectors = embed_texts([s["text"] for s in batch])
        _upsert_batch(client, batch, vectors)
        done += len(batch)
        _write_progress(done)           # bookmark AFTER a successful upsert
        print(f"embedded {done} / {total}")

    # 4. Build + save the BM25 index over the full corpus.
    _build_bm25(snippets)
    print(f"done: {total} snippets indexed; BM25 saved to {config.BM25_PATH}")


def _upsert_batch(client, batch: List[dict], vectors: List[List[float]]) -> None:
    """Upsert one batch of snippets (with their vectors) into Qdrant."""
    from qdrant_client.models import PointStruct  # lazy import

    points = [
        PointStruct(
            id=_point_id(snip["id"]),          # stable int id from the snippet id
            vector=vector,
            # payload carries everything retrieve.py returns in a Passage.
            payload={
                "id": snip["id"],
                "pmid": snip["pmid"],
                "title": snip["title"],
                "text": snip["text"],
            },
        )
        for snip, vector in zip(batch, vectors)
    ]
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)


def _build_bm25(snippets: List[dict]) -> None:
    """Build a BM25Okapi index over the corpus and pickle it for retrieve.py.

    The pickle is a dict with everything needed to map a BM25 hit (an index into
    the corpus) back to a passage:
        {
          "bm25": BM25Okapi,                 # the fitted ranker
          "ids":  [snippet_id, ...],         # row i of bm25 == ids[i]
          "meta": {snippet_id: {pmid, title, text}},  # payload per id
        }
    """
    from rank_bm25 import BM25Okapi  # lazy: only needed when building the index

    # Tokenize each snippet's text; row order here defines BM25's internal order.
    tokenized = [_tokenize(s["text"]) for s in snippets]
    bm25 = BM25Okapi(tokenized)

    ids = [s["id"] for s in snippets]
    meta = {
        s["id"]: {"pmid": s["pmid"], "title": s["title"], "text": s["text"]}
        for s in snippets
    }
    payload = {"bm25": bm25, "ids": ids, "meta": meta}

    # Pickle is fine here: the file is produced and consumed by our own code.
    with open(config.BM25_PATH, "wb") as fh:
        pickle.dump(payload, fh)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    build_index()
