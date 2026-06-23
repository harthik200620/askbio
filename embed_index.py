"""Build the dense (Qdrant) and BM25 indexes from the cleaned corpus.

Heavy deps (sentence_transformers, openai, qdrant_client, rank_bm25) are imported
inside the functions that use them so this module stays importable, and the pure
helpers stay testable, without them installed.
"""
from __future__ import annotations

import json
import pickle
import re
from typing import Iterable, Iterator, List

import config

# Loaded once and reused; spinning up a sentence-transformers model isn't cheap.
_LOCAL_MODEL = None

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase and split into alphanumeric tokens for BM25."""
    if not text:
        return []
    lowered = text.lower()
    return [tok for tok in _NON_ALNUM.split(lowered) if tok]


def _point_id(snippet_id: str) -> int:
    """Stable non-negative int id for Qdrant, derived from the snippet id.

    Qdrant wants int (or UUID) ids. Built-in hash() is salted per process and
    won't survive a restart, which breaks resume/re-index, so use FNV-1a.
    """
    h = 0xCBF29CE484222325
    for byte in snippet_id.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _batched(iterable: Iterable, n: int) -> Iterator[list]:
    """Yield lists of up to n items; final batch may be short. Works on generators."""
    if n < 1:
        raise ValueError("batch size must be >= 1")
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


# Resume bookmark: rows already embedded + upserted, so a re-run can skip them.
def _read_progress() -> int:
    """Rows already indexed, or 0 if there's no bookmark."""
    path = config.EMBED_PROGRESS_PATH
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("done", 0))
    except (json.JSONDecodeError, ValueError, OSError):
        # Corrupt bookmark -> start over rather than crash.
        return 0


def _write_progress(done: int) -> None:
    config.EMBED_PROGRESS_PATH.write_text(
        json.dumps({"done": done}), encoding="utf-8"
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts to vectors via the backend named in config.EMBED_BACKEND."""
    if not texts:
        return []
    backend = config.EMBED_BACKEND
    if backend == "openai":
        return _embed_openai(texts)
    if backend == "local":
        return _embed_local(texts)
    raise ValueError(f"unknown EMBED_BACKEND: {backend!r}")


def _embed_openai(texts: List[str]) -> List[List[float]]:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    vectors: List[List[float]] = []
    for chunk in _batched(texts, config.EMBED_BATCH_SIZE):
        resp = client.embeddings.create(
            model=config.OPENAI_EMBED_MODEL,
            input=chunk,
            dimensions=config.EMBED_DIM,
        )
        # resp.data keeps input order.
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def _get_local_model():
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _LOCAL_MODEL = SentenceTransformer(config.LOCAL_EMBED_MODEL)
    return _LOCAL_MODEL


def _embed_local(texts: List[str]) -> List[List[float]]:
    model = _get_local_model()
    # Unit vectors so cosine similarity behaves.
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def get_qdrant_client():
    """Qdrant client: on-disk in local mode, otherwise a remote server."""
    from qdrant_client import QdrantClient

    if config.QDRANT_LOCAL:
        return QdrantClient(path=config.QDRANT_LOCAL_PATH)
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def _recreate_collection(client) -> None:
    from qdrant_client.models import Distance, VectorParams

    # recreate_collection is deprecated, so drop-then-create ourselves. This also
    # keeps the vector size in sync with whatever backend is active.
    if client.collection_exists(config.QDRANT_COLLECTION):
        client.delete_collection(config.QDRANT_COLLECTION)
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=config.embed_dim(),
            distance=Distance.COSINE,
        ),
    )


def build_index() -> None:
    """Embed the corpus into Qdrant and build a BM25 index alongside it."""
    import ingest

    # Materialize once: we need the resume slice and a full pass for BM25.
    snippets = list(ingest.iter_corpus())
    total = len(snippets)
    print(f"corpus: {total} snippets")

    client = get_qdrant_client()
    already_done = _read_progress()
    if already_done >= total:
        already_done = total
    if already_done == 0:
        _recreate_collection(client)
        print(f"created collection {config.QDRANT_COLLECTION!r}")
    else:
        print(f"resuming: {already_done} / {total} already embedded")

    remaining = snippets[already_done:]
    done = already_done
    for batch in _batched(remaining, config.EMBED_BATCH_SIZE):
        vectors = embed_texts([s["text"] for s in batch])
        _upsert_batch(client, batch, vectors)
        done += len(batch)
        _write_progress(done)           # only after the upsert lands
        print(f"embedded {done} / {total}")

    _build_bm25(snippets)
    print(f"done: {total} snippets indexed; BM25 saved to {config.BM25_PATH}")


def _upsert_batch(client, batch: List[dict], vectors: List[List[float]]) -> None:
    from qdrant_client.models import PointStruct

    points = [
        PointStruct(
            id=_point_id(snip["id"]),
            vector=vector,
            # Whatever retrieve.py needs to build a Passage.
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
    """Fit a BM25Okapi ranker over the corpus and pickle it for retrieve.py.

    retrieve.py expects a dict shaped like:
        {
          "bm25": BM25Okapi,                          # the fitted ranker
          "ids":  [snippet_id, ...],                  # row i of bm25 == ids[i]
          "meta": {snippet_id: {pmid, title, text}},  # payload per id
        }
    """
    from rank_bm25 import BM25Okapi

    # Row order here is BM25's internal order, so ids[] must line up with it.
    tokenized = [_tokenize(s["text"]) for s in snippets]
    bm25 = BM25Okapi(tokenized)

    ids = [s["id"] for s in snippets]
    meta = {
        s["id"]: {"pmid": s["pmid"], "title": s["title"], "text": s["text"]}
        for s in snippets
    }
    payload = {"bm25": bm25, "ids": ids, "meta": meta}

    with open(config.BM25_PATH, "wb") as fh:
        pickle.dump(payload, fh)


if __name__ == "__main__":
    build_index()
