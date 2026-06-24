"""Hybrid retrieval: dense + BM25, fused with RRF, then a cross-encoder rerank.

Dense search catches paraphrase/synonyms but can whiff on rare exact tokens
(gene names, drug codes); BM25 is the reverse. Run both, fuse the ranked lists
with RRF (the two score scales aren't comparable, so we fuse on rank not score),
then rerank the ~20 survivors with a cross-encoder for precision.

Heavy deps (qdrant_client, sentence_transformers, rank_bm25) are imported inside
the functions that use them, so the module and its tests import fine without them.
"""
from __future__ import annotations

import re
from typing import Optional

import config
from schemas import Passage

# Has to match embed_index.py's tokenizer exactly, or query tokens won't line up
# with the indexed ones and BM25 quietly returns garbage.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on runs of non-alphanumerics. Matches embed_index.py.

    "COVID-19 vaccine!" -> ["covid", "19", "vaccine"]
    """
    return [tok for tok in _NON_ALNUM.split(text.lower()) if tok]


# Loaded once and reused; retrieve() gets called in a loop during eval/the app.
_BM25_BUNDLE: Optional[dict] = None
_CROSS_ENCODER = None  # sentence_transformers.CrossEncoder


def dense_search(query: str, top_k: int = config.DENSE_TOP_K, collection: str = None) -> list[Passage]:
    """Embed the query and return its top_k nearest snippets from Qdrant.

    Embeds via embed_index so query and corpus share one embedding path; the
    payload already holds id/pmid/title/text, so we just attach Qdrant's score.
    collection: if set, search that collection instead of config.QDRANT_COLLECTION.
    """
    import embed_index

    query_vector = embed_index.embed_texts([query])[0]
    client = embed_index.get_qdrant_client()
    coll = collection or config.QDRANT_COLLECTION

    response = client.query_points(
        collection_name=coll,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    hits = response.points

    passages: list[Passage] = []
    for hit in hits:
        payload = hit.payload or {}
        passages.append(
            Passage(
                id=str(payload.get("id", "")),
                pmid=str(payload.get("pmid", "")),
                title=payload.get("title", "") or "",
                text=payload.get("text", "") or "",
                score=float(hit.score),
            )
        )
    return passages


def _build_bm25_from_qdrant() -> dict:
    """Build the BM25 bundle from the Qdrant collection's payloads.

    Used when there's no local bm25.pkl - e.g. on a deploy, where the pickle is
    gitignored. Qdrant already stores each snippet's text in its payload, so we
    scroll the whole collection once and build BM25 from that. Nothing to ship.
    """
    from rank_bm25 import BM25Okapi
    import embed_index

    client = embed_index.get_qdrant_client()
    ids: list[str] = []
    meta: dict[str, dict] = {}
    tokenized: list[list[str]] = []

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            doc_id = str(payload.get("id", point.id))
            ids.append(doc_id)
            meta[doc_id] = {
                "pmid": str(payload.get("pmid", "")),
                "title": payload.get("title", "") or "",
                "text": payload.get("text", "") or "",
            }
            tokenized.append(_tokenize(meta[doc_id]["text"]))
        if offset is None:  # scroll returns None for the offset when fully paged
            break

    return {"bm25": BM25Okapi(tokenized), "ids": ids, "meta": meta}


def _load_bm25_bundle() -> dict:
    """Load the BM25 bundle once. Prefer the pickle embed_index.py writes locally;
    if it's missing (e.g. on a deploy with only Qdrant), rebuild it from the
    collection's payloads instead.
    """
    global _BM25_BUNDLE
    if _BM25_BUNDLE is None:
        if config.BM25_PATH.exists():
            import pickle

            with open(config.BM25_PATH, "rb") as fh:
                _BM25_BUNDLE = pickle.load(fh)
        else:
            _BM25_BUNDLE = _build_bm25_from_qdrant()
    return _BM25_BUNDLE


def _meta_for_id(meta, doc_id: str, index: int) -> dict:
    """Get one snippet's metadata. meta may be a dict keyed by id or a list
    aligned with ids; handle both. Always returns a dict."""
    if isinstance(meta, dict):
        return meta.get(doc_id, {}) or {}
    if isinstance(meta, (list, tuple)) and 0 <= index < len(meta):
        return meta[index] or {}
    return {}


def bm25_search(query: str, top_k: int = config.BM25_TOP_K) -> list[Passage]:
    """Return the top_k snippets BM25 scores highest for the query.

    Carries the raw BM25 score through; RRF ignores it but rerank/debug may look.
    """
    bundle = _load_bm25_bundle()
    bm25 = bundle["bm25"]
    ids = bundle["ids"]
    meta = bundle.get("meta")

    scores = bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    passages: list[Passage] = []
    for index in ranked_indices[:top_k]:
        doc_id = str(ids[index])
        fields = _meta_for_id(meta, doc_id, index)
        passages.append(
            Passage(
                id=doc_id,
                pmid=str(fields.get("pmid", "")),
                title=fields.get("title", "") or "",
                text=fields.get("text", "") or "",
                score=float(scores[index]),
            )
        )
    return passages


def rrf_fuse(
    ranked_lists: list[list[Passage]],
    k: int = config.RRF_K,
    top_k: int = config.RRF_TOP_K,
) -> list[Passage]:
    """Fuse several best-first ranked lists with Reciprocal Rank Fusion.

    Ranks are 0-based: the item at position r contributes 1/(k + r), and an id's
    fused score is the sum of those contributions across every list it's in. So a
    doc that ranks well in both lists beats one that tops only a single list -
    the consensus we want from dense+BM25. k (default 60) flattens the curve so
    rank 0 doesn't dominate, which is why RRF works without normalising scores.

    Pure (no globals/IO). Dedupes by id keeping the first passage seen, writes the
    fused score onto each kept passage, returns the top_k sorted descending.
    """
    fused_score: dict[str, float] = {}
    chosen: dict[str, Passage] = {}

    for ranked in ranked_lists:
        for rank, passage in enumerate(ranked):
            doc_id = passage["id"]
            fused_score[doc_id] = fused_score.get(doc_id, 0.0) + 1.0 / (k + rank)
            # First passage wins; lists come best-first so this favours the
            # earlier source on ties.
            if doc_id not in chosen:
                chosen[doc_id] = passage

    # Stable sort -> equal scores keep first-seen order, so output is deterministic.
    ordered_ids = sorted(fused_score, key=lambda doc_id: fused_score[doc_id], reverse=True)

    fused: list[Passage] = []
    for doc_id in ordered_ids[:top_k]:
        passage = dict(chosen[doc_id])  # copy, don't mutate caller's input
        passage["score"] = fused_score[doc_id]
        fused.append(passage)  # type: ignore[arg-type]
    return fused


def _get_cross_encoder():
    """Build and cache the cross-encoder reranker (config.RERANK_MODEL) once."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder(config.RERANK_MODEL)
    return _CROSS_ENCODER


def rerank(
    query: str,
    passages: list[Passage],
    top_k: int = config.RERANK_TOP_K,
) -> list[Passage]:
    """Re-score passages against the query with a cross-encoder, keep the top_k.

    The cross-encoder reads the (query, passage) pair jointly rather than as two
    separate vectors - more accurate but costlier, so we only run it on the fused
    shortlist. Overwrites score with the cross-encoder's relevance (the score the
    final passages are supposed to carry).
    """
    if not passages:
        return []

    model = _get_cross_encoder()

    pairs = [[query, passage["text"]] for passage in passages]
    scores = model.predict(pairs)

    scored: list[Passage] = []
    for passage, score in zip(passages, scores):
        passage = dict(passage)  # copy, don't mutate caller
        passage["score"] = float(score)
        scored.append(passage)  # type: ignore[arg-type]

    scored.sort(key=lambda p: p["score"], reverse=True)
    return scored[:top_k]


def retrieve(query: str, top_k: int = config.RERANK_TOP_K, collection: str = None) -> list[Passage]:
    """Full pipeline: dense + BM25 -> RRF fuse -> cross-encoder rerank.

    If collection is set (e.g., for eval), use that instead of the default.
    When using a non-default collection, skip BM25 (only do dense + rerank).
    The one entry point generate.py / evaluate.py / app.py call.
    """
    if collection:
        dense_hits = dense_search(query, collection=collection)
        return rerank(query, dense_hits, top_k=top_k)

    dense_hits = dense_search(query)
    bm25_hits = bm25_search(query)
    fused = rrf_fuse([dense_hits, bm25_hits], top_k=config.RRF_TOP_K)
    return rerank(query, fused, top_k=top_k)


def _main() -> None:
    """CLI: python retrieve.py "your question here" -> prints ranked passages."""
    import sys

    if len(sys.argv) < 2:
        print('usage: python retrieve.py "your question here"')
        raise SystemExit(2)

    query = " ".join(sys.argv[1:])
    passages = retrieve(query)

    print(f"query: {query}")
    print(f"top {len(passages)} passages")
    for rank, passage in enumerate(passages, start=1):
        print("-" * 60)
        print(f"#{rank}  pmid={passage['pmid']}  score={round(passage['score'], 4)}")
        print(passage["text"][:200])


if __name__ == "__main__":
    _main()
