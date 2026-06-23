"""
AskBio - Phase 3: hybrid retrieval (dense + BM25), RRF fusion, cross-encoder rerank.

Plain-English idea
-------------------
Given a user question, we want the handful of PubMed snippets most likely to
contain the answer. No single retriever is enough on its own:

  * **Dense (vector) search** understands *meaning* - it matches paraphrases and
    synonyms ("heart attack" ~ "myocardial infarction") but can miss an exact,
    rare token (a gene name, a drug code) that wasn't well represented in the
    embedding.
  * **BM25 (keyword) search** is the opposite - it nails exact terms and rare
    tokens but is blind to paraphrase.

So we run BOTH, then combine their ranked lists. The trick is that their scores
are on totally different, non-comparable scales (cosine similarity vs. a BM25
term-weight sum), so we can't just add the scores. Instead we use **Reciprocal
Rank Fusion (RRF)**, which throws the raw scores away and fuses purely on *rank
position* - a robust, parameter-light method that's become the standard for
hybrid search. Finally, a **cross-encoder reranker** reads each surviving
(query, passage) pair *together* (not as two separate vectors) and re-scores
them; this is slow, so we only ever run it on the ~20 fused candidates, not the
whole corpus. The result is the precision of a cross-encoder at the cost of a
cheap first-stage recall sweep.

Design notes
------------
* Heavy libraries (qdrant_client, sentence_transformers, rank_bm25 via the saved
  pickle) are imported **lazily inside the functions that use them** so this
  module - and the pure-function unit tests below - import instantly even on a
  machine without those packages or any built index.
* ``rrf_fuse`` is deliberately **pure** (stdlib only, no I/O, no globals) so the
  fusion math can be unit-tested in isolation - it is the interview-critical bit.
* ``_tokenize`` mirrors embed_index.py's tokenizer **exactly** (lowercase, split
  on non-alphanumerics). BM25 scores are only valid if the query is tokenized the
  same way the documents were at index-build time.
"""
from __future__ import annotations

import re
from typing import Optional

import config
from schemas import Passage

# --------------------------------------------------------------------------- #
# Tokenizer - MUST stay byte-for-byte equivalent to embed_index.py's rule.
# --------------------------------------------------------------------------- #
# BM25 compares the query's tokens against the tokens the documents were indexed
# with. If the two tokenizers disagree (e.g. one keeps punctuation, one lowers
# case and one doesn't), the same word turns into different tokens and the match
# silently fails. We therefore pin ONE rule here and use it for every query:
# lowercase, then split on any run of non-alphanumeric characters, dropping the
# empty strings that splitting at the string edges produces.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """
    Lowercase ``text`` and split it into alphanumeric tokens.

    Identical in behaviour to embed_index.py's tokenizer so the query and the
    stored documents land in the same token space. Examples:
        "COVID-19 vaccine!"   -> ["covid", "19", "vaccine"]
        "  TP53/BRCA1  "      -> ["tp53", "brca1"]
    """
    # Lowercase first so the split pattern only needs the lowercase a-z class.
    return [tok for tok in _NON_ALNUM.split(text.lower()) if tok]


# --------------------------------------------------------------------------- #
# Module-level caches for the expensive-to-load objects.
# --------------------------------------------------------------------------- #
# Loading the BM25 pickle off disk and constructing the cross-encoder both cost
# real time/memory, and ``retrieve`` may be called many times in a row (eval,
# the Streamlit app). We load each once and reuse it for the process lifetime.
_BM25_BUNDLE: Optional[dict] = None
_CROSS_ENCODER = None  # sentence_transformers.CrossEncoder, lazily constructed


# --------------------------------------------------------------------------- #
# Stage 1a - dense (vector) search via Qdrant.
# --------------------------------------------------------------------------- #
def dense_search(query: str, top_k: int = config.DENSE_TOP_K) -> list[Passage]:
    """
    Embed the query and return its ``top_k`` nearest snippets from Qdrant.

    The query is embedded with the SAME backend/model the corpus was embedded
    with (we call straight into embed_index so there is one embedding code path),
    then we ask Qdrant for the nearest vectors in ``config.QDRANT_COLLECTION``.
    Each hit's stored payload already carries ``id/pmid/title/text``, so mapping
    a hit to a ``Passage`` is just attaching Qdrant's similarity ``score``.
    """
    # Lazy: importing embed_index pulls in qdrant_client / the embedding model.
    import embed_index

    query_vector = embed_index.embed_texts([query])[0]
    client = embed_index.get_qdrant_client()

    # qdrant-client >=1.12 removed .search() in favor of .query_points(); the
    # response wraps the hits in .points (each a ScoredPoint with .payload/.score).
    response = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
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
                score=float(hit.score),  # cosine similarity from Qdrant
            )
        )
    return passages


# --------------------------------------------------------------------------- #
# Stage 1b - sparse (BM25 / keyword) search from the saved pickle.
# --------------------------------------------------------------------------- #
def _load_bm25_bundle() -> dict:
    """
    Load (once) the BM25 bundle embed_index.py pickled to ``config.BM25_PATH``.

    The bundle is the dict ``{"bm25", "ids", "meta"}``:
      * ``bm25`` - the fitted rank_bm25 model (so unpickling needs rank_bm25
        importable, which is why this import is lazy/local),
      * ``ids``  - the corpus ids in the SAME order BM25 scores them,
      * ``meta`` - the snippet fields (pmid/title/text) keyed/ordered by id so we
        can rebuild a full Passage from a BM25 hit without re-reading the corpus.
    """
    global _BM25_BUNDLE
    if _BM25_BUNDLE is None:
        import pickle  # stdlib, but only needed on first use

        # Opening in binary; unpickling the model implicitly requires rank_bm25.
        with open(config.BM25_PATH, "rb") as fh:
            _BM25_BUNDLE = pickle.load(fh)
    return _BM25_BUNDLE


def _meta_for_id(meta, doc_id: str, index: int) -> dict:
    """
    Pull one snippet's metadata out of ``meta`` regardless of how it was stored.

    embed_index may serialize ``meta`` either as a dict keyed by id or as a list
    aligned with ``ids``; we support both so retrieval is robust to that choice.
    Always returns a dict (possibly empty) so callers can ``.get`` safely.
    """
    if isinstance(meta, dict):
        return meta.get(doc_id, {}) or {}
    if isinstance(meta, (list, tuple)) and 0 <= index < len(meta):
        return meta[index] or {}
    return {}


def bm25_search(query: str, top_k: int = config.BM25_TOP_K) -> list[Passage]:
    """
    Return the ``top_k`` snippets BM25 scores highest for ``query``.

    We tokenize the query with ``_tokenize`` (matching the index), get a BM25
    score per document, then take the ``top_k`` highest. We sort indices by score
    descending and rebuild a ``Passage`` for each, carrying the raw BM25 score
    (which RRF will discard in favour of rank, but rerank/debug may inspect).
    """
    bundle = _load_bm25_bundle()
    bm25 = bundle["bm25"]
    ids = bundle["ids"]
    meta = bundle.get("meta")

    scores = bm25.get_scores(_tokenize(query))

    # Rank document indices by score, highest first, and keep the top_k. Sorting
    # by index is unnecessary; we only need the leading slice after the sort.
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
                score=float(scores[index]),  # raw BM25 term-weight sum
            )
        )
    return passages


# --------------------------------------------------------------------------- #
# Stage 2 - Reciprocal Rank Fusion (PURE: stdlib only, no I/O, no globals).
# --------------------------------------------------------------------------- #
def rrf_fuse(
    ranked_lists: list[list[Passage]],
    k: int = config.RRF_K,
    top_k: int = config.RRF_TOP_K,
) -> list[Passage]:
    """
    Fuse several ranked passage lists into one, using Reciprocal Rank Fusion.

    Convention (documented): each list is assumed already sorted best-first.
    We use **0-based ranks**, so the item at position ``r`` in a list contributes

        1 / (k + r)

    to its id's fused score (the top item, r=0, contributes the most: 1/k). A
    passage's final score is the SUM of these contributions across every list it
    appears in. Two consequences that make RRF good for hybrid search:

      * A doc ranked highly in BOTH lists accumulates two large terms and so
        beats a doc ranked highly in only one - exactly the consensus behaviour
        we want from dense+BM25.
      * The constant ``k`` (default 60) softens the curve: it stops rank-0 from
        dominating outright and keeps differences between low ranks meaningful,
        which is why RRF is robust without any score normalisation.

    Pure function: depends only on its arguments (no globals, no I/O), so the
    fusion maths is fully unit-testable. Dedupes by Passage ``id`` (keeping the
    first passage object seen for that id, scanning the lists in order), writes
    the fused value into each kept passage's ``score``, and returns the ``top_k``
    passages sorted by fused score descending.
    """
    fused_score: dict[str, float] = {}
    chosen: dict[str, Passage] = {}  # id -> the Passage object we keep for it

    for ranked in ranked_lists:
        for rank, passage in enumerate(ranked):  # rank is 0-based
            doc_id = passage["id"]
            fused_score[doc_id] = fused_score.get(doc_id, 0.0) + 1.0 / (k + rank)
            # Keep the FIRST passage object we encounter for this id. Lists are
            # passed best-list-first, so this prefers the earlier source on ties.
            if doc_id not in chosen:
                chosen[doc_id] = passage

    # Order ids by fused score (desc). Python's sort is stable, so ids with equal
    # scores retain their first-seen insertion order for deterministic output.
    ordered_ids = sorted(fused_score, key=lambda doc_id: fused_score[doc_id], reverse=True)

    fused: list[Passage] = []
    for doc_id in ordered_ids[:top_k]:
        passage = dict(chosen[doc_id])  # copy so we don't mutate the caller's input
        passage["score"] = fused_score[doc_id]  # expose the fused score
        fused.append(passage)  # type: ignore[arg-type]  (a Passage-shaped dict)
    return fused


# --------------------------------------------------------------------------- #
# Stage 3 - cross-encoder reranking (precision pass over the fused candidates).
# --------------------------------------------------------------------------- #
def _get_cross_encoder():
    """
    Lazily build and cache the cross-encoder reranker (``config.RERANK_MODEL``).

    A cross-encoder is expensive to load, so we construct it once and stash it in
    a module global. The import is local so merely importing this module never
    pulls in sentence_transformers / torch.
    """
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
    """
    Re-score ``passages`` against ``query`` with a cross-encoder, keep the best.

    Unlike the first stage (which embeds query and document *separately*), a
    cross-encoder feeds the (query, passage) pair through the model *together*,
    letting every query token attend to every passage token. That joint view is
    far more accurate at judging relevance - but quadratically costlier, so we
    only run it on the small fused candidate set, never the whole corpus.

    We overwrite each passage's ``score`` with its cross-encoder relevance score
    (this is the score the schema says the FINAL passages should carry) and
    return the ``top_k`` highest, sorted descending.
    """
    if not passages:  # nothing to do; avoids a needless model load
        return []

    model = _get_cross_encoder()

    # The model scores a list of [query, text] pairs in one batched forward pass.
    pairs = [[query, passage["text"]] for passage in passages]
    scores = model.predict(pairs)

    scored: list[Passage] = []
    for passage, score in zip(passages, scores):
        passage = dict(passage)  # copy: don't mutate the caller's objects
        passage["score"] = float(score)  # final relevance score per the schema
        scored.append(passage)  # type: ignore[arg-type]

    scored.sort(key=lambda p: p["score"], reverse=True)
    return scored[:top_k]


# --------------------------------------------------------------------------- #
# Public entry point - the full hybrid pipeline.
# --------------------------------------------------------------------------- #
def retrieve(query: str, top_k: int = config.RERANK_TOP_K) -> list[Passage]:
    """
    Full hybrid retrieval: dense + BM25 -> RRF fuse -> cross-encoder rerank.

    This is the ONE function the rest of the system (generate.py, evaluate.py,
    app.py) calls. The flow:
      1. Pull candidates two independent ways (semantic + keyword) for recall.
      2. Fuse the two ranked lists with RRF down to ``config.RRF_TOP_K`` - a
         robust consensus that needs no score normalisation.
      3. Rerank that shortlist with a cross-encoder for precision, returning the
         ``top_k`` passages the answer will actually be grounded in.
    """
    dense_hits = dense_search(query)
    bm25_hits = bm25_search(query)
    fused = rrf_fuse([dense_hits, bm25_hits], top_k=config.RRF_TOP_K)
    return rerank(query, fused, top_k=top_k)


# --------------------------------------------------------------------------- #
# CLI - quick manual check: `python retrieve.py "your question here"`.
# --------------------------------------------------------------------------- #
def _main() -> None:
    """Read a question from argv and print the final ranked passages."""
    import sys

    if len(sys.argv) < 2:
        print('usage: python retrieve.py "your question here"')
        raise SystemExit(2)

    query = " ".join(sys.argv[1:])
    passages = retrieve(query)

    print(f"query: {query}")
    print(f"top {len(passages)} passages")
    for rank, passage in enumerate(passages, start=1):  # 1-based for humans
        print("-" * 60)
        # Round the score for readable output; show a 200-char text preview.
        print(f"#{rank}  pmid={passage['pmid']}  score={round(passage['score'], 4)}")
        print(passage["text"][:200])


if __name__ == "__main__":
    _main()
