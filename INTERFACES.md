# AskBio — module reference

A quick map of the modules and their public functions. Each module imports
constants from `config.py` and the shared dataclasses from `schemas.py`
(`Snippet`, `Passage`, `Citation`, `AnswerResult`) rather than redefining them.
Most modules also run standalone as `python <module>.py` (a small CLI) where noted.

## ingest.py  - Phase 1
- `load_corpus(limit: int = config.CORPUS_SUBSET_SIZE) -> list[Snippet]`
  Stream `config.HF_CORPUS` from Hugging Face, map each record to a `Snippet`
  (extract `pmid`, `title`, `text`), clean (drop empty text, dedupe by `id`),
  cache to `config.CORPUS_PATH` as JSONL, and return the list. **Resumable:** if
  the cache already has >= `limit` rows, load it instead of re-downloading.
- `iter_corpus() -> Iterator[Snippet]`  - stream rows from the cached JSONL.
- CLI: ensure the corpus is cached, then print 3 sample snippets
  (pmid + title + first ~200 chars of text).

## embed_index.py  - Phase 2
- `embed_texts(texts: list[str]) -> list[list[float]]`  - dispatches on
  `config.EMBED_BACKEND` ("openai" -> text-embedding-3-small @ dim 768; "local"
  -> all-MiniLM-L6-v2 @ dim 384).
- `build_index() -> None`
  Read `corpus.jsonl`, embed in batches of `config.EMBED_BATCH_SIZE`, upsert into
  Qdrant (collection `config.QDRANT_COLLECTION`, size `config.embed_dim()`,
  distance Cosine), payload `{id, pmid, title, text}`. **Resumable** via
  `config.EMBED_PROGRESS_PATH`. Also build a BM25 index over the corpus and save
  it to `config.BM25_PATH`. Show progress.
- Provide `get_qdrant_client()` honoring `config.QDRANT_LOCAL`.
- CLI: run `build_index()`.

## retrieve.py  - Phase 3
- `dense_search(query: str, top_k: int = config.DENSE_TOP_K) -> list[Passage]`
- `bm25_search(query: str, top_k: int = config.BM25_TOP_K) -> list[Passage]`
- `rrf_fuse(ranked_lists: list[list[Passage]], k: int = config.RRF_K, top_k: int = config.RRF_TOP_K) -> list[Passage]`
- `rerank(query: str, passages: list[Passage], top_k: int = config.RERANK_TOP_K) -> list[Passage]`
  (cross-encoder `config.RERANK_MODEL`)
- `retrieve(query: str, top_k: int = config.RERANK_TOP_K) -> list[Passage]`
  dense + bm25 -> `rrf_fuse` -> `rerank`. This is the single entry point used by
  `generate.py`, `evaluate.py`, and `app.py`.
- CLI: read a question from argv, print the final passages (pmid + score + text).

## generate.py  - Phase 4
- `generate_answer(query: str, passages: list[Passage]) -> AnswerResult`
  Build a grounded prompt that (a) uses ONLY the numbered passages, (b) cites
  PMIDs inline as `[PMID:1234]`, (c) returns `config.ABSTAIN_MESSAGE`
  (`abstained=True`) when the passages don't support an answer. Parse citations
  and **verify** each cited PMID is among the provided passages (drop invalid
  ones). LLM via `config.LLM_BACKEND` ("openai" | "anthropic" | "gemini" |
  "none"=extractive demo that needs no API key).
- CLI: question in -> answer + citations out.

## evaluate.py  - Phase 5
- `run_eval(sample_size: int = config.EVAL_SAMPLE_SIZE) -> dict`
  Load PubMedQA (`config.HF_EVAL` / `config.EVAL_CONFIG`); for each item run
  `retrieve()` + `generate_answer()`; compute ragas metrics (faithfulness,
  answer_relevancy, context_precision) and a yes/no/maybe accuracy; save the CSV
  (`config.EVAL_RESULTS_PATH`) + bar chart (`config.EVAL_CHART_PATH`); print a
  summary. Pin `ragas` to the 0.2.x API.
- CLI: `run_eval()`.

## app.py  - Phase 6
Streamlit UI: question box -> `retrieve()` -> `generate_answer()` -> render the
answer + expandable citations linking to `config.PUBMED_URL`. Cache heavy objects
with `st.cache_resource`.
