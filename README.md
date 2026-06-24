# AskBio — Biomedical RAG with citations and an abstention guardrail

AskBio is a retrieval-augmented generation (RAG) system that answers biomedical questions from PubMed research. Every claim in the answer is grounded in a retrieved abstract and cited with its PMID, so you can click through and verify. When the evidence isn't there — a question outside the corpus, or one the literature is genuinely unclear on — the app says so rather than inventing an answer. A wrong medical answer delivered confidently is worse than a clear "I don't know."

**Live demo:** [askbio.streamlit.app](https://askbio.streamlit.app/)  
**GitHub:** [harthik200620/askbio](https://github.com/harthik200620/askbio)

---

## What it covers

The indexed corpus focuses on four well-studied areas so the demo reliably retrieves strong evidence:

- **Cardiovascular disease** — hypertension, coronary artery disease, heart failure, stroke, beta-blockers, statins
- **Type 2 diabetes** — insulin resistance, metformin, glycemic control, HbA1c
- **Pain and fever medications** — aspirin, ibuprofen, paracetamol/acetaminophen, NSAIDs, COX-2 inhibitors
- **Bacterial infections and antibiotics** — antibiotic resistance, common antibiotics, sepsis, pneumonia

Questions inside these areas typically get a cited, grounded answer. Questions outside them (earthquakes, quantum physics, etc.) trigger the abstention guardrail. That's by design.

---

## How it works

```
 1. INGEST              2. EMBED + INDEX         3. RETRIEVE (hybrid)
 ─────────────         ─────────────────        ──────────────────────────────
 HF MedRAG/pubmed      local MiniLM-L6-v2       dense search (Qdrant, cosine)  ┐
 stream → filter by    @ 384d  OR  OpenAI        BM25 keyword search             ├─ RRF fuse
 topic keyword →       3-small @ 768d            (exact tokens: drug names, etc) ┘
 corpus.jsonl          → Qdrant Cloud +          → cross-encoder rerank → top 5
 (~14k snippets,       BM25 pickle
 4 topics)

 4. GENERATE                          5. EVALUATE (offline)       6. APP
 ────────────────────────────────     ────────────────────────   ─────────────────────
 grounded prompt: answer from         ragas on PubMedQA:          question → answer
 passages ONLY, no outside            faithfulness, relevancy,    + [PMID:xxxx] citations
 knowledge → cite [PMID:xxxx]         context precision           + clickable PubMed links
 → validate every cited PMID          + yes/no/maybe accuracy     → abstention shown as
 → abstain if evidence weak           → eval_results.csv + chart    yellow warning
```

**Dense + keyword retrieval** run in parallel and fuse with Reciprocal Rank Fusion (RRF). Embeddings catch paraphrase ("heart attack" ≈ "myocardial infarction"); BM25 catches rare exact tokens like drug names that embeddings blur. RRF fuses on rank position, not raw score, so the two incompatible score scales never need normalising.

**Cross-encoder reranking** takes the top ~20 fused candidates and re-scores each (query, passage) pair jointly — much more precise than bi-encoder similarity, and cheap because it only runs on the shortlist.

**Grounded generation** instructs the LLM to answer from the numbered passages only, cite every claim with `[PMID:xxxx]`, and return the abstention phrase verbatim if the passages don't cover the question. After generation, every cited PMID is validated against the retrieved passages: any PMID the model invented is silently dropped before the response reaches the user.

**Relevance guardrail** fires before the LLM even gets called: if the best cross-encoder score is below a configurable threshold, the app abstains immediately rather than generating a low-confidence answer.

---

## Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Corpus | `MedRAG/pubmed` (Hugging Face), ~14k focused abstracts across 4 topics |
| Embeddings | `all-MiniLM-L6-v2` @ 384d (local, free) or OpenAI `text-embedding-3-small` @ 768d |
| Vector DB | Qdrant (cosine similarity) — Qdrant Cloud or on-disk local |
| Keyword search | `rank-bm25` (BM25Okapi), pickled index |
| Fusion | Reciprocal Rank Fusion, k=60 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU, sentence-transformers) |
| Generation | Gemini 2.5 Flash (free), gpt-4o-mini, Claude, or extractive fallback |
| Evaluation | ragas 0.2.x on PubMedQA (`pqa_labeled`), OpenAI or Gemini as judge |
| UI | Streamlit |

Running cost with Gemini + local embeddings + Qdrant Cloud free tier: **$0/month.**

---

## Try it free (no API keys needed)

The whole pipeline runs without any accounts or paid services. You get real hybrid retrieval and reranking; the only difference is the "none" backend quotes passages directly instead of synthesising prose.

```bash
git clone https://github.com/harthik200620/askbio.git
cd askbio
pip install -r requirements.txt
```

Create a `.env` file:

```env
EMBED_BACKEND=local
LLM_BACKEND=none
QDRANT_LOCAL=1
CORPUS_PER_TOPIC_TARGET=50
```

Then build and run:

```bash
python ingest.py        # collects ~200 snippets (50 per topic) into data/corpus.jsonl
python embed_index.py   # local embeddings → on-disk Qdrant + BM25 pickle
streamlit run app.py
```

This is a smoke test, not a benchmark. With only 50 snippets per topic the retrieval is sparse — raise `CORPUS_PER_TOPIC_TARGET` to 3500 for the full corpus.

---

## Run with Gemini (free, real answers)

```bash
cp .env.example .env   # Windows: Copy-Item .env.example .env
```

Edit `.env` and fill in:
- `QDRANT_URL` and `QDRANT_API_KEY` — from [cloud.qdrant.io](https://cloud.qdrant.io) (free tier)
- `GEMINI_API_KEY` — from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (free)

Keep `EMBED_BACKEND=local` and `LLM_BACKEND=gemini`, then:

```bash
python ingest.py        # streams PubMed, fills topic buckets → ~14k snippets
python embed_index.py   # embeds and uploads to Qdrant Cloud (resumable)
streamlit run app.py
python evaluate.py      # optional: ragas + accuracy → data/eval_results.csv
```

The corpus build scans a few hundred thousand PubMed records and stops once each topic bucket reaches its target. `embed_index.py` is resumable — if it's interrupted, just re-run it and it picks up from where it stopped.

---

## Backends

| Variable | Options | Default |
|---|---|---|
| `EMBED_BACKEND` | `local` · `openai` | `local` |
| `LLM_BACKEND` | `gemini` · `openai` · `anthropic` · `none` | `gemini` |
| `QDRANT_LOCAL` | `1` (on-disk) · unset (cloud) | cloud |
| `CORPUS_PER_TOPIC_TARGET` | any integer | `3500` |
| `RELEVANCE_THRESHOLD` | float (cross-encoder score) | `-3.0` |

The Gemini backend rotates through up to three keys (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`) so the free quota goes further without hitting limits mid-session.

---

## Evaluation

Run `python evaluate.py` to measure output quality on a random sample of PubMedQA expert-labelled questions. It writes `data/eval_results.csv` and a bar chart.

ragas judges each answer against its retrieved context using an LLM (OpenAI or Gemini). Set `OPENAI_API_KEY` or `GEMINI_API_KEY` in `.env` before running. To skip ragas and only compute accuracy, set `SKIP_RAGAS=1`.

### Metrics

Measured with ragas (LLM-judged) on a deterministic sample of 50 PubMedQA expert-labelled questions, using a fair eval corpus (PubMedQA abstracts indexed separately so questions can be answered from their source material).

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | **0.82** | Every claim in the answer is supported by the retrieved passages |
| Answer relevancy | **0.78** | The answer actually addresses the question that was asked |
| Context precision | **0.85** | Useful passages were ranked near the top of the retrieved list |
| Accuracy (yes/no/maybe) | **0.60** | End-to-end label accuracy against PubMedQA gold annotations |

*N = 50 questions, `pqa_labeled` split, seed 42.*

**Why abstention helps:** The system abstains on 40% of questions when the literature doesn't clearly support an answer. This conservative stance trades some coverage for accuracy — unanswered questions don't reduce the final score (only scored predictions count toward accuracy), so abstaining appropriately *improves* the metric compared to confidently wrong answers.

To re-run evaluation with your own API key:
```bash
export OPENAI_API_KEY=sk-...      # or GEMINI_API_KEY=...
python evaluate.py                 # rebuilds eval corpus, runs 50 questions, generates metrics
```

The evaluation write results to `data/eval_results.csv` (per-item scores) and `data/eval_chart.png` (aggregate bar chart).

---

## Project layout

```
askbio/
├── ingest.py         # stream PubMed → topic-filtered JSONL corpus
├── embed_index.py    # embed corpus → Qdrant + BM25 pickle (resumable)
├── retrieve.py       # dense + BM25 → RRF → cross-encoder rerank
├── generate.py       # grounded generation with citation validation + abstention
├── evaluate.py       # ragas + yes/no/maybe accuracy on PubMedQA
├── app.py            # Streamlit UI
├── config.py         # all settings, read from .env
├── schemas.py        # TypedDicts: Snippet, Passage, Citation, AnswerResult
├── requirements.txt
└── .env.example      # template — copy to .env and fill in keys
```

Tests live in `tests/` and cover the pure logic in each module (RRF math, citation validation, key rotation, BM25 tokenizer consistency, etc.).
