# AskBio — Biomedical RAG with citations and an abstention guardrail

AskBio answers biomedical questions from PubMed research (the Hugging Face `MedRAG/pubmed` corpus). It grounds every answer in retrieved passages and cites the supporting PMIDs inline as `[PMID:xxxx]`. When the retrieved literature doesn't support an answer, it abstains rather than guessing — in a medical setting a confident wrong answer is worse than "I don't know." Retrieval is hybrid (dense + keyword) with cross-encoder reranking, and there's a ragas evaluation harness so you can actually measure output quality instead of just eyeballing the demo.

- **GitHub:** `<add-repo-link>`
- **Live demo:** `<add-streamlit-link>`

---

## Architecture (7 steps)

```
                                       ┌──────────────────────────────────────────┐
  1. INGEST            2. EMBED+INDEX  │            3. RETRIEVE (hybrid)            │
  ───────────         ───────────────  │  ──────────────────────────────────────   │
  HF MedRAG/pubmed    OpenAI 3-small   │   dense (Qdrant, cosine)  ┐                │
  stream → clean      @768  OR local   │                          ├─ RRF fuse ─┐   │
  → corpus.jsonl  ──► MiniLM @384  ──► │   BM25 (rank-bm25 pickle) ┘            │   │
  (PMID, title,       upsert→Qdrant    │                                        ▼   │
   text)              + BM25 pickle    │              cross-encoder rerank → top-5  │
                                       └──────────────────────────┬───────────────┘
                                                                  │
   ┌──────────────────────────────────────────────────────────────┘
   ▼
  4. GENERATE                         5. EVALUATE              6. APP (Streamlit)
  ──────────────────────────────     ─────────────────────   ────────────────────
  gpt-4o-mini | Claude | "none"      ragas on PubMedQA:       question → answer
  grounded prompt (passages only)    • faithfulness           + clickable PubMed
  → cite [PMID:x] inline             • answer_relevancy         citations (Sources
  → VALIDATE every cited PMID        • context_precision        panel + expanders)
    (drop hallucinated ones)         + yes/no/maybe accuracy   → abstention shown
  → ABSTAIN if evidence weak         → eval_results.csv          as a warning
                                       + eval_chart.png
```

*(Step 7 is the swappable-backend config layer — `EMBED_BACKEND` / `LLM_BACKEND` / `QDRANT_LOCAL` — so the same code runs fully free or fully cloud.)*

---

## How it works

- **Hybrid retrieval** — dense vector search (Qdrant, cosine) and BM25 keyword search run in parallel. Embeddings catch paraphrase ("heart attack" ≈ "myocardial infarction"); BM25 catches exact rare tokens (gene names, drug codes) that embeddings tend to blur. Neither alone covers both cases.
- **Reciprocal Rank Fusion (RRF)** — fuses the two ranked lists on rank position rather than raw score, so there's no need to reconcile incompatible scales (cosine similarity vs. a BM25 term-weight sum). A passage ranked high in both lists wins.
- **Cross-encoder reranking** — a `ms-marco-MiniLM-L-6-v2` cross-encoder reads each (query, passage) pair together and re-scores the ~20 fused candidates down to the top 5. More precise than a bi-encoder, and it only runs on the shortlist so it stays cheap.
- **Grounded answers with validated citations** — the prompt forbids outside knowledge and ties every claim to the numbered passages. After generation, each `[PMID:xxxx]` token is regex-extracted and checked against the passages actually retrieved; any PMID the model invented is dropped. A clickable citation that doesn't support its claim is worse than none.
- **Abstention guardrail** — if the passages don't support an answer, the model returns a fixed opt-out phrase and the app flags `abstained` and shows it as a warning. This is the main anti-hallucination mechanism.
- **Evaluation** — ragas (LLM-judged) scores faithfulness, answer relevancy, and context precision on a PubMedQA sample, alongside a plain keyword-based yes/no/maybe accuracy. Outputs a CSV and a bar chart.
- **Deployable** — Streamlit Community Cloud (app) + Qdrant Cloud (vector DB) + OpenAI (embeddings + answers). See `DEPLOY.md`.

---

## Stack

| Layer            | Choice                                                                 |
| ---------------- | ---------------------------------------------------------------------- |
| Language         | Python 3.11                                                            |
| Corpus           | `MedRAG/pubmed` (Hugging Face), streamed → cleaned JSONL with PMIDs    |
| Embeddings       | OpenAI `text-embedding-3-small` @ 768d **or** local `all-MiniLM-L6-v2` @ 384d |
| Vector DB        | Qdrant (cosine) — Qdrant Cloud or on-disk local                        |
| Keyword search   | `rank-bm25` (BM25Okapi), pickled index                                 |
| Fusion           | Reciprocal Rank Fusion (`k=60`)                                        |
| Reranker         | `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers, CPU)    |
| Generation       | `gpt-4o-mini` (OpenAI), Claude (Anthropic), Gemini, or a free extractive `"none"` backend |
| Evaluation       | `ragas` 0.2.x (langchain pinned to 0.3.x) on `PubMedQA`                |
| UI               | Streamlit                                                              |
| Other            | `datasets`, `pandas`, `matplotlib`, `torch` (CPU)                      |

---

## Try it free (no API keys)

The whole pipeline can run at $0: local embeddings, no LLM (extractive demo answers), on-disk Qdrant, tiny corpus.

```bash
cd askbio
pip install -r requirements.txt
```

Create a `.env` next to `app.py` with these four lines (no keys needed):

```env
EMBED_BACKEND=local
LLM_BACKEND=none
QDRANT_LOCAL=1
CORPUS_SUBSET_SIZE=200
```

Then build the tiny index and launch:

```bash
python ingest.py        # streams 200 PubMed snippets → data/corpus.jsonl
python embed_index.py   # local embeddings → on-disk Qdrant + BM25 pickle
streamlit run app.py    # open the browser UI
```

> The free demo uses a generic 200-snippet slice just to exercise the pipeline end to end. Answer quality and citation relevance improve a lot with a larger, topic-focused corpus and a real LLM — see below.

---

## Run the real system

```bash
cd askbio
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
```

Edit `.env` and add your keys: `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `HF_TOKEN`. Keep the defaults `EMBED_BACKEND=openai` and `LLM_BACKEND=openai`. Then:

```bash
python ingest.py        # stream + clean the PubMed corpus (CORPUS_SUBSET_SIZE rows)
python embed_index.py   # OpenAI embeddings → Qdrant Cloud + BM25 pickle  (resumable)
streamlit run app.py    # ask questions, get cited answers
python evaluate.py      # ragas + accuracy → data/eval_results.csv + eval_chart.png
```

Backends are swappable via `.env`:

| Variable        | Options                                      | Notes                                            |
| --------------- | -------------------------------------------- | ------------------------------------------------ |
| `EMBED_BACKEND` | `openai` \| `local`                          | local = free MiniLM on CPU                        |
| `LLM_BACKEND`   | `openai` \| `anthropic` \| `gemini` \| `none`| `none` = free extractive demo, no API call        |
| `QDRANT_LOCAL`  | `1` \| (unset)                               | `1` = on-disk Qdrant; else Qdrant Cloud via URL/key |

---

## Metrics

Measured with ragas (an LLM judges each answer against its retrieved context) on a deterministic sample of PubMedQA expert questions, plus a keyword-based yes/no/maybe accuracy.

| Metric                | Score        | What it checks                                              |
| --------------------- | ------------ | ---------------------------------------------------------- |
| Faithfulness          | `____`       | Every claim in the answer is supported by retrieved passages |
| Answer relevancy      | `____`       | The answer actually addresses the question                  |
| Context precision     | `____`       | Retrieval ranked the useful passages near the top           |
| Accuracy (yes/no/maybe) | `____`     | End-to-end task accuracy on the gold `final_decision` label |

*Measured on N = `____` PubMedQA questions (`pqa_labeled` split; default sample size 50).*

> Caveat: the free demo runs on a tiny, generic 200-snippet slice of PubMed, so out-of-the-box relevance is low. Faithfulness/relevancy/precision rise once you index a larger and/or topic-focused corpus that actually contains the answers — the free slice is for exercising the pipeline, not benchmarking quality. Run `python evaluate.py` against your own index and fill in the real numbers above.

---

## Cost

Running the real system on a focused demo corpus is roughly **~$5 total**:

| Item                         | Approx. cost                                   |
| ---------------------------- | ---------------------------------------------- |
| OpenAI embeddings            | ~$0.50 (one-time, to build the index)          |
| OpenAI answers (`gpt-4o-mini`) | ~$1 (plus the ragas judge during evaluation) |
| Qdrant Cloud free cluster    | $0 (free tier)                                 |
| Streamlit Community Cloud    | $0 (free tier)                                 |
| Hugging Face datasets        | $0 (read token, free)                          |

And the **free demo mode** (`EMBED_BACKEND=local`, `LLM_BACKEND=none`, `QDRANT_LOCAL=1`) costs **$0** with no accounts at all.
