# LEARN.md — Defend AskBio in an Interview

A concept-by-concept study guide for the AskBio RAG system. Each section gives a plain-English explanation you can say out loud, then two likely interview questions with answer hints. Read it once and you can defend every design decision in the project.

The big picture in one breath: **a question comes in → we retrieve the most relevant PubMed passages two different ways and fuse them (retrieval) → we rerank for precision → an LLM writes an answer using *only* those passages and cites them → we verify the citations and abstain if the evidence is weak → we measure the whole thing with ragas.**

---

## 1. Embeddings

An embedding turns a piece of text into a list of numbers (a vector) that captures its *meaning*, so that texts about similar ideas land close together in vector space. AskBio embeds every PubMed snippet once at index time and embeds the user's question at query time; "relevant" then just means "nearby vector." We use OpenAI's `text-embedding-3-small` truncated to 768 dimensions (paid, high quality) or a free local `all-MiniLM-L6-v2` at 384 dimensions on CPU — the same model must embed both the corpus and the query, or the comparison is meaningless.

- **Q: Why do paraphrases like "heart attack" and "myocardial infarction" match even though they share no words?**
  Because the embedding model maps semantically similar text to nearby vectors regardless of surface wording — it learned meaning from context during training, so synonyms cluster together.
- **Q: Why must the query be embedded with the same model as the documents?**
  Different models produce different, non-comparable vector spaces (and even different dimensionalities). Distances across two spaces are nonsense; AskBio routes both through one `embed_texts` path to guarantee they match.

---

## 2. Vector database (Qdrant, cosine similarity)

A vector database stores millions of embeddings and answers "find me the *k* nearest vectors to this one" in milliseconds, using an approximate-nearest-neighbor index instead of scanning everything. AskBio uses **Qdrant** with **cosine** distance; each stored point carries a payload (`id`, `pmid`, `title`, `text`) so a search hit is already a full passage. It runs either on-disk locally (free, no account) or against Qdrant Cloud — the same client code, switched by one env var.

- **Q: Why cosine similarity rather than Euclidean distance?**
  Cosine compares the *direction* (angle) of vectors, not their magnitude, so it isn't thrown off by document length or vector scale — the standard, robust choice for text embeddings. (The local backend even L2-normalizes vectors so cosine is clean.)
- **Q: How does a vector DB stay fast at scale — does it compare against every vector?**
  No. It builds an approximate-nearest-neighbor index (e.g. HNSW graphs) that finds *near*-nearest neighbors in roughly logarithmic time, trading a tiny bit of recall for a massive speedup over brute force.

---

## 3. Keyword search / BM25

BM25 is a classic, non-neural ranking function that scores a document by how often the query's *exact words* appear in it, weighting rare words more heavily and dampening very long documents. It understands no synonyms at all — but it's unbeatable at exact, rare tokens like a gene name (`TP53`), an acronym, or a drug code that an embedding model may never have seen clearly. AskBio builds a `rank-bm25` (`BM25Okapi`) index over the corpus and pickles it; the query is tokenized with the **exact same** lowercase/split-on-non-alphanumeric rule the documents were, or the term matching silently fails.

- **Q: BM25 is decades old — why keep it alongside modern embeddings?**
  It's the perfect complement: embeddings handle paraphrase but blur rare exact tokens; BM25 is literal and excels at exactly those tokens. Hybrid retrieval uses each where it's strong.
- **Q: Why does the query tokenizer have to match the document tokenizer byte-for-byte?**
  BM25 matches *tokens*, not raw strings. If the query keeps punctuation or casing the documents dropped, "COVID-19" and "covid 19" become different tokens and the match disappears — so AskBio pins one tokenizer for both.

---

## 4. Hybrid retrieval + Reciprocal Rank Fusion (RRF)

Hybrid retrieval runs dense (vector) and sparse (BM25) search in parallel and merges their results, getting semantic recall *and* exact-term precision in one shortlist. The hard part is merging: dense scores are cosine similarities (~0–1) and BM25 scores are unbounded term-weight sums — **completely different scales you can't just add**. **Reciprocal Rank Fusion** sidesteps this by throwing away the raw scores and fusing on *rank position only*: an item at rank `r` in a list contributes `1 / (k + r)` to its fused score (AskBio uses `k=60`), and a passage's final score is the sum across both lists. A document ranked high in *both* lists accumulates two big terms and wins — robust consensus, zero score normalization.

- **Q: Why fuse on rank instead of normalizing and adding the scores?**
  Score normalization is fragile — it depends on each retriever's score distribution, which shifts per query and per corpus. Rank is a stable, universal signal, so RRF "just works" without tuning, which is why it's become the default for hybrid search.
- **Q: What does the constant `k` (=60) actually do?**
  It softens the curve. With small `k`, rank 0 dominates everything; `k=60` keeps the gap between, say, rank 5 and rank 6 meaningful, so good-but-not-top passages from one retriever still count. 60 is the widely used default.

---

## 5. Cross-encoder reranking (vs. bi-encoder)

A **bi-encoder** (what the embedding/vector search is) encodes the query and each document *separately* into vectors and compares them — fast and cacheable, but the two never "see" each other, so it can only approximate relevance. A **cross-encoder** feeds the (query, passage) pair through the model *together*, letting every query token attend to every passage token; it's far more accurate at judging relevance but too slow to run over a whole corpus. AskBio gets the best of both: cheap bi-encoder + BM25 sweep for recall down to ~20 candidates, then the `ms-marco-MiniLM-L-6-v2` cross-encoder reranks just those to the final **top 5**.

- **Q: If a cross-encoder is more accurate, why not use it for the whole search?**
  Cost. A bi-encoder embeds documents once and reuses them; a cross-encoder must run a fresh forward pass for *every* (query, document) pair at query time — infeasible over 100k docs. So it's reserved for reranking a small shortlist.
- **Q: What does the reranker fix that fusion alone doesn't?**
  Fusion is precision-blind — it only knows ranks, not true relevance. The cross-encoder actually *reads* each candidate against the question and reorders them, pushing the genuinely on-topic passages to the top before they reach the LLM.

---

## 6. Grounded generation

Grounded generation means the LLM answers **only** from the retrieved passages, not from its own training memory. AskBio's system prompt explicitly forbids outside knowledge, hands the model numbered, PMID-tagged passages, and requires inline `[PMID:xxxx]` citations; temperature is 0 for deterministic, non-creative answers. This is the first line of defense against hallucination: if the answer can only come from supplied evidence, it can't drift into plausible-sounding fabrication. (A free `"none"` backend instead stitches an extractive answer straight from the top passages, keeping the same grounded+cited shape with no API call.)

- **Q: How is RAG different from just asking the LLM the question directly?**
  A bare LLM answers from frozen, unverifiable training data and can confidently make things up. RAG injects fresh, specific, *citable* source text at query time, so answers are current, traceable to PMIDs, and constrained to real evidence.
- **Q: Why set temperature to 0 for a grounded medical answer?**
  Temperature controls randomness. For factual, evidence-bound answers you want determinism and faithfulness to the passages, not creative variation — 0 minimizes the model wandering off the provided context.

---

## 7. Citation validation

LLMs sometimes emit a plausible-looking citation for a source that was never in the context — a hallucinated reference. AskBio defends against this *after* generation: it regex-extracts every `[PMID:xxxx]` token from the answer and **keeps only PMIDs that were actually among the passages it retrieved**, dropping the rest (and de-duplicating, preserving first-mention order). A clickable citation that doesn't support its claim is *worse* than no citation, because it manufactures false trust — so AskBio only surfaces verifiable ones.

- **Q: Why validate citations programmatically instead of trusting the prompt to only cite real PMIDs?**
  Prompts reduce but never eliminate hallucination — the model can still cite a PMID outside the context. A deterministic post-check is a hard guarantee: an invalid PMID *cannot* reach the user, regardless of what the model wrote.
- **Q: How do you know a cited PMID is valid?**
  Each passage carries its PMID; AskBio builds the set of retrieved PMIDs and checks membership. If the cited PMID isn't in the passages handed to the model for *this* query, it's dropped as unsupported.

---

## 8. Abstention & why it fights hallucination

Abstention is the system's ability to say "I don't have enough information in the literature to answer that" instead of forcing an answer. The prompt instructs the model to return that exact opt-out sentence when the passages don't support an answer; AskBio detects it and raises an explicit `abstained` flag, which the UI shows as a warning rather than a normal answer. In a biomedical context a confident-but-wrong answer can be dangerous, so a calibrated "I don't know" is a *feature*, and it's the single most direct counter to hallucination: the model is given a sanctioned escape hatch instead of being cornered into inventing something.

- **Q: Why is abstaining better than always producing an answer?**
  Because the cost of errors is asymmetric — in medicine a wrong answer can do real harm, while "I don't know" just defers to a human. Abstention trades a little coverage for a lot of trustworthiness, and it's measurable (the eval treats abstentions as non-answers, not wrong guesses).
- **Q: How does the system decide *when* to abstain?**
  It's driven by the retrieved evidence: the grounding prompt tells the model that if the passages don't contain the answer, it must return the exact abstain phrase. If retrieval surfaced nothing on-topic, the model has nothing to ground on and opts out — so retrieval quality and abstention are linked.

---

## 9. ragas evaluation (faithfulness vs. context recall)

ragas is a framework that uses an LLM as a *judge* to score RAG outputs on targeted axes, so you measure quality instead of eyeballing it. The crucial idea is that different metrics **isolate different components of the pipeline**:

- **Faithfulness isolates the *generator*** — it checks whether every claim in the answer is supported by the retrieved context. A low score means the LLM is hallucinating *beyond* what it was given, independent of whether retrieval was good.
- **Context recall / context precision isolate the *retriever*** — did retrieval actually surface the passages needed to answer (recall), and rank the useful ones near the top (precision)? These measure the *evidence supplied*, independent of how well the LLM then wrote.

AskBio reports **faithfulness, answer_relevancy, and context_precision** on a deterministic PubMedQA sample, plus a transparent yes/no/maybe accuracy that needs no LLM. ragas itself needs an API key (its judge is an LLM); if none is set, AskBio skips it with a warning and still prints accuracy — it degrades gracefully.

- **Q: If your answer quality is bad, how do you tell whether retrieval or generation is at fault?**
  Split the metrics. Low context precision/recall but high faithfulness → retrieval is feeding bad evidence. High context scores but low faithfulness → retrieval is fine and the generator is hallucinating. That decomposition is exactly why both kinds of metric exist.
- **Q: Isn't using an LLM to grade an LLM circular?**
  It's a known limitation, which is why AskBio pairs LLM-judged ragas metrics with a *non-LLM* accuracy (mapping answers to the gold yes/no/maybe label with a transparent heuristic). The judge is also pinned to temperature 0 and a fixed model for reproducibility, and the sample is seeded so the numbers are stable and quotable.

---

## 10. Deployment

AskBio deploys as three managed pieces: the Streamlit app on **Streamlit Community Cloud**, the vector index on **Qdrant Cloud**, and embeddings/answers via the **OpenAI API**. The key subtlety is that the hosted app only *queries* — you build the Qdrant index and the BM25 pickle **locally first**, then the cloud app reads from them. Secrets live in Streamlit's encrypted secrets store (TOML), never in the repo. Full walkthrough in `DEPLOY.md`.

- **Q: Why build the index locally instead of on the deployed app?**
  Indexing streams a large corpus and makes thousands of embedding calls — slow, memory-heavy, and not what a request-scoped web app should do on startup. You build once locally, push vectors to Qdrant Cloud, and the app just runs fast read queries.
- **Q: The BM25 index is a local pickle — how does the deployed app get it?**
  It must ship with the app. That's why the live demo uses a *focused, smaller* corpus: a ~10–25k-row `corpus.jsonl` + `bm25.pkl` are small enough to commit to the repo (or rebuild from a committed corpus on first load), whereas a full 100k index would be too large/costly to host for free.

---

### 60-second whiteboard recap

> "It's a retrieval-augmented generation system over PubMed. I retrieve candidate passages two ways — dense embeddings in Qdrant for meaning, and BM25 for exact terms — fuse them with Reciprocal Rank Fusion so I never mix incompatible score scales, then rerank the shortlist with a cross-encoder for precision. The LLM answers using only those passages and cites PMIDs inline; I validate each citation against what was actually retrieved and drop hallucinated ones, and the system abstains when the evidence is weak. I evaluate with ragas — faithfulness to isolate the generator, context metrics to isolate the retriever — plus a non-LLM accuracy on PubMedQA. Every backend is swappable via env vars, so the same code runs fully free locally or on the cloud."
