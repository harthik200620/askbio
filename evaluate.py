"""
AskBio - Phase 5: evaluation (the "resume gold" metric).

Plain-English idea
------------------
A RAG system that *sounds* confident is worthless if it quietly makes things up.
Evaluation is how we prove AskBio is trustworthy instead of merely fluent, and it
is the single most interview-worthy artifact in this project: it turns "I built a
chatbot" into "I measured my chatbot and here are the numbers".

We score two complementary things on a held-out slice of **PubMedQA** (expert
yes/no/maybe questions with long reference answers):

1. **RAG quality, via ragas** - three LLM-judged metrics that target the failure
   modes of retrieval-augmented generation:
     * ``faithfulness``       - is every claim in the answer supported by the
                                retrieved passages? (catches hallucination)
     * ``answer_relevancy``   - does the answer actually address the question?
     * ``context_precision``  - did retrieval surface the *useful* passages near
                                the top, not bury them under noise?
   ragas uses a judge LLM + embeddings to grade these, so it needs an API key. If
   no OpenAI key is configured we **skip ragas with a clear warning** and still
   produce the accuracy section - the eval degrades gracefully instead of
   crashing, which matters when someone clones the repo to try it for free.

2. **Task accuracy** - PubMedQA ships a gold ``final_decision`` label
   (yes/no/maybe). We map our free-text answer back to one of those labels with a
   small, transparent keyword heuristic (``predict_label``) and report the share
   we got right. This is a blunt-but-honest end-to-end score that needs no LLM.

Outputs: a per-item + aggregate CSV (``config.EVAL_RESULTS_PATH``) and a bar
chart of the aggregate metrics (``config.EVAL_CHART_PATH``), plus a printed
summary table. ``run_eval`` returns the aggregate dict so callers can assert on
it.

Heavy / optional dependencies (``datasets``, ``ragas``, ``matplotlib``,
``langchain_openai``) are imported **lazily inside the functions that need
them**, so the pure helpers below - ``predict_label`` and ``compute_accuracy`` -
import and unit-test cleanly with nothing but the standard library installed.
"""
from __future__ import annotations

import random
import re
from typing import Optional

import config

# --------------------------------------------------------------------------- #
# Pure helpers (no heavy imports) - these are the unit-tested core.
# --------------------------------------------------------------------------- #

# The three labels PubMedQA uses for its expert ``final_decision`` field.
_VALID_LABELS = ("yes", "no", "maybe")

# Phrases that signal the system declined to answer. When an answer is an
# abstention we must NOT guess yes/no/maybe - that would reward a non-answer.
# Kept lowercase; matched as substrings against the lowercased answer.
_ABSTAIN_MARKERS = (
    config.ABSTAIN_MESSAGE.lower(),
    "i don't have enough information",
    "i do not have enough information",
    "cannot answer",
    "can't answer",
    "no relevant information",
    "not enough information",
)


def predict_label(answer: str) -> str:
    """
    Map a free-text answer onto a PubMedQA label: "yes" | "no" | "maybe" |
    "unknown".

    Heuristic (deliberately simple and auditable - we want a label we can defend,
    not a black box):
      1. Empty or abstaining answers -> "unknown" (we refuse to score a
         non-answer as if it were a real yes/no/maybe).
      2. "maybe" wins if hedging language is present ("maybe", "unclear",
         "inconclusive", "may ", "might", "possibly", "uncertain") - hedging is
         the whole point of the maybe class, so it takes priority over a stray
         "yes"/"no".
      3. Otherwise look at the FIRST explicit yes/no signal: scan for a "yes"
         word and a "no" word and take whichever appears earliest in the text
         (answers tend to lead with their verdict).
      4. No signal at all -> "unknown".
    """
    if not answer or not answer.strip():
        return "unknown"

    text = answer.lower()

    # (1) An abstention is not a yes/no/maybe answer.
    if any(marker in text for marker in _ABSTAIN_MARKERS):
        return "unknown"

    # (2) Hedging -> "maybe". Word-boundary matches avoid firing on substrings
    # like "mayonnaise" or "another".
    if re.search(r"\b(maybe|unclear|inconclusive|may|might|possibly|uncertain)\b", text):
        return "maybe"

    # (3) Earliest explicit yes/no wins.
    yes_match = re.search(r"\byes\b", text)
    no_match = re.search(r"\bno\b", text)
    if yes_match and no_match:
        return "yes" if yes_match.start() < no_match.start() else "no"
    if yes_match:
        return "yes"
    if no_match:
        return "no"

    # (4) Nothing decisive found.
    return "unknown"


def compute_accuracy(pairs: list[tuple[str, str]]) -> dict:
    """
    Accuracy over (predicted_label, gold_label) pairs.

    We only score pairs where a label was actually predicted: a "unknown"
    prediction means our heuristic could not read a verdict out of the answer, so
    counting it as "wrong" would conflate "model abstained / was unparseable"
    with "model gave the wrong verdict". Reporting both numbers (``scored`` vs
    total) keeps that distinction honest.

    Returns a dict with: ``accuracy`` (float in [0,1] over scored items, or 0.0
    when nothing was scored), ``correct``, ``scored``, and ``total``.
    """
    total = len(pairs)
    scored = 0
    correct = 0
    for predicted, gold in pairs:
        if predicted == "unknown":
            continue  # heuristic gave no verdict -> not scoreable
        scored += 1
        if predicted == gold:
            correct += 1
    accuracy = (correct / scored) if scored else 0.0
    return {"accuracy": accuracy, "correct": correct, "scored": scored, "total": total}


def _sample_indices(n_rows: int, sample_size: int, seed: int) -> list[int]:
    """
    Deterministically choose row indices to evaluate.

    Seeded so re-running the eval picks the SAME questions (reproducible numbers
    you can quote). If ``sample_size`` >= dataset size we just take everything.
    """
    indices = list(range(n_rows))
    if sample_size >= n_rows:
        return indices
    rng = random.Random(seed)
    return sorted(rng.sample(indices, sample_size))


def _has_openai_key() -> bool:
    """True if an OpenAI key is configured (controls whether ragas can run)."""
    return bool(config.OPENAI_API_KEY)


# --------------------------------------------------------------------------- #
# ragas wiring (lazy) - the LLM-judged RAG metrics.
# --------------------------------------------------------------------------- #
def _build_ragas_judge():
    """
    Build the (llm, embeddings) pair ragas uses as its *judge*, targeting the
    **ragas 0.2.x** API.

    ragas does not call our generator here - it independently grades the rows we
    hand it, and to do that it needs its own LLM + embeddings. We configure them
    explicitly (rather than relying on ragas' implicit default) so the judge is
    pinned to ``config.OPENAI_LLM_MODEL`` / ``config.OPENAI_EMBED_MODEL`` and the
    eval is reproducible. We wrap LangChain objects in ragas' wrappers, which is
    the documented 0.2.x integration path.

    All imports are local: this function only runs when a key is present.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    # temperature=0 -> as deterministic a judge as the API allows.
    chat = ChatOpenAI(model=config.OPENAI_LLM_MODEL, temperature=0,
                      api_key=config.OPENAI_API_KEY)
    embed = OpenAIEmbeddings(model=config.OPENAI_EMBED_MODEL,
                             api_key=config.OPENAI_API_KEY)
    judge_llm = LangchainLLMWrapper(chat)
    judge_embeddings = LangchainEmbeddingsWrapper(embed)
    return judge_llm, judge_embeddings


def _run_ragas(rows: list[dict]) -> dict:
    """
    Score ``rows`` with ragas and return ``{metric_name: float}``.

    ``rows`` must already be in ragas' expected shape - one dict per item with
    keys ``question``, ``answer``, ``contexts`` (list[str]) and ``ground_truth``.
    We build a ``datasets.Dataset`` from them (the column layout ragas 0.2.x
    reads) and call ``ragas.evaluate`` with our explicit judge + the three
    metrics.

    Raises on failure; the caller wraps this in try/except so a ragas/API hiccup
    degrades to "accuracy only" instead of crashing the whole eval.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    judge_llm, judge_embeddings = _build_ragas_judge()

    # Dataset.from_list builds columns from our list-of-dicts; ragas 0.2.x reads
    # the `question / answer / contexts / ground_truth` columns by name.
    dataset = Dataset.from_list(rows)
    metrics = [faithfulness, answer_relevancy, context_precision]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,            # explicit judge LLM (no implicit global default)
        embeddings=judge_embeddings,
    )

    # In 0.2.x the EvaluationResult is dict-like and also exposes to_pandas().
    # We average each metric column to get one aggregate score per metric, which
    # is robust whether a value comes back as a scalar or a per-row series.
    scores = result.to_pandas()
    out: dict = {}
    for name in ("faithfulness", "answer_relevancy", "context_precision"):
        if name in scores.columns:
            out[name] = float(scores[name].mean())
    return out


# --------------------------------------------------------------------------- #
# Data loading + the per-item RAG pass.
# --------------------------------------------------------------------------- #
def _load_eval_rows(sample_size: int) -> list[dict]:
    """
    Load + deterministically sample PubMedQA, returning the raw fields we need:
    ``question``, ``final_decision`` (gold label) and ``long_answer`` (reference
    answer used as ragas' ground_truth).
    """
    from datasets import load_dataset

    ds = load_dataset(config.HF_EVAL, config.EVAL_CONFIG, split="train")
    indices = _sample_indices(len(ds), sample_size, config.RANDOM_SEED)

    items: list[dict] = []
    for idx in indices:
        record = ds[idx]
        items.append(
            {
                "question": str(record.get("question", "")),
                "final_decision": str(record.get("final_decision", "")).strip().lower(),
                "long_answer": str(record.get("long_answer", "")),
            }
        )
    return items


def _answer_one(question: str) -> dict:
    """
    Run the real RAG pipeline for one question: retrieve passages, then generate
    a grounded answer. Returns ``{"answer": str, "contexts": list[str]}``.

    Imported lazily so this module loads without the retrieval/generation stack
    (and their heavy deps) being importable - the pure helpers stay testable.
    """
    import generate
    import retrieve

    passages = retrieve.retrieve(question)
    res = generate.generate_answer(question, passages)
    # ragas wants the contexts as a list of plain strings.
    contexts = [p["text"] for p in passages]
    return {"answer": res["answer"], "contexts": contexts}


# --------------------------------------------------------------------------- #
# Output: CSV, chart, summary table.
# --------------------------------------------------------------------------- #
def _save_csv(per_item: list[dict], aggregate: dict) -> None:
    """
    Write the per-item results plus a trailing aggregate summary to
    ``config.EVAL_RESULTS_PATH`` using pandas.

    The aggregate metrics are appended as extra columns on a final ``__AGGREGATE__``
    marker row so the whole eval (rows + headline numbers) lives in one file.
    """
    import pandas as pd

    df = pd.DataFrame(per_item)

    # One summary row carrying the aggregate scalars; NaN elsewhere is expected.
    summary_row = {"id": "__AGGREGATE__"}
    summary_row.update({f"agg_{k}": v for k, v in aggregate.items()})
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    config.EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.EVAL_RESULTS_PATH, index=False)
    print(f"[eval] wrote per-item + aggregate CSV -> {config.EVAL_RESULTS_PATH}")


def _save_chart(aggregate: dict) -> None:
    """
    Save a bar chart of the aggregate metric values to ``config.EVAL_CHART_PATH``.

    Only plots the numeric headline metrics (the ragas scores + accuracy), each
    naturally on a 0..1 scale, so a single shared y-axis reads cleanly. Uses the
    non-interactive "Agg" backend so it works headless (CI, servers).
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: render to file, never open a window
    import matplotlib.pyplot as plt

    # Pull the plottable 0..1 metrics in a stable order; skip anything missing
    # (e.g. ragas keys absent when it was skipped).
    candidates = ["faithfulness", "answer_relevancy", "context_precision", "accuracy"]
    names = [m for m in candidates if isinstance(aggregate.get(m), (int, float))]
    values = [float(aggregate[m]) for m in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    if names:
        bars = ax.bar(names, values, color="#3b7dd8")
        ax.set_ylim(0, 1)
        ax.set_ylabel("score (0-1)")
        ax.set_title("AskBio evaluation - aggregate metrics")
        # Label each bar with its value for an at-a-glance read.
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                    f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "no metrics to plot", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()

    config.EVAL_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.EVAL_CHART_PATH, dpi=120)
    plt.close(fig)
    print(f"[eval] wrote bar chart -> {config.EVAL_CHART_PATH}")


def _print_summary(aggregate: dict, ragas_skipped: bool) -> None:
    """Print a small aligned summary table of the headline numbers."""
    print("\n" + "=" * 44)
    print("AskBio evaluation summary")
    print("=" * 44)
    if ragas_skipped:
        print("ragas metrics : SKIPPED (no OpenAI key configured)")
    for name in ("faithfulness", "answer_relevancy", "context_precision"):
        if name in aggregate:
            print(f"{name:<22}: {aggregate[name]:.3f}")
    acc = aggregate.get("accuracy")
    if isinstance(acc, (int, float)):
        print(f"{'accuracy':<22}: {acc:.3f} "
              f"({aggregate.get('correct', 0)}/{aggregate.get('scored', 0)} scored, "
              f"{aggregate.get('n_items', 0)} items)")
    print("=" * 44 + "\n")


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def run_eval(sample_size: int = config.EVAL_SAMPLE_SIZE) -> dict:
    """
    End-to-end evaluation. See the module docstring for the "why".

    Steps:
      1. Load + deterministically sample PubMedQA.
      2. For each item: ``retrieve()`` -> ``generate_answer()``; collect the row
         shape ragas wants (question / answer / contexts / ground_truth) and the
         predicted vs gold label for accuracy.
      3. ragas (faithfulness / answer_relevancy / context_precision), guarded:
         if no OpenAI key OR ragas errors, skip it with a warning and continue.
      4. Compute yes/no/maybe accuracy.
      5. Save CSV + bar chart, print a summary, and return the aggregate dict.
    """
    items = _load_eval_rows(sample_size)
    print(f"[eval] evaluating {len(items)} PubMedQA items "
          f"(seed={config.RANDOM_SEED})")

    ragas_rows: list[dict] = []   # fed to ragas
    per_item: list[dict] = []     # fed to the CSV
    label_pairs: list[tuple[str, str]] = []

    for i, item in enumerate(items):
        question = item["question"]
        gold = item["final_decision"]
        ground_truth = item["long_answer"]

        result = _answer_one(question)
        answer = result["answer"]
        contexts = result["contexts"]

        predicted = predict_label(answer)
        label_pairs.append((predicted, gold))

        ragas_rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )
        per_item.append(
            {
                "id": i,
                "question": question,
                "answer": answer,
                "n_contexts": len(contexts),
                "predicted_label": predicted,
                "gold_label": gold,
                "correct": int(predicted == gold and predicted != "unknown"),
            }
        )

    # ---- ragas (graceful degradation) ----
    aggregate: dict = {}
    ragas_skipped = False
    if not _has_openai_key():
        ragas_skipped = True
        print("[eval] WARNING: no OPENAI_API_KEY configured - skipping ragas "
              "metrics (faithfulness / answer_relevancy / context_precision). "
              "Accuracy is still computed below.")
    else:
        try:
            aggregate.update(_run_ragas(ragas_rows))
        except Exception as exc:  # noqa: BLE001 - any ragas/API failure is non-fatal
            ragas_skipped = True
            print(f"[eval] WARNING: ragas evaluation failed ({exc!r}) - "
                  "continuing with accuracy only.")

    # ---- accuracy ----
    acc = compute_accuracy(label_pairs)
    aggregate["accuracy"] = acc["accuracy"]
    aggregate["correct"] = acc["correct"]
    aggregate["scored"] = acc["scored"]
    aggregate["n_items"] = acc["total"]
    aggregate["ragas_skipped"] = ragas_skipped

    # ---- outputs ----
    _save_csv(per_item, aggregate)
    _save_chart(aggregate)
    _print_summary(aggregate, ragas_skipped)
    return aggregate


if __name__ == "__main__":
    run_eval()
