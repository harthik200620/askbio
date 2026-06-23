"""
Evaluation for AskBio: scores RAG quality (via ragas) and yes/no/maybe accuracy
on a sample of PubMedQA, writes a CSV + bar chart, and prints a summary.

The heavy/optional deps (datasets, ragas, matplotlib, langchain_openai) are
imported lazily inside the functions that use them, so predict_label and
compute_accuracy can be imported and unit-tested with just the stdlib.
"""
from __future__ import annotations

import random
import re
from typing import Optional

import config

# Labels PubMedQA uses in its final_decision field.
_VALID_LABELS = ("yes", "no", "maybe")

# Substrings that mean the answer abstained; if any match we don't guess a
# yes/no/maybe verdict. Lowercase, compared against the lowercased answer.
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
    """Map a free-text answer to "yes"/"no"/"maybe"/"unknown".

    Just a keyword heuristic, not a classifier: abstentions and empty answers are
    "unknown", any hedging word wins "maybe", otherwise the earliest of yes/no
    wins. Good enough to score against PubMedQA's gold labels.
    """
    if not answer or not answer.strip():
        return "unknown"

    text = answer.lower()

    if any(marker in text for marker in _ABSTAIN_MARKERS):
        return "unknown"

    # Hedging beats a stray yes/no. \b avoids "mayonnaise", "another", etc.
    if re.search(r"\b(maybe|unclear|inconclusive|may|might|possibly|uncertain)\b", text):
        return "maybe"

    yes_match = re.search(r"\byes\b", text)
    no_match = re.search(r"\bno\b", text)
    if yes_match and no_match:
        return "yes" if yes_match.start() < no_match.start() else "no"
    if yes_match:
        return "yes"
    if no_match:
        return "no"

    return "unknown"


def compute_accuracy(pairs: list[tuple[str, str]]) -> dict:
    """Accuracy over (predicted, gold) pairs.

    "unknown" predictions are excluded from the denominator rather than counted
    wrong -- otherwise an unparseable answer looks the same as a wrong verdict.
    We report scored vs total so that's visible.
    """
    total = len(pairs)
    scored = 0
    correct = 0
    for predicted, gold in pairs:
        if predicted == "unknown":
            continue
        scored += 1
        if predicted == gold:
            correct += 1
    accuracy = (correct / scored) if scored else 0.0
    return {"accuracy": accuracy, "correct": correct, "scored": scored, "total": total}


def _sample_indices(n_rows: int, sample_size: int, seed: int) -> list[int]:
    """Pick row indices to evaluate, seeded so re-runs hit the same questions."""
    indices = list(range(n_rows))
    if sample_size >= n_rows:
        return indices
    rng = random.Random(seed)
    return sorted(rng.sample(indices, sample_size))


def _has_openai_key() -> bool:
    return bool(config.OPENAI_API_KEY)


def _build_ragas_judge():
    """Build the (llm, embeddings) judge ragas grades rows with.

    ragas needs its own LLM + embeddings (it doesn't reuse our generator). We
    pin them to the configured models for reproducibility and wrap the LangChain
    objects in ragas' wrappers -- the 0.2.x integration path. Only called when a
    key is present, so imports stay local.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    chat = ChatOpenAI(model=config.OPENAI_LLM_MODEL, temperature=0,
                      api_key=config.OPENAI_API_KEY)
    embed = OpenAIEmbeddings(model=config.OPENAI_EMBED_MODEL,
                             api_key=config.OPENAI_API_KEY)
    judge_llm = LangchainLLMWrapper(chat)
    judge_embeddings = LangchainEmbeddingsWrapper(embed)
    return judge_llm, judge_embeddings


def _run_ragas(rows: list[dict]) -> dict:
    """Score rows with ragas, returning {metric_name: float}.

    rows must already have the keys ragas 0.2.x reads as columns: question,
    answer, contexts (list[str]), ground_truth. Raises on failure -- the caller
    catches it and falls back to accuracy only.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    judge_llm, judge_embeddings = _build_ragas_judge()

    dataset = Dataset.from_list(rows)
    metrics = [faithfulness, answer_relevancy, context_precision]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    # Average each metric column; works whether a value is a scalar or per-row.
    scores = result.to_pandas()
    out: dict = {}
    for name in ("faithfulness", "answer_relevancy", "context_precision"):
        if name in scores.columns:
            out[name] = float(scores[name].mean())
    return out


def _load_eval_rows(sample_size: int) -> list[dict]:
    """Load and sample PubMedQA, keeping question, final_decision (gold label)
    and long_answer (used as ragas' ground_truth)."""
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
    """Retrieve + generate for one question. Returns answer and contexts.

    Imports are lazy so the module loads (and the pure helpers test) without the
    retrieval/generation stack installed.
    """
    import generate
    import retrieve

    passages = retrieve.retrieve(question)
    res = generate.generate_answer(question, passages)
    contexts = [p["text"] for p in passages]
    return {"answer": res["answer"], "contexts": contexts}


def _save_csv(per_item: list[dict], aggregate: dict) -> None:
    """Write per-item rows + a trailing __AGGREGATE__ row to the results CSV."""
    import pandas as pd

    df = pd.DataFrame(per_item)

    summary_row = {"id": "__AGGREGATE__"}
    summary_row.update({f"agg_{k}": v for k, v in aggregate.items()})
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    config.EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.EVAL_RESULTS_PATH, index=False)
    print(f"[eval] wrote per-item + aggregate CSV -> {config.EVAL_RESULTS_PATH}")


def _save_chart(aggregate: dict) -> None:
    """Bar chart of the aggregate metrics (all 0..1) to the chart path."""
    import matplotlib
    matplotlib.use("Agg")  # headless, render straight to file
    import matplotlib.pyplot as plt

    # Skip any metric that's missing (e.g. ragas keys when it was skipped).
    candidates = ["faithfulness", "answer_relevancy", "context_precision", "accuracy"]
    names = [m for m in candidates if isinstance(aggregate.get(m), (int, float))]
    values = [float(aggregate[m]) for m in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    if names:
        bars = ax.bar(names, values, color="#3b7dd8")
        ax.set_ylim(0, 1)
        ax.set_ylabel("score (0-1)")
        ax.set_title("AskBio evaluation - aggregate metrics")
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


def run_eval(sample_size: int = config.EVAL_SAMPLE_SIZE) -> dict:
    """Run the whole eval: sample PubMedQA, retrieve+generate per item, score
    with ragas (skipped if no key), compute accuracy, write outputs, return the
    aggregate dict."""
    items = _load_eval_rows(sample_size)
    print(f"[eval] evaluating {len(items)} PubMedQA items "
          f"(seed={config.RANDOM_SEED})")

    ragas_rows: list[dict] = []   # for ragas
    per_item: list[dict] = []     # for the CSV
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

    # ragas, but skip (don't crash) if there's no key or it errors.
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

    acc = compute_accuracy(label_pairs)
    aggregate["accuracy"] = acc["accuracy"]
    aggregate["correct"] = acc["correct"]
    aggregate["scored"] = acc["scored"]
    aggregate["n_items"] = acc["total"]
    aggregate["ragas_skipped"] = ragas_skipped

    _save_csv(per_item, aggregate)
    _save_chart(aggregate)
    _print_summary(aggregate, ragas_skipped)
    return aggregate


if __name__ == "__main__":
    run_eval()
