"""
Standard-library-only tests for evaluate.py.

These never touch the network, an LLM, ragas, datasets, or matplotlib: they
exercise the two pure helpers that carry the eval's logic -
``predict_label`` (free-text answer -> yes/no/maybe/unknown) and
``compute_accuracy`` (scoring over (predicted, gold) pairs). Importing
``evaluate`` is safe with nothing but the stdlib + config installed because all
heavy deps are imported lazily inside the functions that use them. Run with::

    python -m unittest tests.test_evaluate      # from the askbio/ folder
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the askbio/ package root importable when run as a bare file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate  # noqa: E402  (path tweak must happen before this import)


class TestPredictLabel(unittest.TestCase):
    """predict_label must read a yes/no/maybe verdict out of free text."""

    def test_clear_yes(self):
        self.assertEqual(evaluate.predict_label("Yes, the treatment is effective."), "yes")

    def test_clear_no(self):
        self.assertEqual(evaluate.predict_label("No, there is no such association."), "no")

    def test_clear_maybe_from_hedge(self):
        # Hedging language maps to the "maybe" class.
        self.assertEqual(
            evaluate.predict_label("The evidence is inconclusive and unclear."), "maybe"
        )

    def test_maybe_word_directly(self):
        self.assertEqual(evaluate.predict_label("Maybe, more study is needed."), "maybe")

    def test_abstain_answer_is_unknown(self):
        # The system's abstain message must NOT be scored as a real verdict.
        self.assertEqual(evaluate.predict_label(evaluate.config.ABSTAIN_MESSAGE), "unknown")

    def test_empty_answer_is_unknown(self):
        self.assertEqual(evaluate.predict_label(""), "unknown")
        self.assertEqual(evaluate.predict_label("   "), "unknown")

    def test_no_signal_is_unknown(self):
        # A sentence with no yes/no/maybe signal is unscoreable.
        self.assertEqual(
            evaluate.predict_label("The study enrolled 200 patients across three sites."),
            "unknown",
        )

    def test_hedge_beats_stray_yes(self):
        # "maybe"-type hedging takes priority over an incidental "yes".
        self.assertEqual(
            evaluate.predict_label("Yes in some cases, but results are uncertain."), "maybe"
        )

    def test_earliest_verdict_wins_when_both_present(self):
        # When both "yes" and "no" appear (and no hedge), the earlier one wins.
        self.assertEqual(
            evaluate.predict_label("No clear benefit, though yes for a subgroup."), "no"
        )

    def test_word_boundary_avoids_false_positive(self):
        # "another" contains "no" but must not be read as a "no" verdict, and
        # there is no real verdict here -> unknown.
        self.assertEqual(evaluate.predict_label("This is another finding entirely."), "unknown")


class TestComputeAccuracy(unittest.TestCase):
    """compute_accuracy scores only items where a label was predicted."""

    def test_all_correct(self):
        pairs = [("yes", "yes"), ("no", "no"), ("maybe", "maybe")]
        out = evaluate.compute_accuracy(pairs)
        self.assertEqual(out["accuracy"], 1.0)
        self.assertEqual(out["correct"], 3)
        self.assertEqual(out["scored"], 3)
        self.assertEqual(out["total"], 3)

    def test_half_correct(self):
        pairs = [("yes", "yes"), ("no", "yes")]  # one right, one wrong
        out = evaluate.compute_accuracy(pairs)
        self.assertEqual(out["accuracy"], 0.5)
        self.assertEqual(out["correct"], 1)
        self.assertEqual(out["scored"], 2)

    def test_unknown_predictions_are_not_scored(self):
        # "unknown" rows are excluded from the denominator, not counted wrong.
        pairs = [("yes", "yes"), ("unknown", "no"), ("unknown", "maybe")]
        out = evaluate.compute_accuracy(pairs)
        self.assertEqual(out["scored"], 1)      # only the "yes" row is scoreable
        self.assertEqual(out["correct"], 1)
        self.assertEqual(out["accuracy"], 1.0)  # 1/1 over scored items
        self.assertEqual(out["total"], 3)

    def test_no_scoreable_items_gives_zero(self):
        pairs = [("unknown", "yes"), ("unknown", "no")]
        out = evaluate.compute_accuracy(pairs)
        self.assertEqual(out["scored"], 0)
        self.assertEqual(out["accuracy"], 0.0)  # guarded division by zero

    def test_empty_input(self):
        out = evaluate.compute_accuracy([])
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["scored"], 0)
        self.assertEqual(out["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
