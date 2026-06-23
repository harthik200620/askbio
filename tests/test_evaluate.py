"""
Stdlib-only tests for the two pure helpers in evaluate.py (predict_label and
compute_accuracy). No network/LLM/ragas/datasets/matplotlib -- importing
evaluate is safe because its heavy deps are imported lazily.

    python -m unittest tests.test_evaluate      # from the askbio/ folder
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make askbio/ importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate  # noqa: E402  (path tweak must happen before this import)


class TestPredictLabel(unittest.TestCase):

    def test_clear_yes(self):
        self.assertEqual(evaluate.predict_label("Yes, the treatment is effective."), "yes")

    def test_clear_no(self):
        self.assertEqual(evaluate.predict_label("No, there is no such association."), "no")

    def test_clear_maybe_from_hedge(self):
        self.assertEqual(
            evaluate.predict_label("The evidence is inconclusive and unclear."), "maybe"
        )

    def test_maybe_word_directly(self):
        self.assertEqual(evaluate.predict_label("Maybe, more study is needed."), "maybe")

    def test_abstain_answer_is_unknown(self):
        self.assertEqual(evaluate.predict_label(evaluate.config.ABSTAIN_MESSAGE), "unknown")

    def test_empty_answer_is_unknown(self):
        self.assertEqual(evaluate.predict_label(""), "unknown")
        self.assertEqual(evaluate.predict_label("   "), "unknown")

    def test_no_signal_is_unknown(self):
        self.assertEqual(
            evaluate.predict_label("The study enrolled 200 patients across three sites."),
            "unknown",
        )

    def test_hedge_beats_stray_yes(self):
        self.assertEqual(
            evaluate.predict_label("Yes in some cases, but results are uncertain."), "maybe"
        )

    def test_earliest_verdict_wins_when_both_present(self):
        self.assertEqual(
            evaluate.predict_label("No clear benefit, though yes for a subgroup."), "no"
        )

    def test_word_boundary_avoids_false_positive(self):
        # "another" contains "no" but isn't a verdict.
        self.assertEqual(evaluate.predict_label("This is another finding entirely."), "unknown")


class TestComputeAccuracy(unittest.TestCase):

    def test_all_correct(self):
        pairs = [("yes", "yes"), ("no", "no"), ("maybe", "maybe")]
        out = evaluate.compute_accuracy(pairs)
        self.assertEqual(out["accuracy"], 1.0)
        self.assertEqual(out["correct"], 3)
        self.assertEqual(out["scored"], 3)
        self.assertEqual(out["total"], 3)

    def test_half_correct(self):
        pairs = [("yes", "yes"), ("no", "yes")]
        out = evaluate.compute_accuracy(pairs)
        self.assertEqual(out["accuracy"], 0.5)
        self.assertEqual(out["correct"], 1)
        self.assertEqual(out["scored"], 2)

    def test_unknown_predictions_are_not_scored(self):
        pairs = [("yes", "yes"), ("unknown", "no"), ("unknown", "maybe")]
        out = evaluate.compute_accuracy(pairs)
        self.assertEqual(out["scored"], 1)
        self.assertEqual(out["correct"], 1)
        self.assertEqual(out["accuracy"], 1.0)
        self.assertEqual(out["total"], 3)

    def test_no_scoreable_items_gives_zero(self):
        pairs = [("unknown", "yes"), ("unknown", "no")]
        out = evaluate.compute_accuracy(pairs)
        self.assertEqual(out["scored"], 0)
        self.assertEqual(out["accuracy"], 0.0)

    def test_empty_input(self):
        out = evaluate.compute_accuracy([])
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["scored"], 0)
        self.assertEqual(out["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
