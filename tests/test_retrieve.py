"""Tests for the pure parts of retrieve.py: rrf_fuse and _tokenize.

Both are pure stdlib, so importing retrieve here never drags in qdrant_client /
sentence_transformers / rank_bm25 (those imports live inside the functions).

Run:  python -m unittest tests.test_retrieve   (from askbio/)
"""
from __future__ import annotations

import os
import sys
import unittest

# Let this file find retrieve.py when run directly, not just via -m unittest.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import retrieve  # noqa: E402


def _passage(doc_id: str, text: str = "x") -> dict:
    """Minimal Passage-shaped dict; score is a placeholder RRF overwrites."""
    return {"id": doc_id, "pmid": f"pmid-{doc_id}", "title": "", "text": text, "score": 0.0}


class TestTokenize(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(retrieve._tokenize("HeLLo WORLD"), ["hello", "world"])

    def test_splits_on_non_alphanumeric(self):
        self.assertEqual(
            retrieve._tokenize("COVID-19, vaccine!"),
            ["covid", "19", "vaccine"],
        )

    def test_keeps_alphanumeric_tokens_intact(self):
        # Gene-style tokens mixing letters+digits stay whole.
        self.assertEqual(retrieve._tokenize("TP53/BRCA1"), ["tp53", "brca1"])

    def test_no_empty_tokens_from_edges_or_runs(self):
        self.assertEqual(retrieve._tokenize("  --a___b--  "), ["a", "b"])

    def test_empty_string_yields_no_tokens(self):
        self.assertEqual(retrieve._tokenize(""), [])

    def test_underscore_is_a_separator(self):
        # Underscore splits here, unlike \w.
        self.assertEqual(retrieve._tokenize("alpha_beta"), ["alpha", "beta"])


class TestRrfFuse(unittest.TestCase):
    def test_consensus_doc_beats_single_list_leader(self):
        # B tops both lists, X tops only one -> B should win.
        list_a = [_passage("B"), _passage("A"), _passage("C")]
        list_b = [_passage("B"), _passage("X"), _passage("A")]
        fused = retrieve.rrf_fuse([list_a, list_b], k=10, top_k=10)
        self.assertEqual(fused[0]["id"], "B")
        ids = [p["id"] for p in fused]
        self.assertLess(ids.index("B"), ids.index("X"))

    def test_score_formula_is_zero_based_one_over_k_plus_rank(self):
        k = 60
        single = [_passage("A"), _passage("B"), _passage("C")]
        fused = retrieve.rrf_fuse([single], k=k, top_k=10)
        by_id = {p["id"]: p["score"] for p in fused}
        self.assertAlmostEqual(by_id["A"], 1.0 / (k + 0))
        self.assertAlmostEqual(by_id["B"], 1.0 / (k + 1))
        self.assertAlmostEqual(by_id["C"], 1.0 / (k + 2))

    def test_summation_across_lists(self):
        k = 5
        list_a = [_passage("A"), _passage("B")]
        list_b = [_passage("B"), _passage("A")]
        fused = retrieve.rrf_fuse([list_a, list_b], k=k, top_k=10)
        by_id = {p["id"]: p["score"] for p in fused}
        expected = 1.0 / (k + 0) + 1.0 / (k + 1)  # A and B each rank0 once, rank1 once
        self.assertAlmostEqual(by_id["A"], expected)
        self.assertAlmostEqual(by_id["B"], expected)

    def test_doc_in_one_list_only_still_scores(self):
        list_a = [_passage("A"), _passage("LONELY")]
        list_b = [_passage("A")]
        fused = retrieve.rrf_fuse([list_a, list_b], k=10, top_k=10)
        ids = [p["id"] for p in fused]
        self.assertIn("LONELY", ids)
        by_id = {p["id"]: p["score"] for p in fused}
        self.assertGreater(by_id["A"], by_id["LONELY"])
        self.assertGreater(by_id["LONELY"], 0.0)

    def test_dedupe_keeps_one_passage_per_id(self):
        list_a = [_passage("A", text="from-A")]
        list_b = [_passage("A", text="from-B")]
        fused = retrieve.rrf_fuse([list_a, list_b], k=10, top_k=10)
        ids = [p["id"] for p in fused]
        self.assertEqual(ids, ["A"])
        self.assertEqual(ids.count("A"), 1)
        # list_a scanned first, so its passage wins.
        self.assertEqual(fused[0]["text"], "from-A")

    def test_top_k_truncates_to_highest_scorers(self):
        single = [_passage("A"), _passage("B"), _passage("C"), _passage("D")]
        fused = retrieve.rrf_fuse([single], k=10, top_k=2)
        self.assertEqual(len(fused), 2)
        self.assertEqual([p["id"] for p in fused], ["A", "B"])

    def test_output_sorted_by_fused_score_descending(self):
        list_a = [_passage("LOW"), _passage("MID"), _passage("HIGH")]
        list_b = [_passage("HIGH"), _passage("MID")]
        fused = retrieve.rrf_fuse([list_a, list_b], k=10, top_k=10)
        scores = [p["score"] for p in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(fused[0]["id"], "HIGH")

    def test_does_not_mutate_input_passages(self):
        # rrf_fuse must not write back onto the caller's passages.
        original = _passage("A")
        original["score"] = 123.0
        retrieve.rrf_fuse([[original]], k=10, top_k=10)
        self.assertEqual(original["score"], 123.0)

    def test_empty_input_returns_empty(self):
        self.assertEqual(retrieve.rrf_fuse([], top_k=10), [])
        self.assertEqual(retrieve.rrf_fuse([[]], top_k=10), [])

    def test_defaults_come_from_config(self):
        single = [_passage(str(i)) for i in range(config.RRF_TOP_K + 5)]
        fused = retrieve.rrf_fuse([single])  # default top_k
        self.assertEqual(len(fused), config.RRF_TOP_K)


if __name__ == "__main__":
    unittest.main()
