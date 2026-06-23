"""
Unit tests for retrieve.py - the pure, deterministic parts only.

We test ``rrf_fuse`` (the interview-critical fusion maths) and ``_tokenize``
(which must match embed_index.py's contract). Both are pure stdlib functions, so
these tests import ``retrieve`` without ever touching qdrant_client,
sentence_transformers, rank_bm25, or any built index - exactly because retrieve.py
imports those lazily inside the functions that need them.

Run with:  python -m unittest tests.test_retrieve        (from the askbio/ dir)
       or:  python -m unittest discover -s tests
"""
from __future__ import annotations

import os
import sys
import unittest

# Make the package root (the dir containing retrieve.py) importable when this
# file is run directly, not just via `-m unittest` from the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402  (import after sys.path tweak)
import retrieve  # noqa: E402


def _passage(doc_id: str, text: str = "x") -> dict:
    """Tiny synthetic Passage-shaped dict (score is a placeholder RRF rewrites)."""
    return {"id": doc_id, "pmid": f"pmid-{doc_id}", "title": "", "text": text, "score": 0.0}


class TestTokenize(unittest.TestCase):
    """`_tokenize` must be lowercase + split-on-non-alphanumeric, no empties."""

    def test_lowercases(self):
        self.assertEqual(retrieve._tokenize("HeLLo WORLD"), ["hello", "world"])

    def test_splits_on_non_alphanumeric(self):
        # Hyphens, slashes, punctuation and whitespace are all separators; the
        # digits inside a token survive (alphanumeric = letters AND digits).
        self.assertEqual(
            retrieve._tokenize("COVID-19, vaccine!"),
            ["covid", "19", "vaccine"],
        )

    def test_keeps_alphanumeric_tokens_intact(self):
        # Gene/protein style tokens mixing letters+digits stay as one token.
        self.assertEqual(retrieve._tokenize("TP53/BRCA1"), ["tp53", "brca1"])

    def test_no_empty_tokens_from_edges_or_runs(self):
        # Leading/trailing separators and runs of them must not yield "" tokens.
        self.assertEqual(retrieve._tokenize("  --a___b--  "), ["a", "b"])

    def test_empty_string_yields_no_tokens(self):
        self.assertEqual(retrieve._tokenize(""), [])

    def test_underscore_is_a_separator(self):
        # Underscore is NOT alphanumeric, so it splits (unlike Python's \w).
        self.assertEqual(retrieve._tokenize("alpha_beta"), ["alpha", "beta"])


class TestRrfFuse(unittest.TestCase):
    """Reciprocal Rank Fusion: 0-based ranks, contribution 1/(k + rank)."""

    def test_consensus_doc_beats_single_list_leader(self):
        """
        The core property of RRF for hybrid search: a doc ranked decently in
        BOTH lists should outrank a doc sitting at the very top of only ONE.

        list A: A, B        list B: B, A
        With k small, B is rank0 in B and rank1 in A; A is rank1 in B, rank0 in A
        -> they tie. To break symmetry, give B a top spot in both lists:
        """
        list_a = [_passage("B"), _passage("A"), _passage("C")]
        list_b = [_passage("B"), _passage("X"), _passage("A")]
        # B: rank0 + rank0 ; A: rank1 + rank2 ; so B must come first.
        fused = retrieve.rrf_fuse([list_a, list_b], k=10, top_k=10)
        self.assertEqual(fused[0]["id"], "B")
        # And the consensus doc B should beat the single-list leader X.
        ids = [p["id"] for p in fused]
        self.assertLess(ids.index("B"), ids.index("X"))

    def test_score_formula_is_zero_based_one_over_k_plus_rank(self):
        """A single list: scores must equal 1/(k+0), 1/(k+1), 1/(k+2), ..."""
        k = 60
        single = [_passage("A"), _passage("B"), _passage("C")]
        fused = retrieve.rrf_fuse([single], k=k, top_k=10)
        by_id = {p["id"]: p["score"] for p in fused}
        self.assertAlmostEqual(by_id["A"], 1.0 / (k + 0))
        self.assertAlmostEqual(by_id["B"], 1.0 / (k + 1))
        self.assertAlmostEqual(by_id["C"], 1.0 / (k + 2))

    def test_summation_across_lists(self):
        """A doc in two lists gets the SUM of its per-list contributions."""
        k = 5
        list_a = [_passage("A"), _passage("B")]          # A: rank0, B: rank1
        list_b = [_passage("B"), _passage("A")]          # B: rank0, A: rank1
        fused = retrieve.rrf_fuse([list_a, list_b], k=k, top_k=10)
        by_id = {p["id"]: p["score"] for p in fused}
        expected = 1.0 / (k + 0) + 1.0 / (k + 1)         # same for both A and B
        self.assertAlmostEqual(by_id["A"], expected)
        self.assertAlmostEqual(by_id["B"], expected)

    def test_doc_in_one_list_only_still_scores(self):
        """A doc appearing in just one list still gets a (smaller) score."""
        list_a = [_passage("A"), _passage("LONELY")]
        list_b = [_passage("A")]
        fused = retrieve.rrf_fuse([list_a, list_b], k=10, top_k=10)
        ids = [p["id"] for p in fused]
        self.assertIn("LONELY", ids)
        by_id = {p["id"]: p["score"] for p in fused}
        # A (in both) must outscore LONELY (in one), and LONELY must be > 0.
        self.assertGreater(by_id["A"], by_id["LONELY"])
        self.assertGreater(by_id["LONELY"], 0.0)

    def test_dedupe_keeps_one_passage_per_id(self):
        """Same id across lists collapses to a single fused passage."""
        list_a = [_passage("A", text="from-A")]
        list_b = [_passage("A", text="from-B")]
        fused = retrieve.rrf_fuse([list_a, list_b], k=10, top_k=10)
        ids = [p["id"] for p in fused]
        self.assertEqual(ids, ["A"])                     # exactly one, not two
        self.assertEqual(ids.count("A"), 1)
        # First-seen object wins the dedupe (list_a scanned first).
        self.assertEqual(fused[0]["text"], "from-A")

    def test_top_k_truncates_to_highest_scorers(self):
        """top_k returns only the best N, dropping the rest."""
        single = [_passage("A"), _passage("B"), _passage("C"), _passage("D")]
        fused = retrieve.rrf_fuse([single], k=10, top_k=2)
        self.assertEqual(len(fused), 2)
        self.assertEqual([p["id"] for p in fused], ["A", "B"])  # the two highest

    def test_output_sorted_by_fused_score_descending(self):
        """Whatever the input order, output is sorted by fused score desc."""
        list_a = [_passage("LOW"), _passage("MID"), _passage("HIGH")]
        list_b = [_passage("HIGH"), _passage("MID")]
        fused = retrieve.rrf_fuse([list_a, list_b], k=10, top_k=10)
        scores = [p["score"] for p in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(fused[0]["id"], "HIGH")         # in both, best combined

    def test_does_not_mutate_input_passages(self):
        """Purity guard: caller's passage objects keep their original score."""
        original = _passage("A")
        original["score"] = 123.0
        retrieve.rrf_fuse([[original]], k=10, top_k=10)
        self.assertEqual(original["score"], 123.0)       # untouched

    def test_empty_input_returns_empty(self):
        self.assertEqual(retrieve.rrf_fuse([], top_k=10), [])
        self.assertEqual(retrieve.rrf_fuse([[]], top_k=10), [])

    def test_defaults_come_from_config(self):
        """Sanity: the function's defaults are the config-driven ones."""
        single = [_passage(str(i)) for i in range(config.RRF_TOP_K + 5)]
        fused = retrieve.rrf_fuse([single])              # use default top_k
        self.assertEqual(len(fused), config.RRF_TOP_K)


if __name__ == "__main__":
    unittest.main()
