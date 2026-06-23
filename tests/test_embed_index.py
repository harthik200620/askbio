"""Stdlib-only tests for embed_index's pure helpers.

Also doubles as a guard that importing embed_index doesn't drag in
torch/openai/qdrant/rank_bm25 -- if one of those moves to module top-level, the
import below starts failing.
"""
import os
import sys
import unittest

# Put the dir holding embed_index.py on the path so this runs from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import embed_index  # noqa: E402


class TestTokenize(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(embed_index._tokenize("BRCA Gene"), ["brca", "gene"])

    def test_splits_on_punctuation_and_symbols(self):
        self.assertEqual(
            embed_index._tokenize("TP53-mutant, p<0.05 (cohort/A)"),
            ["tp53", "mutant", "p", "0", "05", "cohort", "a"],
        )

    def test_collapses_runs_of_separators(self):
        # Runs of separators shouldn't leak empty tokens.
        self.assertEqual(
            embed_index._tokenize("  multiple   spaces  "),
            ["multiple", "spaces"],
        )

    def test_keeps_digits_and_alphanumerics(self):
        self.assertEqual(embed_index._tokenize("covid19"), ["covid19"])

    def test_empty_string_returns_empty_list(self):
        self.assertEqual(embed_index._tokenize(""), [])

    def test_only_punctuation_returns_empty_list(self):
        self.assertEqual(embed_index._tokenize("!!! --- ???"), [])


class TestPointId(unittest.TestCase):
    def test_is_deterministic_across_calls(self):
        a = embed_index._point_id("snippet-123")
        b = embed_index._point_id("snippet-123")
        self.assertEqual(a, b)

    def test_known_stable_value(self):
        # Pinned FNV-1a of "abc" -- catches any accidental algorithm change.
        self.assertEqual(embed_index._point_id("abc"), 16654208175385433931)

    def test_different_inputs_give_different_ids(self):
        self.assertNotEqual(
            embed_index._point_id("id-a"), embed_index._point_id("id-b")
        )

    def test_is_non_negative_int(self):
        pid = embed_index._point_id("anything")
        self.assertIsInstance(pid, int)
        self.assertGreaterEqual(pid, 0)


class TestBatched(unittest.TestCase):
    def test_even_split(self):
        self.assertEqual(
            list(embed_index._batched([1, 2, 3, 4], 2)),
            [[1, 2], [3, 4]],
        )

    def test_includes_shorter_final_remainder(self):
        self.assertEqual(
            list(embed_index._batched([1, 2, 3, 4, 5], 2)),
            [[1, 2], [3, 4], [5]],
        )

    def test_batch_larger_than_iterable(self):
        self.assertEqual(list(embed_index._batched([1, 2], 10)), [[1, 2]])

    def test_empty_iterable_yields_nothing(self):
        self.assertEqual(list(embed_index._batched([], 3)), [])

    def test_works_on_a_generator(self):
        # Generator input proves _batched doesn't lean on len().
        gen = (i for i in range(5))
        self.assertEqual(
            list(embed_index._batched(gen, 2)),
            [[0, 1], [2, 3], [4]],
        )

    def test_batch_size_below_one_raises(self):
        with self.assertRaises(ValueError):
            list(embed_index._batched([1, 2, 3], 0))


if __name__ == "__main__":
    unittest.main()
