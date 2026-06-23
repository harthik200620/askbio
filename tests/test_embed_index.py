"""
Unit tests for embed_index.py pure helpers.

These tests use ONLY the standard library and deliberately exercise the helpers
that carry no heavy dependencies (_tokenize, _point_id, _batched). Importing
embed_index must NOT pull in torch / openai / qdrant / rank_bm25, because those
big libraries are imported lazily inside the functions that use them. So this
file is a guard on that contract too: if someone moves a heavy import to module
top-level, importing embed_index here would start failing.
"""
import os
import sys
import unittest

# Make the parent package dir (where embed_index.py lives) importable when this
# test is run from inside the tests/ folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import embed_index  # noqa: E402  (path tweak must precede this import)


class TestTokenize(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(embed_index._tokenize("BRCA Gene"), ["brca", "gene"])

    def test_splits_on_punctuation_and_symbols(self):
        # Hyphens, commas, slashes, parens -> all act as separators.
        self.assertEqual(
            embed_index._tokenize("TP53-mutant, p<0.05 (cohort/A)"),
            ["tp53", "mutant", "p", "0", "05", "cohort", "a"],
        )

    def test_collapses_runs_of_separators(self):
        # Multiple spaces/punctuation in a row must not produce empty tokens.
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
        # Same input -> same id, every call.
        a = embed_index._point_id("snippet-123")
        b = embed_index._point_id("snippet-123")
        self.assertEqual(a, b)

    def test_known_stable_value(self):
        # Pin the FNV-1a output so an accidental algorithm change is caught.
        # (FNV-1a 64-bit of the ASCII bytes of "abc".)
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
        # _batched must not require len(); a generator input proves that.
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
