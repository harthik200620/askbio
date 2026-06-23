"""
Tests for the ingest.py helpers (no network, no datasets dependency).

    python -m unittest tests.test_ingest      # from the askbio/ folder
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Make the askbio/ root importable when run as a bare file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ingest  # noqa: E402  (path tweak must happen before this import)


class TestToSnippet(unittest.TestCase):
    def test_content_key(self):
        rec = {"id": "a1", "content": "x" * 50, "title": "T"}
        snip = ingest._to_snippet(rec, 0)
        self.assertIsNotNone(snip)
        self.assertEqual(snip["text"], "x" * 50)
        self.assertEqual(snip["id"], "a1")
        self.assertEqual(snip["title"], "T")

    def test_contents_key(self):
        rec = {"id": "a2", "contents": "y" * 50}
        snip = ingest._to_snippet(rec, 0)
        self.assertEqual(snip["text"], "y" * 50)

    def test_text_key(self):
        rec = {"id": "a3", "text": "z" * 50}
        self.assertEqual(ingest._to_snippet(rec, 0)["text"], "z" * 50)

    def test_abstract_key_is_last_resort(self):
        rec = {"id": "a4", "abstract": "w" * 50}
        self.assertEqual(ingest._to_snippet(rec, 0)["text"], "w" * 50)

    def test_content_preferred_over_abstract(self):
        # When several text keys exist, first in priority order wins.
        rec = {"id": "a5", "content": "first" * 10, "abstract": "second" * 10}
        self.assertTrue(ingest._to_snippet(rec, 0)["text"].startswith("first"))

    def test_pmid_explicit_field(self):
        rec = {"id": "x", "text": "t" * 50, "pmid": "12345"}
        self.assertEqual(ingest._to_snippet(rec, 0)["pmid"], "12345")

    def test_pmid_uppercase_field(self):
        rec = {"id": "x", "text": "t" * 50, "PMID": "67890"}
        self.assertEqual(ingest._to_snippet(rec, 0)["pmid"], "67890")

    def test_pmid_integer_value(self):
        # MedRAG/pubmed stores PMID as an int; it has to be coerced and used, not
        # skipped in favor of the id's digits (which give the dump year, "23").
        rec = {"id": "pubmed23n0001_5", "text": "t" * 50, "PMID": 21}
        self.assertEqual(ingest._to_snippet(rec, 0)["pmid"], "21")

    def test_pmid_extracted_from_id(self):
        # No pmid field, so fall back to the first digit run in the id.
        rec = {"id": "pubmed23n0001_99887766", "text": "t" * 50}
        self.assertEqual(ingest._to_snippet(rec, 0)["pmid"], "23")

    def test_pmid_missing_everywhere(self):
        rec = {"id": "no-digits-here", "text": "t" * 50}
        self.assertEqual(ingest._to_snippet(rec, 0)["pmid"], "")

    def test_too_short_text_returns_none(self):
        rec = {"id": "x", "text": "tiny"}  # below _MIN_TEXT_CHARS
        self.assertIsNone(ingest._to_snippet(rec, 0))

    def test_missing_text_returns_none(self):
        rec = {"id": "x", "title": "only a title"}
        self.assertIsNone(ingest._to_snippet(rec, 0))

    def test_fallback_id_from_index(self):
        # No id -> "row{i}".
        rec = {"text": "t" * 50}
        self.assertEqual(ingest._to_snippet(rec, 7)["id"], "row7")

    def test_missing_title_becomes_empty_string(self):
        rec = {"id": "x", "text": "t" * 50}
        self.assertEqual(ingest._to_snippet(rec, 0)["title"], "")


class TestDedupeById(unittest.TestCase):
    def test_keeps_first_occurrence_in_order(self):
        snips = [
            {"id": "a", "pmid": "1", "title": "", "text": "first"},
            {"id": "b", "pmid": "2", "title": "", "text": "second"},
            {"id": "a", "pmid": "3", "title": "", "text": "dupe-dropped"},
        ]
        out = ingest._dedupe_by_id(snips)
        self.assertEqual([s["id"] for s in out], ["a", "b"])
        self.assertEqual(out[0]["text"], "first")  # first 'a' wins


class TestJsonlRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        snips = [
            {"id": "1", "pmid": "111", "title": "Alpha", "text": "hello world"},
            {"id": "2", "pmid": "222", "title": "Beta", "text": "ünïcode ✓"},
        ]
        fd, name = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        path = Path(name)
        try:
            ingest._write_jsonl(snips, path)
            self.assertEqual(ingest._count_lines(path), 2)
            self.assertEqual(ingest._read_jsonl(path), snips)
            # limit truncates the read.
            self.assertEqual(ingest._read_jsonl(path, limit=1), snips[:1])
        finally:
            path.unlink()

    def test_count_lines_missing_file_is_zero(self):
        missing = Path(tempfile.gettempdir()) / "definitely_not_here_askbio.jsonl"
        self.assertEqual(ingest._count_lines(missing), 0)


if __name__ == "__main__":
    unittest.main()
