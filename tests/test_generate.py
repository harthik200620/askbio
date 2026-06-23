"""Unit tests for generate.py -- stdlib unittest, no network.

Covers the prompt building, citation validation (hallucinated PMIDs dropped),
and the offline "none" backend (abstain on empty, cite on real input). Backend
tests force config.LLM_BACKEND = "none" so they never reach an API.
"""
from __future__ import annotations

import os
import sys
import unittest

# Make the project root importable when tests are run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import generate
from schemas import Passage


def _passage(pmid: str, text: str, title: str = "", score: float = 1.0) -> Passage:
    """Build a Passage without spelling out every key."""
    return Passage(id=f"id-{pmid}", pmid=pmid, title=title, text=text, score=score)


class BuildPromptTest(unittest.TestCase):
    def test_passages_numbered_and_pmids_present(self):
        passages = [
            _passage("111", "Aspirin reduces fever."),
            _passage("222", "Ibuprofen reduces inflammation."),
        ]
        system, user = generate.build_prompt("What reduces fever?", passages)

        # Numbered [1..n], each tagged with its PMID, and the question is in there.
        self.assertIn("[1]", user)
        self.assertIn("[2]", user)
        self.assertIn("PMID:111", user)
        self.assertIn("PMID:222", user)
        self.assertIn("What reduces fever?", user)

    def test_system_states_grounding_contract(self):
        _system, _user = generate.build_prompt("q", [_passage("1", "text")])
        system_lower = _system.lower()
        # "only" the passages, plus the exact abstain sentence.
        self.assertIn("only", system_lower)
        self.assertIn(config.ABSTAIN_MESSAGE, _system)


class ExtractCitationsTest(unittest.TestCase):
    def test_keeps_valid_drops_hallucinated_pmid(self):
        # 111 is in the passages; 999 isn't, so it should be dropped.
        passages = [_passage("111", "Some grounded fact.", title="Fever Study")]
        answer = "Aspirin helps [PMID:111]. Also unrelated [PMID:999]."

        citations = generate.extract_citations(answer, passages)

        self.assertEqual(len(citations), 1)
        only = citations[0]
        self.assertEqual(only["pmid"], "111")
        self.assertEqual(only["title"], "Fever Study")
        self.assertEqual(only["url"], config.PUBMED_URL.format(pmid="111"))

    def test_deduplicates_repeated_valid_pmid(self):
        passages = [_passage("111", "fact")]
        answer = "First [PMID:111]. Again [PMID:111]."
        citations = generate.extract_citations(answer, passages)
        self.assertEqual(len(citations), 1)


class NoneBackendTest(unittest.TestCase):
    def setUp(self):
        # Offline backend so these never hit an API.
        self._saved_backend = config.LLM_BACKEND
        config.LLM_BACKEND = "none"

    def tearDown(self):
        config.LLM_BACKEND = self._saved_backend

    def test_empty_passages_abstains(self):
        result = generate.generate_answer("anything", [])
        self.assertTrue(result["abstained"])
        self.assertEqual(result["answer"], config.ABSTAIN_MESSAGE)
        self.assertEqual(result["citations"], [])

    def test_nonempty_passages_answers_with_real_pmid(self):
        passages = [_passage("12345", "Metformin lowers blood glucose.")]
        result = generate.generate_answer("How does metformin work?", passages)

        self.assertFalse(result["abstained"])
        # Answer carries the in-context PMID, and it survives validation.
        self.assertIn("[PMID:12345]", result["answer"])
        self.assertEqual([c["pmid"] for c in result["citations"]], ["12345"])
        # passages echoed back for the UI / eval.
        self.assertEqual(result["passages"], passages)


class RelevanceGuardrailTest(unittest.TestCase):
    """Abstain when the best passage scores below the threshold."""

    def setUp(self):
        self._saved_backend = config.LLM_BACKEND
        self._saved_threshold = config.RELEVANCE_THRESHOLD
        config.LLM_BACKEND = "none"

    def tearDown(self):
        config.LLM_BACKEND = self._saved_backend
        config.RELEVANCE_THRESHOLD = self._saved_threshold

    def test_weak_top_score_abstains(self):
        # Passage exists but scores below threshold -> abstain anyway.
        config.RELEVANCE_THRESHOLD = 0.0
        passages = [_passage("111", "Unrelated abstract.", score=-5.0)]
        result = generate.generate_answer("what is dolo 650", passages)
        self.assertTrue(result["abstained"])
        self.assertEqual(result["answer"], config.ABSTAIN_MESSAGE)
        self.assertEqual(result["citations"], [])

    def test_strong_top_score_answers(self):
        # Clears the threshold -> answered normally.
        config.RELEVANCE_THRESHOLD = 0.0
        passages = [_passage("222", "Aspirin reduces fever and pain.", score=3.5)]
        result = generate.generate_answer("does aspirin reduce fever?", passages)
        self.assertFalse(result["abstained"])
        self.assertIn("[PMID:222]", result["answer"])


if __name__ == "__main__":
    unittest.main()
