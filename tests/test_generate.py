"""
Unit tests for generate.py (stdlib ``unittest`` only - NO real API calls).

These exercise the three interview-critical guardrails using only the pure,
offline code paths:
  * the grounding prompt is correctly numbered + instructed,
  * citation validation drops hallucinated PMIDs,
  * the offline "none" backend abstains on empty input and cites on real input.

We force ``config.LLM_BACKEND = "none"`` for the backend tests so they never
depend on the environment or reach the network. The openai/anthropic SDKs are
imported lazily inside generate.py, so importing it here needs no extra deps.
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
    """Tiny helper to build a Passage without repeating the keys everywhere."""
    return Passage(id=f"id-{pmid}", pmid=pmid, title=title, text=text, score=score)


class BuildPromptTest(unittest.TestCase):
    def test_passages_numbered_and_pmids_present(self):
        passages = [
            _passage("111", "Aspirin reduces fever."),
            _passage("222", "Ibuprofen reduces inflammation."),
        ]
        system, user = generate.build_prompt("What reduces fever?", passages)

        # Passages are numbered [1..n] and each carries its PMID.
        self.assertIn("[1]", user)
        self.assertIn("[2]", user)
        self.assertIn("PMID:111", user)
        self.assertIn("PMID:222", user)
        # The user prompt should carry the actual question.
        self.assertIn("What reduces fever?", user)

    def test_system_states_grounding_contract(self):
        _system, _user = generate.build_prompt("q", [_passage("1", "text")])
        system_lower = _system.lower()
        # Grounding: "only" the passages, no outside knowledge.
        self.assertIn("only", system_lower)
        # Abstention: the exact opt-out sentence must be embedded in the system.
        self.assertIn(config.ABSTAIN_MESSAGE, _system)


class ExtractCitationsTest(unittest.TestCase):
    def test_keeps_valid_drops_hallucinated_pmid(self):
        # 111 is in the passages (valid); 999 is not (hallucinated -> dropped).
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
        # Force the free offline backend so these tests never hit an API.
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
        # The extractive answer must carry a real, in-context PMID tag...
        self.assertIn("[PMID:12345]", result["answer"])
        # ...and that PMID must survive citation validation.
        self.assertEqual([c["pmid"] for c in result["citations"]], ["12345"])
        # passages are echoed back on the result for the UI / eval.
        self.assertEqual(result["passages"], passages)


if __name__ == "__main__":
    unittest.main()
