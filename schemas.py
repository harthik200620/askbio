"""Shared data shapes passed between modules.

ingest -> Snippet, retrieve -> Passage, generate -> AnswerResult.
"""
from __future__ import annotations

from typing import List, TypedDict


class Snippet(TypedDict):
    """One chunk of PubMed text (stored in corpus.jsonl)."""
    id: str
    pmid: str        # used to build the citation link
    title: str       # may be ""
    text: str


class Passage(TypedDict):
    """A Snippet plus a relevance score."""
    id: str
    pmid: str
    title: str
    text: str
    score: float     # reranker score for the final passages


class Citation(TypedDict):
    pmid: str
    title: str
    url: str         # PUBMED_URL.format(pmid=...)


class AnswerResult(TypedDict):
    """generate.py output: the answer plus its evidence."""
    answer: str
    citations: List[Citation]
    abstained: bool
    passages: List[Passage]  # passages actually used (UI + eval)
