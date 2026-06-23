"""
AskBio - shared data shapes (the 'contract' every module agrees on).

TypedDicts keep things lightweight and readable while documenting exactly what a
Snippet / Passage / AnswerResult looks like as data flows through the pipeline:

    ingest.py      -> Snippet      (raw cleaned chunk + PMID)
    retrieve.py    -> Passage      (a Snippet + a relevance score)
    generate.py    -> AnswerResult (answer text + verified citations)
"""
from __future__ import annotations

from typing import List, TypedDict


class Snippet(TypedDict):
    """One chunk of PubMed text (ingest.py output, stored in corpus.jsonl)."""
    id: str          # unique id for this chunk
    pmid: str        # PubMed ID, used to build the citation link
    title: str       # article title (may be "")
    text: str        # the snippet text used for retrieval + answering


class Passage(TypedDict):
    """A retrieved, scored snippet handed to the LLM (a Snippet + score)."""
    id: str
    pmid: str
    title: str
    text: str
    score: float     # relevance score (reranker score for the final passages)


class Citation(TypedDict):
    pmid: str
    title: str
    url: str         # PUBMED_URL.format(pmid=...)


class AnswerResult(TypedDict):
    """generate.py output: the grounded answer plus the evidence behind it."""
    answer: str
    citations: List[Citation]
    abstained: bool          # True if the system declined to answer
    passages: List[Passage]  # passages actually used (for the UI + eval)
