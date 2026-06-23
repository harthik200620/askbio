"""Grounded answer generation: ask an LLM to answer from the retrieved passages
only, cite the PMIDs that back each claim, and abstain when the passages don't
cover the question (a wrong biomedical answer is worse than "I don't know").

Backends are dispatched on config.LLM_BACKEND. The "none" backend is a free,
no-key fallback that just quotes the top passages back with their PMID tags;
"gemini" is the free-via-AI-Studio way to get actual synthesised prose. The
openai/anthropic/gemini SDKs are imported inside their backend functions, so
this module and its pure logic import fine with none of them installed.
"""
from __future__ import annotations

import re
from typing import List

import config
from schemas import AnswerResult, Citation, Passage

# Citation token like [PMID:12345], whitespace-tolerant so "[ pmid: 123 ]" matches too.
_CITATION_RE = re.compile(r"\[\s*PMID\s*:\s*(\d+)\s*\]", re.IGNORECASE)


def build_prompt(query: str, passages: List[Passage]) -> tuple[str, str]:
    """Build the (system, user) prompt for a grounded, cited answer.

    Passages are numbered [1..n] and PMID-tagged so the model has a stable handle
    to cite. The system message spells out the grounding contract verbatim.
    """
    numbered = "\n".join(
        f"[{i}] (PMID:{p['pmid']}) {p['text']}"
        for i, p in enumerate(passages, start=1)
    )

    system = (
        "You are AskBio, a careful biomedical assistant. "
        "Answer the question using ONLY the information in the numbered passages "
        "below. Do not use any outside knowledge or prior training. "
        "Cite the PubMed IDs that support each statement inline, in the exact "
        "form [PMID:xxxx] (you may cite several). "
        "If the passages do not contain enough information to answer, reply with "
        f"EXACTLY this sentence and nothing else: {config.ABSTAIN_MESSAGE}"
    )

    user = (
        f"Question: {query}\n\n"
        f"Passages:\n{numbered}\n\n"
        "Answer (grounded in the passages above, with [PMID:xxxx] citations):"
    )
    return system, user


def extract_citations(answer: str, passages: List[Passage]) -> List[Citation]:
    """Pull [PMID:xxxx] tokens out of the answer, keeping only PMIDs that were
    actually in the passages.

    Validating against the passages drops PMIDs the model hallucinated -- a
    clickable citation that doesn't support the claim is worse than none. Dupes
    are removed, first mention wins, and the title comes from the passage.
    """
    # PMID -> title; first passage wins if a PMID repeats.
    pmid_to_title = {}
    for p in passages:
        pmid_to_title.setdefault(p["pmid"], p["title"])

    citations: List[Citation] = []
    seen: set[str] = set()
    for match in _CITATION_RE.finditer(answer):
        pmid = match.group(1)
        # Skip hallucinated PMIDs (not in context) and repeats.
        if pmid not in pmid_to_title or pmid in seen:
            continue
        seen.add(pmid)
        citations.append(
            Citation(
                pmid=pmid,
                title=pmid_to_title[pmid],
                url=config.PUBMED_URL.format(pmid=pmid),
            )
        )
    return citations


def _generate_none(passages: List[Passage]) -> str:
    """Free offline backend: quote the top 1-2 passages back with their PMID tags.

    No LLM, so we can't synthesise prose -- but the answer keeps the same
    grounded+cited shape as a real one, so the rest of the app and the tests run
    unchanged. No passages means nothing to answer from, so abstain.
    """
    if not passages:
        return config.ABSTAIN_MESSAGE

    parts: List[str] = []
    for p in passages[:2]:
        snippet = p["text"].strip()
        if len(snippet) > 300:
            snippet = snippet[:300].rstrip() + "..."
        parts.append(f"{snippet} [PMID:{p['pmid']}]")
    return " ".join(parts)


def _generate_openai(system: str, user: str) -> str:
    """OpenAI chat completions backend."""
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,  # keep it deterministic
    )
    return (response.choices[0].message.content or "").strip()


def _generate_anthropic(system: str, user: str) -> str:
    """Anthropic Messages backend."""
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.ANTHROPIC_LLM_MODEL,
        max_tokens=1024,
        system=system,  # system prompt is a top-level arg here, not a message
        messages=[{"role": "user", "content": user}],
        temperature=0,
    )
    # Response is a list of content blocks; keep the text ones.
    text = "".join(block.text for block in response.content if block.type == "text")
    return text.strip()


def _generate_gemini(system: str, user: str) -> str:
    """Google Gemini backend (google-genai SDK). Free via Google AI Studio."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_LLM_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
        ),
    )
    return (response.text or "").strip()


def _passages_too_weak(passages: List[Passage]) -> bool:
    """True when there are no passages, or the best rerank score is below
    config.RELEVANCE_THRESHOLD -- lets us abstain on off-topic questions instead
    of answering from passages that don't match. Threshold defaults to off
    (-1e9); the demo .env raises it.
    """
    if not passages:
        return True
    top_score = max(p.get("score", 0.0) for p in passages)
    return top_score < config.RELEVANCE_THRESHOLD


def generate_answer(query: str, passages: List[Passage]) -> AnswerResult:
    """Produce a grounded AnswerResult for the query.

    Dispatches on config.LLM_BACKEND for the raw text, then validates citations
    and sets the abstained flag the same way for every backend.
    """
    # Bail out before calling any backend if retrieval is too weak.
    if _passages_too_weak(passages):
        return AnswerResult(
            answer=config.ABSTAIN_MESSAGE,
            citations=[],
            abstained=True,
            passages=passages,
        )

    backend = config.LLM_BACKEND

    if backend == "none":
        answer = _generate_none(passages)
    else:
        system, user = build_prompt(query, passages)
        if backend == "openai":
            answer = _generate_openai(system, user)
        elif backend == "anthropic":
            answer = _generate_anthropic(system, user)
        elif backend == "gemini":
            answer = _generate_gemini(system, user)
        else:
            raise ValueError(
                f"Unknown LLM_BACKEND {backend!r}; expected "
                "'openai', 'anthropic', 'gemini', or 'none'."
            )

    citations = extract_citations(answer, passages)

    # Trim before comparing so trailing whitespace from an SDK doesn't hide an abstention.
    abstained = answer.strip() == config.ABSTAIN_MESSAGE.strip() or (
        backend == "none" and not passages
    )

    return AnswerResult(
        answer=answer,
        citations=citations,
        abstained=abstained,
        passages=passages,
    )


def _main() -> None:
    """CLI: python generate.py "a question" -> retrieve, answer, cite."""
    import sys

    if len(sys.argv) < 2:
        print('usage: python generate.py "your question"')
        raise SystemExit(1)

    query = sys.argv[1]

    # Imported here, not at module top: retrieve pulls in heavy ML deps (Qdrant,
    # rerankers) we don't want just to import this module's logic.
    import retrieve

    passages = retrieve.retrieve(query)
    result = generate_answer(query, passages)

    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
    if result["abstained"]:
        print("(abstained - not enough information in the retrieved literature)")
    if result["citations"]:
        print("\nCitations:")
        for c in result["citations"]:
            print(f"  PMID:{c['pmid']}  {c['url']}")


if __name__ == "__main__":
    _main()
