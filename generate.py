"""
AskBio - Phase 4: grounded answer generation (with citations + abstention).

Plain-English idea
------------------
This is the part of the RAG system that actually *answers*. Given a user's
question and the handful of passages retrieval found, it asks an LLM to write a
short answer that is **grounded** - i.e. built ONLY from those passages, never
from the model's own background knowledge - and to cite the PubMed IDs (PMIDs)
that back each claim inline as ``[PMID:1234]``. If the passages don't actually
contain the answer, the model is told to say so verbatim (``config.ABSTAIN_MESSAGE``)
rather than guess. In a biomedical setting a confident-but-wrong answer is worse
than "I don't know", so abstention is a feature, not a failure.

Why this matters (the three guardrails)
---------------------------------------
1. **Grounding prompt** - the system instruction forbids outside knowledge and
   pins every answer to the numbered passages. This is the first line of defence
   against hallucination.
2. **Citation validation** - LLMs sometimes cite a plausible-looking PMID that
   was never in the context. After generation we regex out every ``[PMID:xxxx]``
   token and *drop any PMID that is not among the passages we actually supplied*.
   A citation the user can click but that doesn't support the claim is worse than
   none, so we only keep verifiable ones. This is the citation-validity guardrail.
3. **Abstention** - we surface an explicit ``abstained`` flag (the answer equals
   ``config.ABSTAIN_MESSAGE``, or the offline backend had no passages) so the UI
   and the evaluator can treat "declined to answer" distinctly from a real answer.

Backends (``config.LLM_BACKEND``)
---------------------------------
- ``"openai"`` / ``"anthropic"`` - real chat completions.
- ``"none"`` - a free, no-API **extractive demo**: it stitches an answer out of
  the top passages' own text and appends their PMID tags. This lets the entire
  app run end-to-end at $0 with no key.

The ``openai`` / ``anthropic`` SDKs are imported *lazily inside their backend
functions* so this module - and all of its pure logic (prompt building, citation
parsing/validation, the offline backend) - imports and unit-tests cleanly with
neither library installed and no API key set.
"""
from __future__ import annotations

import re
from typing import List

import config
from schemas import AnswerResult, Citation, Passage

# Matches an inline citation token like ``[PMID:12345]`` and captures the digits.
# Case-insensitive and whitespace-tolerant so we still catch ``[ pmid: 123 ]``.
_CITATION_RE = re.compile(r"\[\s*PMID\s*:\s*(\d+)\s*\]", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Prompt building (pure - no API, no I/O)
# --------------------------------------------------------------------------- #
def build_prompt(query: str, passages: List[Passage]) -> tuple[str, str]:
    """
    Build the ``(system, user)`` prompt pair for a grounded, cited answer.

    Passages are numbered ``[1..n]`` and each is tagged with its PMID, e.g.::

        [1] (PMID:12345) <text>

    The number gives the model a stable handle to reason about; the PMID is what
    it must echo back in ``[PMID:xxxx]`` citations. The system message states the
    grounding contract (answer only from the passages, cite PMIDs, abstain with
    an exact phrase, no outside knowledge) - that wording is the primary
    anti-hallucination guardrail, so it is deliberately explicit.
    """
    # One numbered, PMID-tagged line per passage. The PMID appears both so the
    # model can cite it and so a human reading the prompt can audit grounding.
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


# --------------------------------------------------------------------------- #
# Citation parsing + validation (pure - the citation-validity guardrail)
# --------------------------------------------------------------------------- #
def extract_citations(answer: str, passages: List[Passage]) -> List[Citation]:
    """
    Pull ``[PMID:xxxx]`` tokens out of ``answer`` and keep only the *valid* ones.

    "Valid" means the PMID was actually among the passages we handed the model:
    anything else is a hallucinated citation and is dropped. We de-duplicate
    while preserving first-mention order, and look up each kept PMID's title from
    its passage so the UI can show a human-readable, clickable reference.
    """
    # Map PMID -> title for fast membership tests and title lookup. If the same
    # PMID appears twice, the first passage's title wins (consistent + cheap).
    pmid_to_title = {}
    for p in passages:
        pmid_to_title.setdefault(p["pmid"], p["title"])

    citations: List[Citation] = []
    seen: set[str] = set()
    for match in _CITATION_RE.finditer(answer):
        pmid = match.group(1)
        # Guardrail: skip PMIDs not in context (hallucinated) and repeats.
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


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _generate_none(passages: List[Passage]) -> str:
    """
    Offline, no-API **extractive** answer (the free ``"none"`` backend).

    With no LLM available we can't synthesise prose, so we do the honest minimum:
    quote the top 1-2 passages' text back and append their ``[PMID:xxxx]`` tags.
    This keeps the same grounded+cited *shape* as a real answer (so the rest of
    the app and the tests exercise the real code path) while costing nothing.
    Empty passages -> abstain, because there is genuinely nothing to answer from.
    """
    if not passages:
        return config.ABSTAIN_MESSAGE

    parts: List[str] = []
    for p in passages[:2]:  # top 1-2 passages keep the demo answer short
        snippet = p["text"].strip()
        # Trim to a sentence-ish length so the demo answer stays readable.
        if len(snippet) > 300:
            snippet = snippet[:300].rstrip() + "..."
        parts.append(f"{snippet} [PMID:{p['pmid']}]")
    return " ".join(parts)


def _generate_openai(system: str, user: str) -> str:
    """Call OpenAI chat completions. SDK imported lazily so import stays cheap."""
    from openai import OpenAI  # lazy: only needed for this backend

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,  # deterministic, grounded answers - no creative drift
    )
    return (response.choices[0].message.content or "").strip()


def _generate_anthropic(system: str, user: str) -> str:
    """Call the Anthropic Messages API. SDK imported lazily."""
    import anthropic  # lazy: only needed for this backend

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.ANTHROPIC_LLM_MODEL,
        max_tokens=1024,
        system=system,  # Anthropic takes the system prompt as a top-level arg
        messages=[{"role": "user", "content": user}],
        temperature=0,
    )
    # Messages return a list of content blocks; concatenate the text ones.
    text = "".join(block.text for block in response.content if block.type == "text")
    return text.strip()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_answer(query: str, passages: List[Passage]) -> AnswerResult:
    """
    Produce a grounded ``AnswerResult`` for ``query`` over ``passages``.

    Dispatches on ``config.LLM_BACKEND`` to get the raw answer text, then applies
    the two post-hoc guardrails uniformly across every backend:
      * validate citations (drop PMIDs not in the passages), and
      * compute ``abstained`` (True when the answer is the abstain message, or
        the offline backend had no passages to work from).
    """
    backend = config.LLM_BACKEND

    if backend == "none":
        # Offline demo path: build the answer straight from passage text.
        answer = _generate_none(passages)
    else:
        # Real LLM path: build the grounded prompt, then call the chosen SDK.
        system, user = build_prompt(query, passages)
        if backend == "openai":
            answer = _generate_openai(system, user)
        elif backend == "anthropic":
            answer = _generate_anthropic(system, user)
        else:
            raise ValueError(
                f"Unknown LLM_BACKEND {backend!r}; expected "
                "'openai', 'anthropic', or 'none'."
            )

    citations = extract_citations(answer, passages)

    # Abstained when the model returned the exact opt-out phrase, or the offline
    # backend had nothing to answer from. Compare trimmed so trailing whitespace
    # from an SDK never hides a genuine abstention.
    abstained = answer.strip() == config.ABSTAIN_MESSAGE.strip() or (
        backend == "none" and not passages
    )

    return AnswerResult(
        answer=answer,
        citations=citations,
        abstained=abstained,
        passages=passages,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main() -> None:
    """CLI: ``python generate.py "a question"`` -> retrieve, answer, cite."""
    import sys

    if len(sys.argv) < 2:
        print('usage: python generate.py "your question"')
        raise SystemExit(1)

    query = sys.argv[1]

    # Lazy import: retrieve.py pulls in heavy ML deps (Qdrant, rerankers); we only
    # want them for the live CLI, not when this module is imported for its logic.
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
