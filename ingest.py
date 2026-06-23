"""
AskBio - Phase 1: corpus ingestion.

Plain-English idea
------------------
A RAG system can only answer from text it has actually read. This module is that
"reading" step: it pulls a slice of PubMed snippets from Hugging Face
(``MedRAG/pubmed``), normalises each raw record into our tiny ``Snippet`` shape
(``id / pmid / title / text``), throws away junk (empty or too-short text,
duplicate ids), and saves the survivors to a local JSONL file
(``config.CORPUS_PATH``) - one JSON object per line.

Why JSONL + a cache? Streaming a public dataset is slow and flaky, so we do it
*once* and write the result to disk. The job is **resumable**: if the cache
already holds enough rows we skip the network entirely and just read the file
back. Later phases (embedding, BM25, retrieval) all start from this same file,
so ingestion is the single source of truth for "what text exists".

Defensive mapping: MedRAG records do not have stable column names (text might
live under ``content``, ``contents``, ``text`` or ``abstract``; the PMID might
be a field or be buried inside the ``id`` string). ``_to_snippet`` tries each
known location in turn so we are not brittle to schema drift.

Heavy dependency (``datasets``) is imported *inside* the functions that need it,
so this module - and its unit tests - import cleanly even when that library is
not installed.
"""
from __future__ import annotations

import json
import re
from typing import Iterator, Optional

import config
from schemas import Snippet

# Minimum characters of text for a snippet to be worth keeping. Anything shorter
# is almost certainly a fragment/label, not a usable passage.
_MIN_TEXT_CHARS = 20

# Matches the first run of digits in a string (used to dig a PMID out of an id
# like "pubmed23n0001_12345").
_DIGITS = re.compile(r"\d+")


def _first_nonempty(*values: object) -> str:
    """Return the first value that is a non-empty/non-whitespace string."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _topic_keywords() -> list[str]:
    """Lowercased keywords from config.CORPUS_TOPIC (comma/space separated)."""
    return [w.lower() for w in config.CORPUS_TOPIC.replace(",", " ").split() if w.strip()]


def _matches_topic(snippet: Snippet, keywords: list[str]) -> bool:
    """True if the snippet mentions ANY keyword (case-insensitive). With no
    keywords every snippet matches - that is the unfiltered, spec-default path."""
    if not keywords:
        return True
    haystack = (snippet["title"] + " " + snippet["text"]).lower()
    return any(kw in haystack for kw in keywords)


def _extract_pmid(record: dict) -> str:
    """
    Find the PubMed ID for a record, trying the most reliable source first:
      1. an explicit ``PMID`` / ``pmid`` field,
      2. otherwise the first run of digits inside the ``id`` string
         (MedRAG ids frequently embed the PMID).
    Falls back to "" when nothing numeric is present.
    """
    # PMID may be stored as an int (e.g. 21) or a string; coerce before the
    # emptiness check so a numeric PMID is not mistaken for "missing" and we
    # don't wrongly fall through to the id (which embeds the dump year, e.g.
    # "pubmed23n0001" -> "23").
    for key in ("PMID", "pmid"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    match = _DIGITS.search(str(record.get("id", "")))
    return match.group(0) if match else ""


def _to_snippet(record: dict, i: int) -> Optional[Snippet]:
    """
    Map ONE raw dataset record to a ``Snippet``, defensively.

    ``i`` is the row index, used only to mint a stable fallback id when the
    record has none. Returns ``None`` (caller drops the row) when the text is
    missing or too short to be useful for retrieval.
    """
    # Text may hide under any of these keys depending on the dataset dump.
    text = _first_nonempty(
        record.get("content"),
        record.get("contents"),
        record.get("text"),
        record.get("abstract"),
    )
    if len(text) < _MIN_TEXT_CHARS:
        return None  # nothing worth indexing

    # str(...) guards against numeric ids; "row{i}" keeps ids unique if absent.
    snippet_id = str(record.get("id") or f"row{i}")
    return Snippet(
        id=snippet_id,
        pmid=_extract_pmid(record),
        title=_first_nonempty(record.get("title")),
        text=text,
    )


def _count_lines(path) -> int:
    """Number of lines in a file, or 0 if it does not exist yet."""
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _write_jsonl(snippets: list[Snippet], path) -> None:
    """Write snippets to ``path`` as UTF-8 JSONL, one compact object per line."""
    with open(path, "w", encoding="utf-8") as fh:
        for snippet in snippets:
            # ensure_ascii=False keeps non-English characters readable on disk.
            fh.write(json.dumps(snippet, ensure_ascii=False) + "\n")


def _read_jsonl(path, limit: Optional[int] = None) -> list[Snippet]:
    """Read up to ``limit`` snippets back from a JSONL file (all if None)."""
    snippets: list[Snippet] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            snippets.append(json.loads(line))
            if limit is not None and len(snippets) >= limit:
                break
    return snippets


def _dedupe_by_id(snippets: list[Snippet]) -> list[Snippet]:
    """Keep the first occurrence of each id, preserving order."""
    seen: set[str] = set()
    unique: list[Snippet] = []
    for snippet in snippets:
        if snippet["id"] in seen:
            continue
        seen.add(snippet["id"])
        unique.append(snippet)
    return unique


def load_corpus(limit: int = config.CORPUS_SUBSET_SIZE) -> list[Snippet]:
    """
    Return up to ``limit`` cleaned PubMed snippets, caching them to
    ``config.CORPUS_PATH``.

    Resumable: if the cache already holds >= ``limit`` rows we load and return
    from it instead of re-streaming the dataset. Otherwise we stream
    ``config.HF_CORPUS``, map + clean + dedupe, then overwrite the cache.
    """
    # Fast path: the cache already has enough rows -> no network needed.
    if _count_lines(config.CORPUS_PATH) >= limit:
        return _read_jsonl(config.CORPUS_PATH, limit=limit)

    # ``datasets`` is heavy and optional at import time, so load it lazily here.
    from datasets import load_dataset

    stream = load_dataset(config.HF_CORPUS, split="train", streaming=True)
    keywords = _topic_keywords()

    snippets: list[Snippet] = []
    logged_keys = False
    for i, record in enumerate(stream):
        if not logged_keys:
            # Log the real schema of the FIRST record once so we can confirm which
            # column names this particular dataset dump actually uses.
            print(f"[ingest] first record keys: {sorted(record.keys())}")
            logged_keys = True
        if len(snippets) >= limit:
            break  # collected enough
        if config.CORPUS_SCAN_LIMIT and i >= config.CORPUS_SCAN_LIMIT:
            print(f"[ingest] scan cap {config.CORPUS_SCAN_LIMIT} reached")
            break
        snippet = _to_snippet(record, i)
        if snippet is None:
            continue  # junk row (empty / too-short text)
        if not _matches_topic(snippet, keywords):
            continue  # off-topic when a CORPUS_TOPIC filter is set
        snippets.append(snippet)

    snippets = _dedupe_by_id(snippets)
    _write_jsonl(snippets, config.CORPUS_PATH)
    note = f" (topic filter {keywords})" if keywords else ""
    print(f"[ingest] cached {len(snippets)} snippets to {config.CORPUS_PATH}{note}")
    return snippets


def iter_corpus() -> Iterator[Snippet]:
    """Yield Snippets one at a time from the cached JSONL (memory-friendly)."""
    if not config.CORPUS_PATH.exists():
        return  # nothing cached yet -> empty iterator
    with open(config.CORPUS_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _main() -> None:
    """CLI: ensure the corpus is cached, then show 3 sample snippets."""
    load_corpus()  # respects config.CORPUS_SUBSET_SIZE
    for snippet in _read_jsonl(config.CORPUS_PATH, limit=3):
        print("-" * 60)
        print(f"pmid : {snippet['pmid']}")
        print(f"title: {snippet['title']}")
        print(f"text : {snippet['text'][:200]}")


if __name__ == "__main__":
    _main()
