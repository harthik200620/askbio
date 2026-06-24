"""
Pulls PubMed snippets from HuggingFace (MedRAG/pubmed), cleans them, and caches
them to a local JSONL file for the rest of the pipeline to use.

datasets is imported inside the functions that use it so the module (and tests)
import fine without it installed.
"""
from __future__ import annotations

import json
import re
from typing import Iterator, Optional

import config
from schemas import Snippet

# Shorter than this is usually a fragment/label, not a real passage.
_MIN_TEXT_CHARS = 20

_DIGITS = re.compile(r"\d+")


def _first_nonempty(*values: object) -> str:
    """First non-empty/non-whitespace string, or ""."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _topic_keywords() -> list[str]:
    return [w.lower() for w in config.CORPUS_TOPIC.replace(",", " ").split() if w.strip()]


def _matches_topic(snippet: Snippet, keywords: list[str]) -> bool:
    """Whether the snippet mentions any keyword. No keywords means match everything."""
    if not keywords:
        return True
    haystack = (snippet["title"] + " " + snippet["text"]).lower()
    return any(kw in haystack for kw in keywords)


def _extract_pmid(record: dict) -> str:
    """PMID from an explicit field, else the first digit run in the id, else ""."""
    # PMID can come through as an int (e.g. 21), so str() it before checking for
    # emptiness -- otherwise a numeric PMID looks "missing" and we fall through to
    # the id, whose leading digits are the dump year ("pubmed23n0001" -> "23").
    for key in ("PMID", "pmid"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    match = _DIGITS.search(str(record.get("id", "")))
    return match.group(0) if match else ""


def _to_snippet(record: dict, i: int) -> Optional[Snippet]:
    """Map a raw record to a Snippet, or None if the text is missing/too short.

    i is the row index, only used to build a fallback id when there isn't one.
    """
    # Which key holds the text varies by dump.
    text = _first_nonempty(
        record.get("content"),
        record.get("contents"),
        record.get("text"),
        record.get("abstract"),
    )
    if len(text) < _MIN_TEXT_CHARS:
        return None

    snippet_id = str(record.get("id") or f"row{i}")
    return Snippet(
        id=snippet_id,
        pmid=_extract_pmid(record),
        title=_first_nonempty(record.get("title")),
        text=text,
    )


def _count_lines(path) -> int:
    """Line count, or 0 if the file doesn't exist."""
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _write_jsonl(snippets: list[Snippet], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for snippet in snippets:
            # ensure_ascii=False so non-English chars stay readable on disk.
            fh.write(json.dumps(snippet, ensure_ascii=False) + "\n")


def _read_jsonl(path, limit: Optional[int] = None) -> list[Snippet]:
    """Read snippets from a JSONL file, up to limit (all if None)."""
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
    """Keep the first occurrence of each id, in order."""
    seen: set[str] = set()
    unique: list[Snippet] = []
    for snippet in snippets:
        if snippet["id"] in seen:
            continue
        seen.add(snippet["id"])
        unique.append(snippet)
    return unique


def _load_corpus_topic_focused() -> list[Snippet]:
    """Stream PubMed and fill per-topic buckets until each reaches its target.

    Each snippet is assigned to the first topic whose keywords appear in its
    title+text. A snippet is never added to more than one bucket, so there are
    no duplicates before the final _dedupe_by_id pass.
    """
    from datasets import load_dataset

    targets = config.CORPUS_TOPICS
    per_target = config.CORPUS_PER_TOPIC_TARGET
    total_target = len(targets) * per_target

    # Resume: if corpus.jsonl already has enough rows, skip streaming.
    if _count_lines(config.CORPUS_PATH) >= total_target:
        print(f"[ingest] corpus already has {total_target}+ rows, reading cache")
        return _read_jsonl(config.CORPUS_PATH)

    buckets: dict[str, list[Snippet]] = {t: [] for t in targets}

    stream = load_dataset(config.HF_CORPUS, split="train", streaming=True)
    logged_keys = False

    for i, record in enumerate(stream):
        if not logged_keys:
            print(f"[ingest] first record keys: {sorted(record.keys())}")
            logged_keys = True

        # Progress report every 50k rows scanned
        if i > 0 and i % 50000 == 0:
            counts = {t: len(v) for t, v in buckets.items()}
            print(f"[ingest] scanned {i} rows | {counts}")

        # Stop if every bucket is full
        if all(len(v) >= per_target for v in buckets.values()):
            break

        if config.CORPUS_SCAN_LIMIT and i >= config.CORPUS_SCAN_LIMIT:
            print(f"[ingest] scan cap {config.CORPUS_SCAN_LIMIT} reached")
            break

        snippet = _to_snippet(record, i)
        if snippet is None:
            continue

        haystack = (snippet["title"] + " " + snippet["text"]).lower()

        for topic, keywords in targets.items():
            if len(buckets[topic]) >= per_target:
                continue
            if any(kw in haystack for kw in keywords):
                buckets[topic].append(snippet)
                break  # one bucket per snippet — no duplicates

    # Merge and report
    all_snippets: list[Snippet] = []
    for topic, snips in buckets.items():
        print(f"[ingest] {topic}: {len(snips)} snippets")
        all_snippets.extend(snips)

    all_snippets = _dedupe_by_id(all_snippets)
    _write_jsonl(all_snippets, config.CORPUS_PATH)
    print(f"[ingest] total: {len(all_snippets)} snippets -> {config.CORPUS_PATH}")
    return all_snippets


def load_corpus(limit: int = config.CORPUS_SUBSET_SIZE) -> list[Snippet]:
    """Return cleaned snippets, caching them to config.CORPUS_PATH.

    If CORPUS_TOPICS is configured (the default), uses per-topic bucket filling
    for a dense, balanced corpus. Otherwise falls back to a flat keyword/limit scan.
    """
    if config.CORPUS_TOPICS:
        return _load_corpus_topic_focused()

    # Flat fallback (no topics defined): first N snippets matching CORPUS_TOPIC.
    if _count_lines(config.CORPUS_PATH) >= limit:
        return _read_jsonl(config.CORPUS_PATH, limit=limit)

    from datasets import load_dataset

    stream = load_dataset(config.HF_CORPUS, split="train", streaming=True)
    keywords = _topic_keywords()

    snippets: list[Snippet] = []
    logged_keys = False
    for i, record in enumerate(stream):
        if not logged_keys:
            print(f"[ingest] first record keys: {sorted(record.keys())}")
            logged_keys = True
        if len(snippets) >= limit:
            break
        if config.CORPUS_SCAN_LIMIT and i >= config.CORPUS_SCAN_LIMIT:
            print(f"[ingest] scan cap {config.CORPUS_SCAN_LIMIT} reached")
            break
        snippet = _to_snippet(record, i)
        if snippet is None:
            continue
        if not _matches_topic(snippet, keywords):
            continue
        snippets.append(snippet)

    snippets = _dedupe_by_id(snippets)
    _write_jsonl(snippets, config.CORPUS_PATH)
    note = f" (topic filter {keywords})" if keywords else ""
    print(f"[ingest] cached {len(snippets)} snippets to {config.CORPUS_PATH}{note}")
    return snippets


def iter_corpus() -> Iterator[Snippet]:
    """Yield Snippets one at a time from the cached JSONL."""
    if not config.CORPUS_PATH.exists():
        return
    with open(config.CORPUS_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _main() -> None:
    """Cache the corpus, then print a few samples."""
    load_corpus()
    for snippet in _read_jsonl(config.CORPUS_PATH, limit=3):
        print("-" * 60)
        print(f"pmid : {snippet['pmid']}")
        print(f"title: {snippet['title']}")
        print(f"text : {snippet['text'][:200]}")


if __name__ == "__main__":
    _main()
