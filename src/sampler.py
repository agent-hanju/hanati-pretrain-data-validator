"""Core logic for randomly sampling text snippets from JSONL files.

Uses reservoir sampling (Algorithm R) so memory usage is O(n) regardless
of total file size — safe for multi-GB inputs.
"""

import random
import sys
from pathlib import Path
from typing import Any

import orjson


def _iter_documents(
    file_paths: list[str],
    text_field: str,
) -> Any:
    """Yield documents one at a time from multiple JSONL files."""
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            print(f"[WARN] File not found, skipping: {fp}")
            continue
        auto_id = 0
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                if not isinstance(obj, dict) or text_field not in obj:
                    continue

                doc_id = obj.get("id")
                if doc_id is None:
                    auto_id += 1
                    doc_id = f"auto-{auto_id}"

                yield {
                    "file": str(path),
                    "id": doc_id,
                    "text": obj[text_field],
                }


def reservoir_sample(
    file_paths: list[str],
    n: int,
    text_field: str,
) -> list[dict[str, Any]]:
    """Reservoir sampling (Algorithm R): pick n items from a stream of unknown length."""
    reservoir: list[dict[str, Any]] = []
    for i, doc in enumerate(_iter_documents(file_paths, text_field)):
        if i < n:
            reservoir.append(doc)
        else:
            j = random.randint(0, i)
            if j < n:
                reservoir[j] = doc
    return reservoir


def extract_snippet(text: str, min_lines: int = 1, max_lines: int = 5) -> str:
    """Extract 1-5 consecutive lines from a random position in the text."""
    lines = text.splitlines()
    if not lines:
        return ""
    num_lines = min(random.randint(min_lines, max_lines), len(lines))
    start = random.randint(0, len(lines) - num_lines)
    return "\n".join(lines[start : start + num_lines])


def sample(
    file_paths: list[str],
    n: int,
    text_field: str = "text",
    seed: int | None = None,
) -> list[dict[str, str]]:
    """Sample n documents and extract snippets."""
    if seed is not None:
        random.seed(seed)

    picked = reservoir_sample(file_paths, n, text_field)
    if not picked:
        print("[ERROR] No valid documents found.")
        sys.exit(1)

    if len(picked) < n:
        print(f"[WARN] Only {len(picked)} documents available (requested {n}).")

    results: list[dict[str, str]] = []
    for doc in picked:
        snippet = extract_snippet(doc["text"])
        results.append({
            "file": doc["file"],
            "id": doc["id"],
            "original": doc["text"],
            "text": snippet,
        })
    return results
