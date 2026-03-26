import sys
from pathlib import Path
from typing import Any, cast

import orjson


def load_jsonl(file_path: str, prompt_field: str = "prompt") -> list[dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] Input file not found: {file_path}")
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    auto_id = 0
    with open(path, "rb") as f:  # orjson.loads accepts bytes directly
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj: Any = orjson.loads(line)
            except orjson.JSONDecodeError as e:
                print(f"[WARN] Line {lineno}: JSON parse error — skipped. ({e})")
                continue

            if not isinstance(obj, dict):
                print(f"[WARN] Line {lineno}: not a JSON object — skipped.")
                continue

            row = cast(dict[str, Any], obj)

            if prompt_field not in row:
                print(f"[WARN] Line {lineno}: missing '{prompt_field}' field — skipped.")
                continue

            if "id" not in row:
                auto_id += 1
                row["id"] = f"auto-{auto_id}"

            rows.append(row)

    return rows
