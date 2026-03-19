import argparse
import asyncio
from typing import Any

from src.config import Config, load_config
from src.generator import CompletionResult, call_completions, make_async_client
from src.loader import load_jsonl
from src.writer import CsvWriter

FIXED_COLUMNS: list[str] = [
    "id", "type", "prompt", "generated", "model", "finish_reason",
    "prompt_tokens", "completion_tokens",
]


def make_fieldnames(first_raw: dict[str, Any], prompt_field: str) -> list[str]:
    extra = [k for k in first_raw if k not in ("id", "type", prompt_field)]
    return FIXED_COLUMNS + extra


def build_row(
    raw: dict[str, Any],
    result: CompletionResult,
    config: Config,
    prompt_field: str,
) -> dict[str, Any]:
    extra_keys = [k for k in raw if k not in ("id", "type", prompt_field)]
    row: dict[str, Any] = {
        "id": raw.get("id", ""),
        "type": raw.get("type", ""),
        "prompt": raw.get(prompt_field, ""),
        "generated": result["generated"],
        "model": config["api"]["model"],
        "finish_reason": result["finish_reason"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
    }
    for k in extra_keys:
        row[k] = raw[k]
    return row


async def async_main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prompt_field = config["input"]["prompt_field"]
    rows_input = load_jsonl(args.input, prompt_field)

    print(f"Loaded {len(rows_input)} rows from {args.input}")

    if args.dry_run:
        print("[dry-run] Config and input file are valid. Exiting without API calls.")
        return

    total = len(rows_input)
    concurrency = config["api"]["concurrency"]
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async with make_async_client(config["api"]["base_url"]) as client:
        async def bounded_call(raw: dict[str, Any]) -> dict[str, Any]:
            nonlocal completed
            async with semaphore:
                result = await call_completions(client, raw[prompt_field], config)
            completed += 1
            status = "ok" if result["finish_reason"] != "error" else "error"
            print(f"[{completed}/{total}] id={raw['id']} → {status} ({result['finish_reason']})")
            return build_row(raw, result, config, prompt_field)

        tasks = [bounded_call(raw) for raw in rows_input]
        rows_output: list[dict[str, Any]] = await asyncio.gather(*tasks)

    fieldnames = make_fieldnames(rows_input[0], prompt_field)
    with CsvWriter(config, fieldnames, args.output, args.input) as writer:
        for row in rows_output:  # asyncio.gather preserves input order
            writer.write_row(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation Set Generator")
    parser.add_argument("--config", default="config.yml", help="YAML config file path")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config and jsonl without calling the API")
    asyncio.run(async_main(parser.parse_args()))


if __name__ == "__main__":
    main()
