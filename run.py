import argparse
import asyncio
from pathlib import Path
from typing import Any

import orjson

from src.config import Config, load_config
from src.generator import CompletionResult, call_completions, make_async_client
from src.loader import load_jsonl
from src.sampler import sample
from src.writer import make_writer, convert_csv_to_xlsx

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


async def run_validate(args: argparse.Namespace) -> None:
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

    fmt = args.format or ("xlsx" if args.output.endswith(".xlsx") else "csv")
    fieldnames = make_fieldnames(rows_input[0], prompt_field)
    with make_writer(fmt, config, fieldnames, args.output, args.input) as writer:
        for row in rows_output:  # asyncio.gather preserves input order
            writer.write_row(row)


def run_sample(args: argparse.Namespace) -> None:
    results = sample(args.inputs, args.n, args.text_field, args.seed)

    out = Path(args.output)
    with open(out, "wb") as f:
        for row in results:
            f.write(orjson.dumps(row) + b"\n")

    print(f"Wrote {len(results)} samples to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain Data Validator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- validate ---
    p_validate = subparsers.add_parser(
        "validate", help="Run validation via LLM API",
    )
    p_validate.add_argument("--config", default="config.yml", help="YAML config file path")
    p_validate.add_argument("--input", required=True, help="Input JSONL file path")
    p_validate.add_argument("--output", required=True, help="Output file path")
    p_validate.add_argument("--format", choices=["csv", "xlsx"], default=None,
                            help="Output format (default: inferred from --output extension)")
    p_validate.add_argument("--dry-run", action="store_true",
                            help="Validate config and jsonl without calling the API")

    # --- convert ---
    p_convert = subparsers.add_parser(
        "convert", help="Convert CSV output to xlsx",
    )
    p_convert.add_argument("--input", required=True, help="Input CSV file path")
    p_convert.add_argument("--output", default=None, help="Output xlsx file path (default: same name with .xlsx)")

    # --- sample ---
    p_sample = subparsers.add_parser(
        "sample", help="Sample random snippets from JSONL files",
    )
    p_sample.add_argument("inputs", nargs="+", help="Input JSONL file paths")
    p_sample.add_argument("-n", type=int, default=10,
                          help="Number of documents to sample (default: 10)")
    p_sample.add_argument("-o", "--output", required=True, help="Output JSONL file path")
    p_sample.add_argument("--text-field", default="text",
                          help="Name of the text field (default: text)")
    p_sample.add_argument("--seed", type=int, default=None,
                          help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.command == "validate":
        asyncio.run(run_validate(args))
    elif args.command == "convert":
        xlsx_out = args.output or args.input.removesuffix(".csv") + ".xlsx"
        convert_csv_to_xlsx(args.input, xlsx_out)
    elif args.command == "sample":
        run_sample(args)


if __name__ == "__main__":
    main()
