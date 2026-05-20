import argparse
import asyncio
from typing import Any

import orjson

from src.config import Config, load_config
from src.generator import CompletionResult, call_chat_messages, call_completions, make_async_client, resolve_model
from src.loader import load_jsonl
from src.writer import make_writer

_MESSAGES_SKIP_FIELDS = {"id", "type", "messages", "tools", "tool_choice", "response_format"}

FIXED_COLUMNS: list[str] = [
    "id", "type", "prompt", "generated", "tool_calls", "reasoning", "model", "finish_reason",
    "prompt_tokens", "completion_tokens",
]


def _extra_keys(raw: dict[str, Any], prompt_field: str) -> list[str]:
    skip = _MESSAGES_SKIP_FIELDS if prompt_field == "messages" else {"id", "type", prompt_field}
    return [k for k in raw if k not in skip]


def make_fieldnames(first_raw: dict[str, Any], prompt_field: str) -> list[str]:
    return FIXED_COLUMNS + _extra_keys(first_raw, prompt_field)


def build_row(
    raw: dict[str, Any],
    result: CompletionResult,
    config: Config,
    prompt_field: str,
) -> dict[str, Any]:
    extra_keys = _extra_keys(raw, prompt_field)
    prompt_val = raw.get(prompt_field, "")
    if not isinstance(prompt_val, str):
        prompt_val = orjson.dumps(prompt_val).decode()
    row: dict[str, Any] = {
        "id": raw.get("id", ""),
        "type": raw.get("type", raw.get("eval_type", "")),
        "prompt": prompt_val,
        "generated": result["generated"],
        "tool_calls": result["tool_calls"],
        "reasoning": result["reasoning"],
        "model": config["api"]["model"],
        "finish_reason": result["finish_reason"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
    }
    for k in extra_keys:
        v = raw[k]
        row[k] = orjson.dumps(v).decode() if not isinstance(v, (str, int, float, bool, type(None))) else v
    return row


async def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    is_chat = False
    prompt_field = "prompt"
    if args.mode == "messages":
        is_chat = True
        prompt_field = "messages"
    rows_input = load_jsonl(args.input, is_chat)

    if config["api"]["model"] is None:
        config["api"]["model"] = await resolve_model(config["api"]["base_url"], float(config["api"]["timeout"]))

    print(f"Loaded {len(rows_input)} rows from {args.input}")
    print(f"Mode: {args.mode}")

    total = len(rows_input)
    concurrency = config["api"]["concurrency"]
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async with make_async_client(config["api"]["base_url"], config["api"]["timeout"]) as client:
        async def bounded_call(raw: dict[str, Any]) -> dict[str, Any]:
            nonlocal completed
            async with semaphore:
                if is_chat:
                    result = await call_chat_messages(
                        client,
                        raw["messages"],
                        config,
                        tools=raw.get("tools"),
                        tool_choice=raw.get("tool_choice"),
                        response_format=raw.get("response_format"),
                    )
                else:
                    result = await call_completions(client, raw[prompt_field], config)
            completed += 1
            status = "ok" if result["finish_reason"] != "error" else "error"
            print(f"[{completed}/{total}] id={raw['id']} → {status} ({result['finish_reason']})")
            return build_row(raw, result, config, prompt_field)

        tasks = [bounded_call(raw) for raw in rows_input]
        rows_output: list[dict[str, Any]] = await asyncio.gather(*tasks)

    fieldnames = make_fieldnames(rows_input[0], prompt_field)
    with make_writer(args.format, config, fieldnames, args.output, args.input) as writer:
        for row in rows_output:
            writer.write_row(row)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", help="YAML config file path")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--mode", choices=["prompt", "messages"], default="messages", help="prompt: completions API, messages: chat completions API")
    parser.add_argument("--format", choices=["csv", "jsonl"], default="csv", help="Output format")

    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
