import sys
from typing import Any, TypedDict, cast

import httpx
import orjson
from openai import AsyncOpenAI
from openai.types import Completion
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from .config import Config

class CompletionResult(TypedDict):
    generated: str
    reasoning: str
    tool_calls: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


async def resolve_model(base_url: str, timeout: float) -> str:
    """base_url/models 에서 첫 번째 모델 ID를 가져온다."""
    url = base_url.rstrip("/") + "/models"
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url, headers={"Authorization": "Bearer EMPTY"})
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                print(f"[ERROR] No models returned from {url}")
                sys.exit(1)
            model_id: str = data[0]["id"]
            print(f"[INFO] model not set — using '{model_id}' from {url}")
            return model_id
        except httpx.HTTPError as e:
            print(f"[ERROR] Failed to fetch model list from {url}: {e}")
            sys.exit(1)


def make_async_client(base_url: str, timeout: float) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=base_url,
        api_key="EMPTY",
        max_retries=0,
        timeout=timeout,
    )


def _extra_body(repetition_penalty: float | None, top_k: int | None) -> dict[str, Any] | None:
    body: dict[str, Any] = {}
    if repetition_penalty is not None:
        body["repetition_penalty"] = repetition_penalty
    if top_k is not None:
        body["top_k"] = top_k
    return body or None


def _error_result() -> CompletionResult:
    return CompletionResult(generated="", tool_calls="", reasoning="", finish_reason="error", prompt_tokens=0, completion_tokens=0)


async def call_completions(client: AsyncOpenAI, prompt: str, config: Config) -> CompletionResult:
    """prompt 필드 → /v1/completions (text completions API)."""
    api = config["api"]
    gen = config["generation"]

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            model = api["model"]
            assert model is not None, "model must be set before calling API"
            comp_kwargs: dict[str, Any] = dict(model=model, prompt=prompt)
            if (v := gen.get("max_tokens")) is not None:
                comp_kwargs["max_tokens"] = v
            if (v := gen.get("temperature")) is not None:
                comp_kwargs["temperature"] = v
            if (v := gen.get("top_p")) is not None:
                comp_kwargs["top_p"] = v
            if (v := gen.get("seed")) is not None:
                comp_kwargs["seed"] = v
            extra = _extra_body(gen.get("repetition_penalty"), gen.get("top_k"))
            if extra is not None:
                comp_kwargs["extra_body"] = extra
            response = cast(Completion, await client.completions.create(**comp_kwargs))
            choice = response.choices[0]
            usage = response.usage
            return CompletionResult(
                generated=choice.text,
                tool_calls="",
                reasoning="",
                finish_reason=choice.finish_reason or "unknown",
                prompt_tokens=usage.prompt_tokens if usage is not None else 0,
                completion_tokens=usage.completion_tokens if usage is not None else 0,
            )
        except Exception as e:
            last_error = e
            if attempt == 0:
                print(f"\n[WARN] API call failed, retrying... ({e})")

    print(f"\n[ERROR] API call failed after retry: {last_error}")
    return _error_result()


async def call_chat_messages(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    config: Config,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
) -> CompletionResult:
    """messages 배열 → /v1/chat/completions (chat completions API)."""
    api = config["api"]
    gen = config["generation"]

    kwargs: dict[str, Any] = dict(model=api["model"], messages=messages)
    if (v := gen.get("max_tokens")) is not None:
        kwargs["max_tokens"] = v
    if (v := gen.get("temperature")) is not None:
        kwargs["temperature"] = v
    if (v := gen.get("top_p")) is not None:
        kwargs["top_p"] = v
    if (v := gen.get("seed")) is not None:
        kwargs["seed"] = v
    extra = _extra_body(gen.get("repetition_penalty"), gen.get("top_k"))
    if extra is not None:
        kwargs["extra_body"] = extra
    if tools:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = cast(ChatCompletion, await client.chat.completions.create(**kwargs))
            choice = response.choices[0]
            usage = response.usage
            msg = choice.message
            fn_calls = [tc for tc in msg.tool_calls if isinstance(tc, ChatCompletionMessageToolCall)] if msg.tool_calls else []
            tool_calls_str = orjson.dumps(
                [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                 for tc in fn_calls]
            ).decode() if fn_calls else ""
            reasoning: str = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            return CompletionResult(
                generated=msg.content or "",
                tool_calls=tool_calls_str,
                reasoning=reasoning,
                finish_reason=choice.finish_reason or "unknown",
                prompt_tokens=usage.prompt_tokens if usage is not None else 0,
                completion_tokens=usage.completion_tokens if usage is not None else 0,
            )
        except Exception as e:
            last_error = e
            if attempt == 0:
                print(f"\n[WARN] API call failed, retrying... ({e})")

    print(f"\n[ERROR] API call failed after retry: {last_error}")
    return _error_result()
