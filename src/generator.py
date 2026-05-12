from typing import Any, TypedDict

from openai import AsyncOpenAI

from .config import Config

# vLLM default timeout can be long; 120 s covers most large-model inference
_TIMEOUT_SECONDS = 120.0


class CompletionResult(TypedDict):
    generated: str
    tool_calls: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


def make_async_client(base_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=base_url,
        api_key="EMPTY",
        max_retries=0,       # we handle retries ourselves
        timeout=_TIMEOUT_SECONDS,
    )


async def call_text_completions(client: AsyncOpenAI, prompt: str, config: Config) -> CompletionResult:
    """chat.completions 대신 text completions API (/v1/completions) 사용."""
    api = config["api"]
    gen = config["generation"]

    rep_penalty = gen["repetition_penalty"]
    top_k = gen["top_k"]
    extra_body: dict[str, Any] = {}
    if rep_penalty != 1.0:
        extra_body["repetition_penalty"] = rep_penalty
    if top_k != -1:
        extra_body["top_k"] = top_k

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = await client.completions.create(
                model=api["model"],
                prompt=prompt,
                max_tokens=gen["max_tokens"],
                temperature=gen["temperature"],
                top_p=gen["top_p"],
                seed=gen["seed"],
                extra_body=extra_body or None,
            )
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
    return CompletionResult(generated="", tool_calls="", reasoning="", finish_reason="error", prompt_tokens=0, completion_tokens=0)


async def call_chat_messages(
    client: AsyncOpenAI,
    messages: list[dict],
    config: Config,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> CompletionResult:
    """messages 배열(tool_calls 포함 multi-turn)을 그대로 chat completions에 전달."""
    api = config["api"]
    gen = config["generation"]

    extra_body: dict[str, Any] = {}
    rep_penalty = gen["repetition_penalty"]
    top_k = gen["top_k"]
    if rep_penalty != 1.0:
        extra_body["repetition_penalty"] = rep_penalty
    if top_k != -1:
        extra_body["top_k"] = top_k

    kwargs: dict[str, Any] = dict(
        model=api["model"],
        messages=messages,
        max_tokens=gen["max_tokens"],
        temperature=gen["temperature"],
        top_p=gen["top_p"],
        seed=gen["seed"],
        extra_body=extra_body or None,
    )
    if tools:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            usage = response.usage
            msg = choice.message
            import orjson
            tool_calls_str = orjson.dumps(
                [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                 for tc in msg.tool_calls]
            ).decode() if msg.tool_calls else ""
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
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
    return CompletionResult(generated="", tool_calls="", reasoning="", finish_reason="error", prompt_tokens=0, completion_tokens=0)


async def call_completions(client: AsyncOpenAI, prompt: str, config: Config) -> CompletionResult:
    api = config["api"]
    gen = config["generation"]

    rep_penalty = gen["repetition_penalty"]
    top_k = gen["top_k"]
    extra_body: dict[str, Any] = {}
    if rep_penalty != 1.0:
        extra_body["repetition_penalty"] = rep_penalty
    if top_k != -1:
        extra_body["top_k"] = top_k

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model=api["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=gen["max_tokens"],
                temperature=gen["temperature"],
                top_p=gen["top_p"],
                seed=gen["seed"],
                extra_body=extra_body or None,
            )
            choice = response.choices[0]
            usage = response.usage
            return CompletionResult(
                generated=choice.message.content or "",
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
    return CompletionResult(generated="", tool_calls="", reasoning="", finish_reason="error", prompt_tokens=0, completion_tokens=0)
