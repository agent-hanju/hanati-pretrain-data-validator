from typing import Any, TypedDict

from openai import AsyncOpenAI

from .config import Config

# vLLM default timeout can be long; 120 s covers most large-model inference
_TIMEOUT_SECONDS = 120.0


class CompletionResult(TypedDict):
    generated: str
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
                finish_reason=choice.finish_reason or "unknown",
                prompt_tokens=usage.prompt_tokens if usage is not None else 0,
                completion_tokens=usage.completion_tokens if usage is not None else 0,
            )
        except Exception as e:
            last_error = e
            if attempt == 0:
                print(f"\n[WARN] API call failed, retrying... ({e})")

    print(f"\n[ERROR] API call failed after retry: {last_error}")
    return CompletionResult(
        generated="",
        finish_reason="error",
        prompt_tokens=0,
        completion_tokens=0,
    )


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
                finish_reason=choice.finish_reason or "unknown",
                prompt_tokens=usage.prompt_tokens if usage is not None else 0,
                completion_tokens=usage.completion_tokens if usage is not None else 0,
            )
        except Exception as e:
            last_error = e
            if attempt == 0:
                print(f"\n[WARN] API call failed, retrying... ({e})")

    print(f"\n[ERROR] API call failed after retry: {last_error}")
    return CompletionResult(
        generated="",
        finish_reason="error",
        prompt_tokens=0,
        completion_tokens=0,
    )
