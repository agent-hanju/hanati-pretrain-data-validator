from typing import cast
from unittest.mock import AsyncMock, MagicMock

from src.config import Config
from src.generator import call_chat_messages, call_completions

SAMPLE_CONFIG: Config = cast(Config, {
    "api": {"model": "test-model"},
    "generation": {
        "max_tokens": 100,
        "temperature": 0.1,
        "seed": 42,
    },
})


def _make_text_response(
    text: str = "response text",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    choice = MagicMock()
    choice.text = text
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_chat_response(
    content: str = "chat response",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    tool_calls=None,
    reasoning: str = "",
) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.reasoning_content = reasoning
    msg.reasoning = None

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_completions_client(side_effect=None, return_value=None) -> MagicMock:
    client = MagicMock()
    if side_effect is not None:
        client.completions.create = AsyncMock(side_effect=side_effect)
    else:
        client.completions.create = AsyncMock(return_value=return_value)
    return client


def _make_chat_client(side_effect=None, return_value=None) -> MagicMock:
    client = MagicMock()
    if side_effect is not None:
        client.chat.completions.create = AsyncMock(side_effect=side_effect)
    else:
        client.chat.completions.create = AsyncMock(return_value=return_value)
    return client


# ── call_completions (text API) ───────────────────────────────────────────────

async def test_call_completions_success():
    client = _make_completions_client(return_value=_make_text_response("hello", "stop", 8, 4))
    result = await call_completions(client, "test prompt", SAMPLE_CONFIG)

    assert result["generated"] == "hello"
    assert result["finish_reason"] == "stop"
    assert result["prompt_tokens"] == 8
    assert result["completion_tokens"] == 4
    assert result["tool_calls"] == ""
    assert result["reasoning"] == ""


async def test_call_completions_passes_prompt_not_messages():
    client = _make_completions_client(return_value=_make_text_response())
    await call_completions(client, "my prompt", SAMPLE_CONFIG)

    kwargs = client.completions.create.call_args.kwargs
    assert kwargs["prompt"] == "my prompt"
    assert "messages" not in kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["max_tokens"] == 100
    assert kwargs["temperature"] == 0.1
    assert kwargs["seed"] == 42


async def test_call_completions_retry_on_first_failure(capsys):
    client = _make_completions_client(side_effect=[Exception("timeout"), _make_text_response("ok")])
    result = await call_completions(client, "prompt", SAMPLE_CONFIG)

    assert result["generated"] == "ok"
    assert "WARN" in capsys.readouterr().out
    assert client.completions.create.call_count == 2


async def test_call_completions_all_fail_returns_error(capsys):
    client = _make_completions_client(side_effect=Exception("fail"))
    result = await call_completions(client, "prompt", SAMPLE_CONFIG)

    assert result["generated"] == ""
    assert result["finish_reason"] == "error"
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert "ERROR" in capsys.readouterr().out


async def test_call_completions_repetition_penalty_in_extra_body():
    cfg = cast(Config, {**SAMPLE_CONFIG, "generation": {**SAMPLE_CONFIG["generation"], "repetition_penalty": 1.3}})
    client = _make_completions_client(return_value=_make_text_response())
    await call_completions(client, "p", cfg)

    kwargs = client.completions.create.call_args.kwargs
    assert kwargs["extra_body"]["repetition_penalty"] == 1.3


async def test_call_completions_top_k_in_extra_body():
    cfg = cast(Config, {**SAMPLE_CONFIG, "generation": {**SAMPLE_CONFIG["generation"], "top_k": 50}})
    client = _make_completions_client(return_value=_make_text_response())
    await call_completions(client, "p", cfg)

    kwargs = client.completions.create.call_args.kwargs
    assert kwargs["extra_body"]["top_k"] == 50


async def test_call_completions_default_params_no_extra_body():
    client = _make_completions_client(return_value=_make_text_response())
    await call_completions(client, "p", SAMPLE_CONFIG)

    kwargs = client.completions.create.call_args.kwargs
    assert kwargs.get("extra_body") is None


# ── call_chat_messages (chat API) ─────────────────────────────────────────────

async def test_call_chat_messages_success():
    messages = [{"role": "user", "content": "hello"}]
    client = _make_chat_client(return_value=_make_chat_response("hi there", "stop", 6, 3))
    result = await call_chat_messages(client, messages, SAMPLE_CONFIG)

    assert result["generated"] == "hi there"
    assert result["finish_reason"] == "stop"
    assert result["prompt_tokens"] == 6
    assert result["completion_tokens"] == 3
    assert result["tool_calls"] == ""


async def test_call_chat_messages_passes_messages_array():
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    client = _make_chat_client(return_value=_make_chat_response())
    await call_chat_messages(client, messages, SAMPLE_CONFIG)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == messages
    assert kwargs["model"] == "test-model"


async def test_call_chat_messages_with_tools():
    messages = [{"role": "user", "content": "use tool"}]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    client = _make_chat_client(return_value=_make_chat_response())
    await call_chat_messages(client, messages, SAMPLE_CONFIG, tools=tools, tool_choice="auto")

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"


async def test_call_chat_messages_no_tools_when_none():
    messages = [{"role": "user", "content": "hi"}]
    client = _make_chat_client(return_value=_make_chat_response())
    await call_chat_messages(client, messages, SAMPLE_CONFIG)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


async def test_call_chat_messages_retry_on_first_failure(capsys):
    messages = [{"role": "user", "content": "hi"}]
    client = _make_chat_client(side_effect=[Exception("timeout"), _make_chat_response("ok")])
    result = await call_chat_messages(client, messages, SAMPLE_CONFIG)

    assert result["generated"] == "ok"
    assert "WARN" in capsys.readouterr().out
    assert client.chat.completions.create.call_count == 2


async def test_call_chat_messages_all_fail_returns_error(capsys):
    client = _make_chat_client(side_effect=Exception("fail"))
    result = await call_chat_messages(client, [{"role": "user", "content": "hi"}], SAMPLE_CONFIG)

    assert result["generated"] == ""
    assert result["finish_reason"] == "error"
    assert "ERROR" in capsys.readouterr().out


async def test_call_chat_messages_reasoning_field():
    messages = [{"role": "user", "content": "think"}]
    client = _make_chat_client(return_value=_make_chat_response(reasoning="<think>...</think>"))
    result = await call_chat_messages(client, messages, SAMPLE_CONFIG)

    assert result["reasoning"] == "<think>...</think>"
