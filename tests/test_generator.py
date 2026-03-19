import pytest
from unittest.mock import AsyncMock, MagicMock

from src.generator import call_completions

SAMPLE_CONFIG = {
    "api": {"model": "test-model"},
    "generation": {
        "max_tokens": 100,
        "temperature": 0.1,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "seed": 42,
    },
}


def make_mock_response(
    text: str = "response text",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    message = MagicMock()
    message.content = text

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def make_client(side_effect=None, return_value=None) -> MagicMock:
    client = MagicMock()
    if side_effect is not None:
        client.chat.completions.create = AsyncMock(side_effect=side_effect)
    else:
        client.chat.completions.create = AsyncMock(return_value=return_value)
    return client


async def test_call_completions_success():
    client = make_client(return_value=make_mock_response("hello", "stop", 8, 4))
    result = await call_completions(client, "test prompt", SAMPLE_CONFIG)

    assert result["generated"] == "hello"
    assert result["finish_reason"] == "stop"
    assert result["prompt_tokens"] == 8
    assert result["completion_tokens"] == 4


async def test_call_completions_retry_on_first_failure(capsys):
    client = make_client(side_effect=[Exception("timeout"), make_mock_response("ok")])
    result = await call_completions(client, "prompt", SAMPLE_CONFIG)

    assert result["generated"] == "ok"
    assert result["finish_reason"] == "stop"
    assert "WARN" in capsys.readouterr().out
    assert client.chat.completions.create.call_count == 2


async def test_call_completions_all_fail_returns_error_dict(capsys):
    client = make_client(side_effect=Exception("fail"))
    result = await call_completions(client, "prompt", SAMPLE_CONFIG)

    assert result["generated"] == ""
    assert result["finish_reason"] == "error"
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert "ERROR" in capsys.readouterr().out


async def test_repetition_penalty_adds_extra_body():
    cfg = {
        "api": {"model": "m"},
        "generation": {
            "max_tokens": 10,
            "temperature": 0.1,
            "top_p": 1.0,
            "repetition_penalty": 1.3,
            "seed": 0,
        },
    }
    client = make_client(return_value=make_mock_response())
    await call_completions(client, "p", cfg)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["repetition_penalty"] == 1.3


async def test_repetition_penalty_default_no_extra_body():
    client = make_client(return_value=make_mock_response())
    await call_completions(client, "p", SAMPLE_CONFIG)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs.get("extra_body") is None


async def test_call_passes_correct_params():
    client = make_client(return_value=make_mock_response())
    await call_completions(client, "my prompt", SAMPLE_CONFIG)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"] == [{"role": "user", "content": "my prompt"}]
    assert kwargs["max_tokens"] == 100
    assert kwargs["temperature"] == 0.1
    assert kwargs["top_p"] == 1.0
    assert kwargs["seed"] == 42
