from typing import cast

from src.config import Config
from src.generator import CompletionResult
from run import build_row, make_fieldnames, FIXED_COLUMNS

SAMPLE_CONFIG: Config = cast(Config, {
    "api": {"model": "test-model"},
    "generation": {},
})

API_RESULT: CompletionResult = CompletionResult(
    generated="output text",
    tool_calls="",
    reasoning="",
    finish_reason="stop",
    prompt_tokens=10,
    completion_tokens=5,
)


# ── make_fieldnames ────────────────────────────────────────────────────────────

def test_make_fieldnames_prompt_mode_no_extra():
    raw = {"id": "1", "type": "qa", "prompt": "hello"}
    fields = make_fieldnames(raw, prompt_field="prompt")
    assert fields == FIXED_COLUMNS


def test_make_fieldnames_prompt_mode_with_extra():
    raw = {"id": "1", "type": "qa", "prompt": "hello", "category": "math", "source": "wiki"}
    fields = make_fieldnames(raw, prompt_field="prompt")
    assert "category" in fields
    assert "source" in fields
    assert fields.index("category") > fields.index("completion_tokens")


def test_make_fieldnames_messages_mode_excludes_messages_tools():
    raw = {"id": "1", "type": "chat", "messages": [], "tools": [], "tool_choice": "auto", "tag": "x"}
    fields = make_fieldnames(raw, prompt_field="messages")
    assert "messages" not in fields
    assert "tools" not in fields
    assert "tool_choice" not in fields
    assert "tag" in fields


# ── build_row ─────────────────────────────────────────────────────────────────

def test_build_row_basic():
    raw = {"id": "42", "type": "qa", "prompt": "hello"}
    row = build_row(raw, API_RESULT, SAMPLE_CONFIG, prompt_field="prompt")

    assert row["id"] == "42"
    assert row["type"] == "qa"
    assert row["prompt"] == "hello"
    assert row["generated"] == "output text"
    assert row["tool_calls"] == ""
    assert row["reasoning"] == ""
    assert row["model"] == "test-model"
    assert row["finish_reason"] == "stop"
    assert row["prompt_tokens"] == 10
    assert row["completion_tokens"] == 5


def test_build_row_with_extra_fields():
    raw = {"id": "1", "type": "cls", "prompt": "text", "category": "sci", "difficulty": "hard"}
    row = build_row(raw, API_RESULT, SAMPLE_CONFIG, prompt_field="prompt")
    assert row["category"] == "sci"
    assert row["difficulty"] == "hard"


def test_build_row_missing_type_defaults_empty():
    raw = {"id": "1", "prompt": "hello"}
    row = build_row(raw, API_RESULT, SAMPLE_CONFIG, prompt_field="prompt")
    assert row["type"] == ""


def test_build_row_messages_mode_serializes_messages():
    messages = [{"role": "user", "content": "hi"}]
    raw = {"id": "1", "type": "chat", "messages": messages}
    row = build_row(raw, API_RESULT, SAMPLE_CONFIG, prompt_field="messages")
    import orjson
    assert row["prompt"] == orjson.dumps(messages).decode()


def test_build_row_complex_extra_field_serialized_to_json():
    raw = {"id": "1", "prompt": "hi", "meta": {"key": "value"}}
    row = build_row(raw, API_RESULT, SAMPLE_CONFIG, prompt_field="prompt")
    import orjson
    assert row["meta"] == orjson.dumps({"key": "value"}).decode()


def test_build_row_scalar_extra_field_not_serialized():
    raw = {"id": "1", "prompt": "hi", "score": 0.9, "count": 3, "flag": True}
    row = build_row(raw, API_RESULT, SAMPLE_CONFIG, prompt_field="prompt")
    assert row["score"] == 0.9
    assert row["count"] == 3
    assert row["flag"] is True
