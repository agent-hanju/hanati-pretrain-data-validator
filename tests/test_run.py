from run import build_row, make_fieldnames

SAMPLE_CONFIG = {
    "api": {"model": "test-model"},
    "generation": {},
    "input": {"file": "./data.jsonl"},
    "output": {"file": "./out.csv"},
}

API_RESULT = {
    "generated": "output text",
    "finish_reason": "stop",
    "prompt_tokens": 10,
    "completion_tokens": 5,
}


def test_make_fieldnames_no_extra():
    raw = {"id": "1", "type": "qa", "prompt": "hello"}
    fields = make_fieldnames(raw, prompt_field="prompt")
    assert fields == ["id", "type", "prompt", "generated", "model",
                      "finish_reason", "prompt_tokens", "completion_tokens"]


def test_make_fieldnames_with_extra():
    raw = {"id": "1", "type": "qa", "prompt": "hello", "category": "math", "source": "wiki"}
    fields = make_fieldnames(raw, prompt_field="prompt")
    assert "category" in fields
    assert "source" in fields
    # extra columns come after fixed columns
    assert fields.index("category") > fields.index("completion_tokens")


def test_build_row_basic():
    raw = {"id": "42", "type": "qa", "prompt": "hello"}
    row = build_row(raw, API_RESULT, SAMPLE_CONFIG, prompt_field="prompt")

    assert row["id"] == "42"
    assert row["type"] == "qa"
    assert row["prompt"] == "hello"
    assert row["generated"] == "output text"
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


def test_build_row_custom_prompt_field():
    raw = {"id": "1", "text": "custom field value"}
    row = build_row(raw, API_RESULT, SAMPLE_CONFIG, prompt_field="text")
    assert row["prompt"] == "custom field value"
