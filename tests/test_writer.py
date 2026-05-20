import csv
from typing import cast
import pytest
import orjson
from pathlib import Path

from src.config import Config
from src.writer import CsvWriter, JsonlWriter, build_comment

SAMPLE_CONFIG: Config = cast(Config, {
    "api": {"model": "test-model"},
    "generation": {
        "max_tokens": 100,
        "temperature": 0.1,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "seed": 42,
    },
})


# ── build_comment ─────────────────────────────────────────────────────────────

def test_build_comment_format():
    comment = build_comment(SAMPLE_CONFIG, "./test.jsonl")
    assert comment.startswith("#")
    assert "model=test-model" in comment
    assert "max_tokens=100" in comment
    assert "temperature=0.1" in comment
    assert "seed=42" in comment
    assert "input=./test.jsonl" in comment


# ── CsvWriter ─────────────────────────────────────────────────────────────────

def test_csv_writer_writes_comment_and_header(tmp_path):
    out = tmp_path / "result.csv"
    fields = ["id", "prompt", "generated"]

    with CsvWriter(SAMPLE_CONFIG, fields, str(out), "./in.jsonl") as w:
        w.write_row({"id": "1", "prompt": "hi", "generated": "hello"})

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("#")
    assert lines[1] == "id,prompt,generated"
    assert "hello" in lines[2]


def test_csv_writer_row_count(tmp_path):
    out = tmp_path / "result.csv"

    with CsvWriter(SAMPLE_CONFIG, ["id", "generated"], str(out), "./in.jsonl") as w:
        w.write_row({"id": "1", "generated": "a"})
        w.write_row({"id": "2", "generated": "b"})
        assert w._count == 2


def test_csv_writer_multiple_rows_parseable(tmp_path):
    out = tmp_path / "result.csv"
    fields = ["id", "generated"]
    data = [{"id": str(i), "generated": f"text {i}"} for i in range(5)]

    with CsvWriter(SAMPLE_CONFIG, fields, str(out), "./in.jsonl") as w:
        for row in data:
            w.write_row(row)

    content = out.read_text(encoding="utf-8")
    lines = [l for l in content.splitlines() if not l.startswith("#")]
    reader = list(csv.DictReader(lines))
    assert len(reader) == 5
    assert reader[2]["id"] == "2"


def test_csv_writer_invalid_path_exits():
    with pytest.raises(SystemExit) as exc:
        with CsvWriter(SAMPLE_CONFIG, ["id"], "/nonexistent_dir_xyz/result.csv", "./in.jsonl"):
            pass
    assert exc.value.code == 1


# ── JsonlWriter ───────────────────────────────────────────────────────────────

def test_jsonl_writer_writes_valid_jsonl(tmp_path):
    out = tmp_path / "result.jsonl"
    fields = ["id", "generated"]
    rows = [{"id": "1", "generated": "a"}, {"id": "2", "generated": "b"}]

    with JsonlWriter(SAMPLE_CONFIG, fields, str(out), "./in.jsonl") as w:
        for row in rows:
            w.write_row(row)

    lines = out.read_bytes().splitlines()
    assert len(lines) == 2
    assert orjson.loads(lines[0])["generated"] == "a"
    assert orjson.loads(lines[1])["id"] == "2"


def test_jsonl_writer_row_count(tmp_path):
    out = tmp_path / "result.jsonl"

    with JsonlWriter(SAMPLE_CONFIG, ["id"], str(out), "./in.jsonl") as w:
        w.write_row({"id": "1"})
        w.write_row({"id": "2"})
        assert w._count == 2


def test_jsonl_writer_invalid_path_exits():
    with pytest.raises(SystemExit) as exc:
        with JsonlWriter(SAMPLE_CONFIG, ["id"], "/nonexistent_dir_xyz/result.jsonl", "./in.jsonl"):
            pass
    assert exc.value.code == 1


def test_jsonl_writer_preserves_nested_values(tmp_path):
    out = tmp_path / "result.jsonl"
    row = {"id": "1", "tool_calls": [{"name": "fn", "args": {}}]}

    with JsonlWriter(SAMPLE_CONFIG, ["id", "tool_calls"], str(out), "./in.jsonl") as w:
        w.write_row(row)

    parsed = orjson.loads(out.read_bytes().splitlines()[0])
    assert parsed["tool_calls"][0]["name"] == "fn"
