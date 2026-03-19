import pytest
import orjson
from pathlib import Path

from src.loader import load_jsonl


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "wb") as f:
        for row in rows:
            f.write(orjson.dumps(row) + b"\n")


def test_valid_rows(tmp_path):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [
        {"id": "1", "prompt": "hello"},
        {"id": "2", "prompt": "world", "extra": "ok"},
    ])
    rows = load_jsonl(str(f))
    assert len(rows) == 2
    assert rows[0]["id"] == "1"
    assert rows[1]["extra"] == "ok"


def test_skips_invalid_json(tmp_path, capsys):
    f = tmp_path / "data.jsonl"
    f.write_bytes(b'{"id":"1","prompt":"ok"}\nnot-json\n{"id":"2","prompt":"ok"}\n')
    rows = load_jsonl(str(f))
    assert len(rows) == 2
    assert "WARN" in capsys.readouterr().out


def test_skips_missing_id(tmp_path, capsys):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [{"prompt": "no id"}, {"id": "1", "prompt": "ok"}])
    rows = load_jsonl(str(f))
    assert len(rows) == 1
    assert "WARN" in capsys.readouterr().out


def test_skips_missing_prompt_field(tmp_path, capsys):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [{"id": "1"}, {"id": "2", "prompt": "ok"}])
    rows = load_jsonl(str(f))
    assert len(rows) == 1
    assert "WARN" in capsys.readouterr().out


def test_custom_prompt_field(tmp_path):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [{"id": "1", "text": "hello"}])
    rows = load_jsonl(str(f), prompt_field="text")
    assert len(rows) == 1


def test_empty_lines_skipped(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_bytes(b'\n{"id":"1","prompt":"ok"}\n\n')
    rows = load_jsonl(str(f))
    assert len(rows) == 1


def test_file_not_found_exits():
    with pytest.raises(SystemExit) as exc:
        load_jsonl("/nonexistent/path/data.jsonl")
    assert exc.value.code == 1
