import pytest
import orjson
from pathlib import Path

from src.loader import load_jsonl


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "wb") as f:
        for row in rows:
            f.write(orjson.dumps(row) + b"\n")


# ── prompt mode (is_chat=False) ───────────────────────────────────────────────

def test_prompt_mode_loads_valid_rows(tmp_path):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [
        {"id": "1", "prompt": "hello"},
        {"id": "2", "prompt": "world", "extra": "ok"},
    ])
    rows = load_jsonl(str(f), is_chat=False)
    assert len(rows) == 2
    assert rows[1]["extra"] == "ok"


def test_prompt_mode_skips_missing_prompt(tmp_path, capsys):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [{"id": "1"}, {"id": "2", "prompt": "ok"}])
    rows = load_jsonl(str(f), is_chat=False)
    assert len(rows) == 1
    assert "WARN" in capsys.readouterr().out


# ── messages mode (is_chat=True) ──────────────────────────────────────────────

def test_messages_mode_loads_valid_rows(tmp_path):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [
        {"id": "1", "messages": [{"role": "user", "content": "hi"}]},
        {"id": "2", "messages": [{"role": "user", "content": "hello"}]},
    ])
    rows = load_jsonl(str(f), is_chat=True)
    assert len(rows) == 2


def test_messages_mode_skips_missing_messages(tmp_path, capsys):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [
        {"id": "1", "prompt": "no messages field"},
        {"id": "2", "messages": [{"role": "user", "content": "ok"}]},
    ])
    rows = load_jsonl(str(f), is_chat=True)
    assert len(rows) == 1
    assert "WARN" in capsys.readouterr().out


def test_messages_mode_skips_non_list_messages(tmp_path, capsys):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [
        {"id": "1", "messages": "not a list"},
        {"id": "2", "messages": [{"role": "user", "content": "ok"}]},
    ])
    rows = load_jsonl(str(f), is_chat=True)
    assert len(rows) == 1
    assert "WARN" in capsys.readouterr().out


# ── common behavior ───────────────────────────────────────────────────────────

def test_auto_id_assigned_when_missing(tmp_path):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [{"prompt": "no id"}, {"id": "1", "prompt": "ok"}, {"prompt": "also no id"}])
    rows = load_jsonl(str(f), is_chat=False)
    assert rows[0]["id"] == "auto-1"
    assert rows[1]["id"] == "1"
    assert rows[2]["id"] == "auto-2"


def test_skips_invalid_json(tmp_path, capsys):
    f = tmp_path / "data.jsonl"
    f.write_bytes(b'{"id":"1","prompt":"ok"}\nnot-json\n{"id":"2","prompt":"ok"}\n')
    rows = load_jsonl(str(f), is_chat=False)
    assert len(rows) == 2
    assert "WARN" in capsys.readouterr().out


def test_empty_lines_skipped(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_bytes(b'\n{"id":"1","prompt":"ok"}\n\n')
    rows = load_jsonl(str(f), is_chat=False)
    assert len(rows) == 1


def test_skips_non_object_lines(tmp_path, capsys):
    f = tmp_path / "data.jsonl"
    f.write_bytes(b'["array","not","object"]\n{"id":"1","prompt":"ok"}\n')
    rows = load_jsonl(str(f), is_chat=False)
    assert len(rows) == 1
    assert "WARN" in capsys.readouterr().out


def test_file_not_found_exits():
    with pytest.raises(SystemExit) as exc:
        load_jsonl("/nonexistent/path/data.jsonl", is_chat=False)
    assert exc.value.code == 1
