import csv
import pytest
from pathlib import Path

from src.writer import CsvWriter, build_comment

SAMPLE_CONFIG = {
    "api": {"model": "test-model"},
    "generation": {
        "max_tokens": 100,
        "temperature": 0.1,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "seed": 42,
    },
    "input": {"file": "./test.jsonl"},
    "output": {"file": ""},  # filled per test
}


def make_config(output_path: str) -> dict:
    cfg = {k: dict(v) for k, v in SAMPLE_CONFIG.items()}
    cfg["output"]["file"] = output_path
    return cfg


def test_build_comment_format():
    comment = build_comment(SAMPLE_CONFIG)
    assert comment.startswith("#")
    assert "model=test-model" in comment
    assert "max_tokens=100" in comment
    assert "temperature=0.1" in comment
    assert "seed=42" in comment
    assert "input=./test.jsonl" in comment


def test_csv_writer_writes_comment_and_header(tmp_path):
    out = tmp_path / "result.csv"
    cfg = make_config(str(out))
    fields = ["id", "prompt", "generated"]

    with CsvWriter(cfg, fields) as w:
        w.write_row({"id": "1", "prompt": "hi", "generated": "hello"})

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("#")
    assert lines[1] == "id,prompt,generated"
    assert "1" in lines[2]
    assert "hello" in lines[2]


def test_csv_writer_row_count(tmp_path):
    out = tmp_path / "result.csv"
    cfg = make_config(str(out))

    with CsvWriter(cfg, ["id", "generated"]) as w:
        w.write_row({"id": "1", "generated": "a"})
        w.write_row({"id": "2", "generated": "b"})
        assert w._count == 2


def test_csv_writer_multiple_rows_parseable(tmp_path):
    out = tmp_path / "result.csv"
    cfg = make_config(str(out))
    fields = ["id", "generated"]
    data = [{"id": str(i), "generated": f"text {i}"} for i in range(5)]

    with CsvWriter(cfg, fields) as w:
        for row in data:
            w.write_row(row)

    # skip comment line, parse as CSV
    content = out.read_text(encoding="utf-8")
    lines = [l for l in content.splitlines() if not l.startswith("#")]
    reader = list(csv.DictReader(lines))
    assert len(reader) == 5
    assert reader[2]["id"] == "2"


def test_csv_writer_invalid_path_exits():
    cfg = make_config("/nonexistent_dir_xyz/result.csv")
    with pytest.raises(SystemExit) as exc:
        with CsvWriter(cfg, ["id"]):
            pass
    assert exc.value.code == 1
