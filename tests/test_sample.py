import orjson
from pathlib import Path

from src.sampler import _iter_documents, reservoir_sample, extract_snippet, sample


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "wb") as f:
        for row in rows:
            f.write(orjson.dumps(row) + b"\n")


def test_iter_documents(tmp_path):
    f1 = tmp_path / "a.jsonl"
    f2 = tmp_path / "b.jsonl"
    _write_jsonl(f1, [
        {"id": "a1", "text": "hello\nworld"},
        {"id": "a2", "text": "foo"},
    ])
    _write_jsonl(f2, [
        {"text": "no id here"},
    ])
    docs = list(_iter_documents([str(f1), str(f2)], "text"))
    assert len(docs) == 3
    assert docs[0]["id"] == "a1"
    assert docs[2]["id"] == "auto-1"
    assert docs[2]["file"] == str(f2)


def test_iter_skips_missing_file(tmp_path, capsys):
    docs = list(_iter_documents([str(tmp_path / "nope.jsonl")], "text"))
    assert docs == []
    assert "WARN" in capsys.readouterr().out


def test_reservoir_sample_all(tmp_path):
    """When n >= total docs, all documents are returned."""
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [{"id": str(i), "text": f"doc {i}"} for i in range(5)])
    picked = reservoir_sample([str(f)], n=100, text_field="text")
    assert len(picked) == 5


def test_reservoir_sample_subset(tmp_path):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [{"id": str(i), "text": f"doc {i}"} for i in range(1000)])
    picked = reservoir_sample([str(f)], n=10, text_field="text")
    assert len(picked) == 10
    # all picked docs should have valid ids from the original set
    ids = {d["id"] for d in picked}
    assert all(id_ in {str(i) for i in range(1000)} for id_ in ids)


def test_extract_snippet_single_line():
    text = "only one line"
    result = extract_snippet(text)
    assert result == "only one line"


def test_extract_snippet_range():
    text = "\n".join(f"line {i}" for i in range(20))
    result = extract_snippet(text, min_lines=1, max_lines=5)
    lines = result.splitlines()
    assert 1 <= len(lines) <= 5


def test_extract_snippet_empty():
    assert extract_snippet("") == ""


def test_sample_basic(tmp_path):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [
        {"id": str(i), "text": f"line a\nline b\nline c\nline d\nline e"}
        for i in range(20)
    ])
    results = sample([str(f)], n=5, seed=42)
    assert len(results) == 5
    for r in results:
        assert "file" in r
        assert "id" in r
        assert "text" in r
        assert "original" in r
        assert len(r["text"]) > 0
        assert r["original"] == "line a\nline b\nline c\nline d\nline e"


def test_sample_fewer_docs_than_n(tmp_path, capsys):
    f = tmp_path / "small.jsonl"
    _write_jsonl(f, [{"id": "1", "text": "hello"}])
    results = sample([str(f)], n=10, seed=0)
    assert len(results) == 1
    assert "WARN" in capsys.readouterr().out


def test_sample_deterministic(tmp_path):
    f = tmp_path / "data.jsonl"
    _write_jsonl(f, [
        {"id": str(i), "text": f"line {i}\nsecond {i}\nthird {i}"}
        for i in range(50)
    ])
    r1 = sample([str(f)], n=5, seed=123)
    r2 = sample([str(f)], n=5, seed=123)
    assert [r["id"] for r in r1] == [r["id"] for r in r2]
