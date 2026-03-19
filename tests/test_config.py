import pytest
import yaml

from src.config import load_config

VALID_CONFIG = {
    "api": {"base_url": "http://localhost:8000/v1", "model": "my-model"},
    "generation": {"max_tokens": 512, "temperature": 0.7, "seed": 0},
    "input": {"file": "./data.jsonl"},
    "output": {"file": "./out.csv"},
}


def write_yaml(path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def test_valid_config(tmp_path):
    p = tmp_path / "config.yml"
    write_yaml(p, VALID_CONFIG)
    cfg = load_config(str(p))
    assert cfg["api"]["model"] == "my-model"
    assert cfg["generation"]["max_tokens"] == 512


def test_defaults_applied(tmp_path):
    p = tmp_path / "config.yml"
    write_yaml(p, VALID_CONFIG)
    cfg = load_config(str(p))
    assert cfg["generation"]["top_p"] == 1.0
    assert cfg["generation"]["repetition_penalty"] == 1.0
    assert cfg["input"]["prompt_field"] == "prompt"
    assert cfg["api"]["concurrency"] == 8


def test_explicit_values_not_overwritten_by_defaults(tmp_path):
    custom = {k: dict(v) for k, v in VALID_CONFIG.items()}
    custom["generation"]["top_p"] = 0.8
    custom["api"]["concurrency"] = 16
    p = tmp_path / "config.yml"
    write_yaml(p, custom)
    cfg = load_config(str(p))
    assert cfg["generation"]["top_p"] == 0.8
    assert cfg["api"]["concurrency"] == 16


def test_missing_model_exits(tmp_path):
    bad = {k: dict(v) for k, v in VALID_CONFIG.items()}
    del bad["api"]["model"]
    p = tmp_path / "config.yml"
    write_yaml(p, bad)
    with pytest.raises(SystemExit) as exc:
        load_config(str(p))
    assert exc.value.code == 1


def test_missing_base_url_exits(tmp_path):
    bad = {k: dict(v) for k, v in VALID_CONFIG.items()}
    del bad["api"]["base_url"]
    p = tmp_path / "config.yml"
    write_yaml(p, bad)
    with pytest.raises(SystemExit) as exc:
        load_config(str(p))
    assert exc.value.code == 1


def test_missing_section_exits(tmp_path):
    bad = {k: dict(v) for k, v in VALID_CONFIG.items()}
    del bad["output"]
    p = tmp_path / "config.yml"
    write_yaml(p, bad)
    with pytest.raises(SystemExit) as exc:
        load_config(str(p))
    assert exc.value.code == 1


def test_file_not_found_exits():
    with pytest.raises(SystemExit) as exc:
        load_config("/no/such/file.yml")
    assert exc.value.code == 1


def test_empty_yaml_exits(tmp_path):
    p = tmp_path / "config.yml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        load_config(str(p))
    assert exc.value.code == 1
