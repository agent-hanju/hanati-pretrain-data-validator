import pytest
import yaml

from src.config import load_config

VALID_CONFIG = {
    "api": {"base_url": "http://localhost:8000/v1", "model": "my-model"},
    "generation": {"max_tokens": 512, "temperature": 0.7, "seed": 0},
}


def write_yaml(path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def test_valid_config_loads(tmp_path):
    p = tmp_path / "config.yml"
    write_yaml(p, VALID_CONFIG)
    cfg = load_config(str(p))
    assert cfg["api"]["model"] == "my-model"
    assert cfg["generation"].get("max_tokens") == 512


def test_defaults_applied(tmp_path):
    p = tmp_path / "config.yml"
    write_yaml(p, VALID_CONFIG)
    cfg = load_config(str(p))
    assert cfg["api"]["concurrency"] == 256
    assert cfg["api"]["timeout"] == 120.0


def test_generation_params_absent_when_not_set(tmp_path):
    p = tmp_path / "config.yml"
    cfg_data = {"api": {"base_url": "http://localhost/v1"}, "generation": {}}
    write_yaml(p, cfg_data)
    cfg = load_config(str(p))
    assert cfg["generation"].get("top_p") is None
    assert cfg["generation"].get("top_k") is None
    assert cfg["generation"].get("repetition_penalty") is None
    assert cfg["generation"].get("temperature") is None


def test_explicit_values_not_overwritten_by_defaults(tmp_path):
    custom = {**VALID_CONFIG, "generation": {**VALID_CONFIG["generation"], "top_p": 0.8}}
    custom = {**custom, "api": {**VALID_CONFIG["api"], "concurrency": 16}}
    p = tmp_path / "config.yml"
    write_yaml(p, custom)
    cfg = load_config(str(p))
    assert cfg["generation"].get("top_p") == 0.8
    assert cfg["api"]["concurrency"] == 16


def test_missing_base_url_exits(tmp_path):
    bad = {**VALID_CONFIG, "api": {"model": "x"}}
    p = tmp_path / "config.yml"
    write_yaml(p, bad)
    with pytest.raises(SystemExit) as exc:
        load_config(str(p))
    assert exc.value.code == 1


def test_model_defaults_to_none_when_absent(tmp_path):
    cfg_data = {**VALID_CONFIG, "api": {"base_url": "http://localhost/v1"}}
    p = tmp_path / "config.yml"
    write_yaml(p, cfg_data)
    cfg = load_config(str(p))
    assert cfg["api"]["model"] is None


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
