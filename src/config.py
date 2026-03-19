import sys
from pathlib import Path
from typing import Any, TypedDict, cast

import yaml


class ApiConfig(TypedDict):
    base_url: str
    model: str
    concurrency: int


class GenerationConfig(TypedDict):
    max_tokens: int
    temperature: float
    top_p: float
    seed: int
    repetition_penalty: float


class InputConfig(TypedDict):
    prompt_field: str


class Config(TypedDict):
    api: ApiConfig
    generation: GenerationConfig
    input: InputConfig


REQUIRED_KEYS: list[tuple[str, str]] = [
    ("api", "base_url"),
    ("api", "model"),
    ("generation", "max_tokens"),
    ("generation", "temperature"),
    ("generation", "seed"),
]

DEFAULTS: dict[tuple[str, str], int | float | str] = {
    ("generation", "top_p"): 1.0,
    ("generation", "repetition_penalty"): 1.0,
    ("input", "prompt_field"): "prompt",
    ("api", "concurrency"): 8,  # max simultaneous requests to vLLM
}


def load_config(config_path: str) -> Config:
    path = Path(config_path)
    if not path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        loaded: object = yaml.safe_load(f)

    if not isinstance(loaded, dict):
        print("[ERROR] config.yml is empty or invalid.")
        sys.exit(1)

    # Cast immediately after isinstance check to avoid dict[Unknown, Unknown] propagation
    cfg = cast(dict[str, Any], loaded)

    missing: list[str] = []
    for section, key in REQUIRED_KEYS:
        section_data: Any = cfg.get(section)
        if not isinstance(section_data, dict) or cast(dict[str, Any], section_data).get(key) is None:
            missing.append(f"{section}.{key}")

    if missing:
        print(f"[ERROR] Missing required config keys: {', '.join(missing)}")
        sys.exit(1)

    for (section, key), default in DEFAULTS.items():
        if not isinstance(cfg.get(section), dict):
            cfg[section] = {}
        section_cfg = cast(dict[str, Any], cfg[section])
        if section_cfg.get(key) is None:
            section_cfg[key] = default

    return cast(Config, cfg)
