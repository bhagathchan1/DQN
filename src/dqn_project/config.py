"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    required = {"run", "environment", "training", "replay", "evaluation"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing configuration sections: {sorted(missing)}")
    return config
