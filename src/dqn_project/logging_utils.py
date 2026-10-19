"""Append-only CSV metrics with one project-wide schema."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


METRIC_FIELDS = [
    "run_id",
    "seed",
    "environment_steps",
    "frames",
    "episode",
    "episode_reward",
    "episode_length",
    "evaluation_reward_mean",
    "evaluation_reward_std",
    "loss",
    "epsilon",
    "wall_time_seconds",
    "config_id",
    "replay_type",
    "event",
]


class MetricsLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writeheader()

    def log(self, **values: Any) -> None:
        unknown = set(values) - set(METRIC_FIELDS)
        if unknown:
            raise ValueError(f"unknown metric fields: {sorted(unknown)}")
        row = {field: values.get(field, "") for field in METRIC_FIELDS}
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writerow(row)
