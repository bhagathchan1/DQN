"""Compact learning curves from ``trained_models/metrics`` for the lab's training page.

The raw CSVs hold one row per episode (tens of thousands of rows). The browser only
needs the shape of each run, so episodes are smoothed with a trailing 100-episode
mean and thinned to a few hundred points. Evaluation rows are returned as-is.
"""
from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path

METRICS_DIR = Path(__file__).resolve().parents[2] / 'trained_models' / 'metrics'
WINDOW = 100
MAX_POINTS = 240


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def smooth(points: list[tuple[float, float]], window: int = WINDOW, max_points: int = MAX_POINTS) -> list[list[float]]:
    """Trailing mean over ``window`` values, thinned evenly to at most ``max_points``.

    The first point is kept from the start of training, so curves begin where the run
    began rather than after the first full window.
    """
    if not points:
        return []
    smoothed, total = [], 0.0
    for index, (step, value) in enumerate(points):
        total += value
        if index >= window:
            total -= points[index - window][1]
        smoothed.append((step, total / min(index + 1, window)))
    stride = max(1, math.ceil(len(smoothed) / max_points))
    thinned = smoothed[::stride]
    if thinned[-1] is not smoothed[-1]:
        thinned.append(smoothed[-1])
    return [[round(step), round(value, 3)] for step, value in thinned]


def _episode_curve(path: Path) -> list[list[float]]:
    points = []
    for row in _rows(path):
        if row.get('event') != 'episode':
            continue
        step, reward = _number(row.get('environment_steps')), _number(row.get('episode_reward'))
        if step is not None and reward is not None:
            points.append((step, reward))
    return smooth(points)


def _validation(path: Path) -> list[list[float]]:
    result = []
    for row in _rows(path):
        if row.get('event') in {'validation', 'evaluation'}:
            step, mean = _number(row.get('environment_steps')), _number(row.get('evaluation_reward_mean'))
            if step is not None and mean is not None:
                result.append([round(step), round(mean, 3)])
    return result


def _lab_curves(path: Path) -> tuple[list[list[float]], list[list[float]]]:
    games, evals = [], []
    for row in _rows(path):
        step = _number(row.get('step'))
        if step is None:
            continue
        if row.get('event') == 'game' and _number(row.get('return')) is not None:
            games.append((step, _number(row.get('return'))))
        elif row.get('event') == 'eval' and _number(row.get('eval_mean')) is not None:
            evals.append([round(step), round(_number(row.get('eval_mean')), 3)])
    return smooth(games), evals


@lru_cache(maxsize=4)
def load_curves(directory: str | None = None) -> dict:
    root = Path(directory) if directory else METRICS_DIR
    lab_nature_games, lab_nature_eval = _lab_curves(root / 'pong_lab_nature_metrics.csv')
    lab_dueling_games, lab_dueling_eval = _lab_curves(root / 'pong_lab_dueling_metrics.csv')
    return {
        'window': WINDOW,
        'pong_single': {
            'training': {'nature': _episode_curve(root / 'pong_nature_seed7.csv'),
                         'dueling': _episode_curve(root / 'pong_dueling_seed7.csv')},
            'validation': {'nature': _validation(root / 'pong_nature_seed7.csv'),
                           'dueling': _validation(root / 'pong_dueling_seed7.csv')},
        },
        'pong_two_player': {
            'training': {'nature': lab_nature_games, 'dueling': lab_dueling_games},
            'evaluation': {'nature': lab_nature_eval, 'dueling': lab_dueling_eval},
        },
        'breakout': {
            'training': {'uniform': _episode_curve(root / 'breakout_uniform_10M_seed7.csv'),
                         'prioritized': _episode_curve(root / 'breakout_prioritized_10M_seed7.csv'),
                         'upstream': _episode_curve(root / 'breakout_upstream_10M_seed7.csv')},
        },
    }
