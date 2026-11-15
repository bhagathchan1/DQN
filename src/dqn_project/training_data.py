"""Real DQN training data for the lab's explainer page.

A DQN has no prepared dataset: its data is the experience it records while playing.
``sample_experience`` plays a short stretch of real Atari (with a trained model, or
randomly as at the start of training) and returns transitions exactly as replay
memory stores them: the 4 stacked 84x84 frames, the button, the reward and the next
frame. ``TrainingRecords`` pages through the per-game logs written during training.
"""
from __future__ import annotations

import base64
import csv
import io
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / 'trained_models'
GAMES = {
    'pong': ('ALE/Pong-v5', 'pong_single/pong_nature_seed7_best.pt',
             'Single-player Pong, as in pre-training: the DQN is the green paddle on the right.'),
    'breakout': ('ALE/Breakout-v5', 'breakout/breakout_uniform_10M_seed7_final.pt',
                 'Breakout, as in training: the DQN moves the paddle at the bottom.'),
}
POLICIES = {'random': 1.0, 'trained': 0.05}
ACTION_NAMES = {'NOOP': 'Do nothing', 'FIRE': 'Fire / serve', 'RIGHT': 'Right', 'LEFT': 'Left',
                'RIGHTFIRE': 'Right + fire', 'LEFTFIRE': 'Left + fire', 'UP': 'Up', 'DOWN': 'Down'}
SAMPLE_DECISIONS = 600
SHOWN = 8

_lock = threading.Lock()
_cache: dict[tuple, dict] = {}


def _png(array: np.ndarray) -> str:
    from PIL import Image
    stream = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(stream, 'PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def _pick(rewards: list[float], shown: int = SHOWN) -> list[int]:
    """All reward moments (up to half the cards) plus evenly spaced ordinary ones."""
    events = [i for i, r in enumerate(rewards) if r != 0]
    if len(events) > shown // 2:
        events = [events[round(k * (len(events) - 1) / (shown // 2 - 1))] for k in range(shown // 2)]
    quiet = [i for i, r in enumerate(rewards) if r == 0]
    need = shown - len(events)
    if quiet and need > 0:
        quiet = [quiet[round(k * (len(quiet) - 1) / max(need - 1, 1))] for k in range(min(need, len(quiet)))]
    return sorted(set(events) | set(quiet))


def sample_experience(game: str, policy: str, seed: int = 7, round_factory=None) -> dict:
    if game not in GAMES or policy not in POLICIES:
        raise ValueError('game must be pong or breakout and policy random or trained')
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 10_000:
        raise ValueError('seed must be an integer in [0, 10000)')
    key = (game, policy, seed)
    with _lock:
        if key in _cache:
            return _cache[key]
    env_id, checkpoint, description = GAMES[game]
    if round_factory is None:
        import torch
        from .arena import ArenaRound
        torch.set_num_threads(1)
        round_factory = ArenaRound
    round_ = round_factory(env_id, 'ai', seed, str(ROOT / checkpoint), max_steps=SAMPLE_DECISIONS,
                           epsilon=POLICIES[policy])
    try:
        states, actions, rewards, successors, ends = [], [], [], [], []
        while not round_.done:
            before = np.array(round_.state, dtype=np.uint8, copy=True)
            round_.step()
            states.append(before)
            actions.append(round_.last_action)
            rewards.append(round_.last_reward)
            successors.append(np.array(round_.state[-1], dtype=np.uint8, copy=True))
            ends.append(bool(round_.done and round_.reason == 'game_over'))
        meanings = list(round_.meanings)
        score = float(round_.score)
    finally:
        round_.close()
    counts = np.bincount(actions, minlength=len(meanings))
    transitions = []
    for index in _pick(rewards):
        transitions.append({
            'step': index + 1,
            'frames': _png(np.concatenate(list(states[index]), axis=1)),
            'next_frame': _png(successors[index]),
            'action': meanings[actions[index]],
            'action_label': ACTION_NAMES.get(meanings[actions[index]], meanings[actions[index]]),
            'reward': rewards[index],
            'training_reward': float(np.sign(rewards[index])),
            'game_over': ends[index],
        })
    result = {
        'game': game, 'policy': policy, 'seed': seed, 'description': description,
        'epsilon': POLICIES[policy], 'decisions': len(actions), 'score': score,
        'positive': int(sum(r > 0 for r in rewards)), 'negative': int(sum(r < 0 for r in rewards)),
        'actions': [{'action': name, 'label': ACTION_NAMES.get(name, name), 'count': int(counts[i])}
                    for i, name in enumerate(meanings)],
        'transitions': transitions,
    }
    with _lock:
        _cache[key] = result
    return result


# ----- the per-game training logs -------------------------------------------------------
RUNS = {
    'pong_nature': ('metrics/pong_nature_seed7.csv', 'Pong pre-training · Nature DQN', 'episodes'),
    'pong_dueling': ('metrics/pong_dueling_seed7.csv', 'Pong pre-training · Dueling DQN', 'episodes'),
    'pong_lab_nature': ('metrics/pong_lab_nature_metrics.csv', 'Pong two-player fine-tune · Nature DQN', 'lab'),
    'pong_lab_dueling': ('metrics/pong_lab_dueling_metrics.csv', 'Pong two-player fine-tune · Dueling DQN', 'lab'),
    'breakout_uniform': ('metrics/breakout_uniform_10M_seed7.csv', 'Breakout · uniform replay', 'episodes'),
    'breakout_prioritized': ('metrics/breakout_prioritized_10M_seed7.csv', 'Breakout · prioritized replay', 'episodes'),
    'breakout_upstream': ('metrics/breakout_upstream_10M_seed7.csv', 'Breakout · upstream setup', 'episodes'),
}
COLUMNS = {
    'episodes': [('event', 'Row'), ('episode', 'Game #'), ('environment_steps', 'Decisions so far'),
                 ('episode_reward', 'Score'), ('episode_length', 'Game length'),
                 ('evaluation_reward_mean', 'Test score'), ('epsilon', 'ε'), ('loss', 'Loss'),
                 ('wall_time_seconds', 'Time (s)')],
    'lab': [('event', 'Row'), ('game', 'Game #'), ('step', 'Decisions so far'), ('return', 'Score'),
            ('decisions', 'Game length'), ('eval_mean', 'Test score'), ('epsilon', 'ε'), ('loss', 'Loss'),
            ('mean_q', 'Mean Q'), ('wall_seconds', 'Time (s)')],
}


def run_path(run: str) -> Path:
    if run not in RUNS:
        raise ValueError('Unknown training run.')
    return ROOT / RUNS[run][0]


@lru_cache(maxsize=8)
def _load(run: str) -> tuple[list[str], list[list[str]]]:
    kind = RUNS[run][2]
    keys = [key for key, _ in COLUMNS[kind]]
    with run_path(run).open(newline='', encoding='utf-8') as stream:
        rows = [[row.get(key, '') or '' for key in keys] for row in csv.DictReader(stream)]
    return keys, rows


def _short(value: str) -> str:
    try:
        number = float(value)
    except ValueError:
        return value
    if number != number:  # NaN, e.g. no loss before learning starts
        return ''
    # Small values keep 4 significant digits; large ones (steps, seconds) print as whole numbers.
    return f'{number:.4g}' if abs(number) < 1000 else f'{number:.0f}'


def list_runs() -> list[dict]:
    return [{'id': run, 'label': label, 'kind': kind} for run, (_, label, kind) in RUNS.items()]


def records(run: str, offset: int = 0, limit: int = 50) -> dict:
    if not isinstance(offset, int) or not isinstance(limit, int) or offset < 0 or not 1 <= limit <= 200:
        raise ValueError('offset must be >= 0 and limit between 1 and 200')
    run_path(run)  # rejects unknown runs before anything is read
    keys, rows = _load(run)
    kind = RUNS[run][2]
    games = [r for r in rows if r[0] in ('episode', 'game')]
    steps_index = keys.index('environment_steps' if kind == 'episodes' else 'step')
    time_index = keys.index('wall_time_seconds' if kind == 'episodes' else 'wall_seconds')
    def last_number(index):
        for row in reversed(rows):
            try:
                return float(row[index])
            except ValueError:
                continue
        return None
    total_seconds = last_number(time_index)
    return {
        'run': run, 'label': RUNS[run][1], 'total': len(rows), 'offset': offset,
        'columns': [label for _, label in COLUMNS[kind]],
        'rows': [[_short(value) for value in row] for row in rows[offset:offset + limit]],
        'summary': {'rows': len(rows), 'games': len(games), 'decisions': last_number(steps_index),
                    'hours': round(total_seconds / 3600, 2) if total_seconds else None},
    }
