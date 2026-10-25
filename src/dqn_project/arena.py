"""Shared rules for scored human and checkpoint-controlled Atari rounds."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .agent import select_action
from .environment import make_env
from .model import DQN


def keyboard_action(meanings, *, up=False, down=False, left=False, right=False, fire=False):
    """Translate held keys to the game's minimal ALE action set."""
    vertical = 'UP' if up and not down else 'DOWN' if down and not up else ''
    horizontal = 'LEFT' if left and not right else 'RIGHT' if right and not left else ''
    direction = vertical + horizontal
    candidates = [direction + ('FIRE' if fire else ''), direction, 'FIRE' if fire else 'NOOP', 'NOOP']
    for name in candidates:
        if name in meanings:
            return meanings.index(name)
    return 0


class ArenaRound:
    def __init__(self, env_id, mode, seed, checkpoint=None, max_steps=4500, epsilon=0.05, noop_max=30):
        if mode not in {'human', 'ai'} or max_steps <= 0 or not 0 <= epsilon <= 1:
            raise ValueError('invalid round settings')
        self.env_id, self.mode, self.seed = env_id, mode, seed
        self.max_steps, self.epsilon, self.noop_max = max_steps, epsilon, noop_max
        self.checkpoint = str(Path(checkpoint).resolve()) if checkpoint else ''
        self.network = None
        self.checkpoint_steps = None
        if mode == 'ai':
            if not checkpoint:
                raise ValueError('Select a trained checkpoint before starting AI.')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            metadata = payload.get('metadata', {})
            if metadata.get('environment_id') != env_id:
                raise ValueError('Checkpoint game does not match the selected game.')
            self.network = DQN(payload['action_count'], payload.get('input_channels', 4))
            self.network.load_state_dict(payload['online'])
            self.network.eval()
            self.checkpoint_steps = metadata.get('environment_steps')
        self.env = make_env(env_id, render_mode='rgb_array', noop_max=noop_max)
        try:
            if self.network and self.network.action_count != self.env.action_space.n:
                raise ValueError('Checkpoint action count does not match the game.')
            self.state, _ = self.env.reset(seed=seed)
            self.meanings = self.env.env.unwrapped.get_action_meanings()
        except BaseException:
            self.env.close()
            raise
        self.rng = np.random.default_rng(seed)
        self.score = 0.0
        self.steps = 0
        self.frames = 0
        self.done = False
        self.reason = ''

    def step(self, human_action=0):
        if self.done:
            return
        action = (select_action(self.network, self.state, self.env.action_space.n,
                                self.epsilon, self.rng, 'cpu') if self.mode == 'ai' else human_action)
        self.state, reward, terminated, truncated, info = self.env.step(action)
        self.score += reward
        self.steps += 1
        self.frames += info['emulator_frames']
        self.done = terminated or truncated or self.steps >= self.max_steps
        if self.done:
            self.reason = 'game_over' if terminated else 'time_limit'

    def frame(self):
        return self.env.env.render()

    def result(self):
        return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), game=self.env_id,
                    mode=self.mode, seed=self.seed, score=self.score, decisions=self.steps,
                    emulator_frames=self.frames, max_steps=self.max_steps, epsilon=self.epsilon,
                    noop_max=self.noop_max, checkpoint=self.checkpoint,
                    checkpoint_training_steps=self.checkpoint_steps,
                    completed=self.done, reason=self.reason or 'aborted')

    def close(self):
        self.env.close()


def append_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        if not exists:
            writer.writeheader()
        writer.writerow(result)


def paired_scores(results, game, checkpoint):
    """Latest completed attempt per seed; compare only the same game and protocol."""
    paired = {}
    for row in results:
        if row['game'] != game or row['checkpoint'] != checkpoint or not row['completed']:
            continue
        key = (row['seed'], row['max_steps'], row['epsilon'], row['noop_max'])
        paired.setdefault(key, {})[row['mode']] = row['score']
    return [(key[0], value['human'], value['ai']) for key, value in paired.items()
            if 'human' in value and 'ai' in value]
