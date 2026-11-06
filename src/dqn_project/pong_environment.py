"""Two-player Atari Pong with one human paddle and one DQN paddle.

The PettingZoo environment advances one emulator frame per parallel step. This
adapter owns the sole four-frame repeat and the machine's four-frame stack.
"""
from __future__ import annotations

from collections import deque
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .environment import preprocess_frame


_ALE_ACTIONS = (
    'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
    'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
    'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE',
)


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return 'unavailable'


def _make_parallel_env() -> Any:
    try:
        from pettingzoo.atari import pong_v3
        import ale_py
    except ImportError as exc:
        raise RuntimeError(
            "Two-player Atari Pong requires the pong extras: pip install -e '.[pong]'. "
            "The multi-agent Atari runtime must support this Python/platform."
        ) from exc
    # Modern ALE wheels include the ROM; let PettingZoo locate that same file.
    rom_directory = Path(ale_py.__file__).parent / 'roms'
    options: dict[str, Any] = dict(
        num_players=2, game_version='classic', render_mode='rgb_array', max_cycles=10_000_000,
    )
    if (rom_directory / 'pong.bin').is_file():
        options['auto_rom_install_path'] = str(rom_directory)
    try:
        return pong_v3.parallel_env(**options)
    except (OSError, ImportError, ValueError, AssertionError) as exc:
        raise RuntimeError(
            'Two-player Atari Pong could not start. Check the PettingZoo/multi-agent-ale-py '
            'installation and the Pong ROM. This demo does not substitute a simulated game. '
            f'Details: {exc}'
        ) from exc


class PongEnvironment:
    """Machine transition interface over a genuine two-player Pong environment.

    `first_0` controls the left paddle and is assigned to the human; `second_0`
    controls the right paddle and is assigned to the machine. Minimal ALE actions
    are resolved from the installed environment rather than single-agent indices.
    `_parallel_env` is a test-only dependency injection point.
    """

    def __init__(self, *, _parallel_env: Any | None = None, action_repeat: int = 4) -> None:
        if action_repeat != 4:
            raise ValueError('the demo protocol uses exactly four emulator frames per decision')
        self.env = _parallel_env if _parallel_env is not None else _make_parallel_env()
        self.raw = getattr(self.env, 'unwrapped', self.env)
        self.action_repeat = action_repeat
        agents = list(self.env.possible_agents)
        if agents != ['first_0', 'second_0']:
            self.env.close()
            raise ValueError(f'expected two Pong players first_0 and second_0, got {agents}')
        self.human_agent, self.machine_agent = agents
        self.action_count = int(self.env.action_space(self.machine_agent).n)
        if int(self.env.action_space(self.human_agent).n) != self.action_count:
            self.env.close()
            raise ValueError('human and machine action spaces must match')
        mapping = np.asarray(getattr(self.raw, 'action_mapping', []), dtype=np.int64)
        if mapping.shape != (self.action_count,) or np.any(mapping < 0) or np.any(mapping >= len(_ALE_ACTIONS)):
            self.env.close()
            raise RuntimeError('installed Atari runtime does not expose a verified ALE action mapping')
        self.action_meanings = [_ALE_ACTIONS[int(action)] for action in mapping]
        self.spec = {
            'environment_family': 'pettingzoo.atari', 'environment_id': 'pong_v3',
            'game_version': 'classic', 'num_players': 2,
            'pettingzoo_version': _version('pettingzoo'),
            'multi_agent_ale_version': _version('multi-agent-ale-py'),
            'human_agent': self.human_agent, 'machine_agent': self.machine_agent,
            'human_side': 'left', 'machine_side': 'right',
            'actions': self.action_meanings, 'ale_action_codes': mapping.tolist(),
            'action_repeat': action_repeat, 'raw_frames_per_parallel_step': 1,
            'preprocessing': 'pillow_luminance_bilinear_84x84_max2_stack4_v1',
            'state_shape': [4, 84, 84], 'score_mode': 'return',
            'goal_events_verified': False,
        }
        self.frames: deque[np.ndarray] = deque(maxlen=4)
        self._rgb: np.ndarray | None = None
        self._ended = True

    def reset(self, seed: int) -> np.ndarray:
        observations, _ = self.env.reset(seed=seed)
        self._rgb = np.asarray(observations[self.machine_agent], dtype=np.uint8).copy()
        processed = preprocess_frame(self._rgb)
        self.frames.clear()
        self.frames.extend(processed.copy() for _ in range(4))
        self._ended = False
        return np.stack(tuple(self.frames))

    def action_for_keys(self, keys: set[str]) -> int:
        """Map vertical controls to installed ALE actions, including optional serve.

        Pong's paddle controller uses RIGHT/LEFT for up/down in its minimal set;
        an installed mapping containing actual UP/DOWN takes precedence.
        """
        up, down, fire = 'up' in keys, 'down' in keys, 'fire' in keys
        direction = ''
        if up and not down:
            direction = 'UP' if 'UP' in self.action_meanings else 'RIGHT'
        elif down and not up:
            direction = 'DOWN' if 'DOWN' in self.action_meanings else 'LEFT'
        candidates = [direction + ('FIRE' if fire else ''), direction,
                      'FIRE' if fire else 'NOOP', 'NOOP']
        for action in candidates:
            if action in self.action_meanings:
                return self.action_meanings.index(action)
        return 0

    def step(self, human_action: int, machine_action: int):
        if self._ended:
            raise RuntimeError('reset the game before stepping a completed match')
        for action in (human_action, machine_action):
            if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < self.action_count:
                raise ValueError('action index outside the installed Pong action space')
        recent: deque[np.ndarray] = deque(maxlen=2)
        human_reward = machine_reward = 0.0
        terminated = truncated = False
        count = 0
        for _ in range(self.action_repeat):
            observations, rewards, terminations, truncations, _ = self.env.step({
                self.human_agent: int(human_action), self.machine_agent: int(machine_action),
            })
            count += 1
            human_reward += float(rewards.get(self.human_agent, 0.0))
            machine_reward += float(rewards.get(self.machine_agent, 0.0))
            observation = observations.get(self.machine_agent)
            if observation is None:
                observation = self.env.render()
            if observation is None:
                raise RuntimeError('Atari did not return a final RGB observation')
            self._rgb = np.asarray(observation, dtype=np.uint8).copy()
            recent.append(self._rgb)
            terminated = any(bool(value) for value in terminations.values())
            truncated = any(bool(value) for value in truncations.values())
            if terminated or truncated:
                break
        pooled = np.maximum(recent[0], recent[-1])
        self.frames.append(preprocess_frame(pooled))
        self._ended = terminated or truncated
        info = {
            'human_reward': human_reward, 'machine_reward': machine_reward,
            'emulator_frames': count, 'score_mode': 'return',
            'goal_events_verified': False, 'human_goals': None, 'machine_goals': None,
            'human_serve_penalties': None, 'machine_serve_penalties': None,
        }
        return np.stack(tuple(self.frames)), machine_reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        if self._rgb is None:
            raise RuntimeError('reset the game before requesting a frame')
        return self._rgb.copy()

    def close(self) -> None:
        self.env.close()
        self._ended = True
