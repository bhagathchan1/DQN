"""Image preprocessing and a small Gymnasium adapter for DQN."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from PIL import Image


def preprocess_frame(
    frame: np.ndarray,
    size: tuple[int, int] = (84, 84),
    method: str = "pillow",
) -> np.ndarray:
    """Convert HxW[xC] observation to uint8 luminance at the requested size."""
    array = np.asarray(frame)
    if method == "opencv":
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional Atari dependency
            raise RuntimeError("install OpenCV for upstream preprocessing: pip install opencv-python-headless") from exc
        if array.ndim == 3 and array.shape[-1] >= 3:
            array = cv2.cvtColor(array[..., :3], cv2.COLOR_RGB2GRAY)
        elif array.ndim != 2:
            raise ValueError(f"expected HxW or HxWxC frame, got {array.shape}")
        return cv2.resize(
            np.clip(array, 0, 255).astype(np.uint8), size, interpolation=cv2.INTER_AREA
        ).astype(np.uint8)
    if method != "pillow":
        raise ValueError("method must be 'pillow' or 'opencv'")
    if array.ndim == 3 and array.shape[-1] >= 3:
        rgb = array[..., :3].astype(np.float32)
        array = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    elif array.ndim != 2:
        raise ValueError(f"expected HxW or HxWxC frame, got {array.shape}")
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array).resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


class DQNEnv:
    """Gymnasium-compatible adapter with max-pooling, action repeat and frame stacks.

    `terminated` is preserved separately from `truncated`. The project pipeline uses
    true MDP terminals; the upstream pipeline intentionally treats life loss as a
    terminal training boundary to match its episodic-life wrapper.
    """

    def __init__(
        self,
        env: Any,
        frame_stack: int = 4,
        action_repeat: int = 4,
        noop_max: int = 0,
        preprocessing: str = "pillow",
    ) -> None:
        if frame_stack <= 0 or action_repeat <= 0:
            raise ValueError("frame_stack and action_repeat must be positive")
        self.env = env
        self.frame_stack = frame_stack
        self.action_repeat = action_repeat
        self.noop_max = noop_max
        self.rng = np.random.default_rng()
        self.lives = None
        self.preprocessing = preprocessing
        self.frames: deque[np.ndarray] = deque(maxlen=frame_stack)

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def render(self) -> np.ndarray | None:
        return self.env.render()

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        if kwargs.get('seed') is not None:
            self.rng = np.random.default_rng(kwargs['seed'])
        if self.noop_max:
            for _ in range(int(self.rng.integers(1, self.noop_max + 1))):
                observation, _, terminated, truncated, info = self.env.step(0)
                if terminated or truncated:
                    observation, info = self.env.reset()
        self.lives = info.get('lives')
        processed = preprocess_frame(observation, method=self.preprocessing)
        self.frames.clear()
        self.frames.extend(processed.copy() for _ in range(self.frame_stack))
        return self._state(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        terminated = False
        truncated = False
        recent: list[np.ndarray] = []
        emulator_frames = 0
        info: dict[str, Any] = {}
        for _ in range(self.action_repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            emulator_frames += int(info.get('emulator_frames', 1))
            recent.append(np.asarray(observation))
            if terminated or truncated:
                break
        pooled = np.maximum(recent[-1], recent[-2]) if len(recent) >= 2 else recent[-1]
        self.frames.append(preprocess_frame(pooled, method=self.preprocessing))
        info = dict(info)
        lives = info.get('lives')
        info['life_lost'] = lives is not None and self.lives is not None and lives < self.lives
        info['emulator_frames'] = emulator_frames
        self.lives = lives
        return self._state(), total_reward, bool(terminated), bool(truncated), info

    def _state(self) -> np.ndarray:
        return np.stack(tuple(self.frames), axis=0)

    def close(self) -> None:
        self.env.close()


def make_env(
    env_id: str,
    seed: int | None = None,
    noop_max: int = 0,
    pipeline: str = "project",
    clip_rewards: bool = False,
    render_mode: str | None = None,
) -> DQNEnv:
    """Create an image-observation Gymnasium environment.

    Gymnasium is imported lazily so model/replay tests do not require Atari extras.
    """
    try:
        import gymnasium as gym
        import ale_py
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("install environment extras: pip install -e '.[env]'") from exc
    # Gymnasium 1.x requires explicit registration of external ALE environments.
    gym.register_envs(ale_py)
    env = gym.make(
        env_id,
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode=render_mode,
    )
    if pipeline == "upstream":
        from .upstream_data import (
            EpisodicLifeEnv,
            FireResetEnv,
            MaxAndSkipEnv,
            NoopResetEnv,
            SignClipRewardEnv,
        )

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        if clip_rewards:
            env = SignClipRewardEnv(env)
        wrapped = DQNEnv(env, action_repeat=1, preprocessing="opencv")
    elif pipeline == "project":
        wrapped = DQNEnv(env, noop_max=noop_max)
    else:
        env.close()
        raise ValueError("pipeline must be 'project' or 'upstream'")
    if seed is not None:
        wrapped.reset(seed=seed)
        wrapped.action_space.seed(seed)
    return wrapped
