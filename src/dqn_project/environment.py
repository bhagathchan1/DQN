"""Image preprocessing and a small Gymnasium adapter for DQN."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from PIL import Image


def preprocess_frame(frame: np.ndarray, size: tuple[int, int] = (84, 84)) -> np.ndarray:
    """Convert HxW[xC] observation to uint8 luminance at the requested size."""
    array = np.asarray(frame)
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

    `terminated` is preserved separately from `truncated`. The training loop resets on
    either signal but only true termination disables Bellman bootstrapping.
    """

    def __init__(self, env: Any, frame_stack: int = 4, action_repeat: int = 4) -> None:
        if frame_stack <= 0 or action_repeat <= 0:
            raise ValueError("frame_stack and action_repeat must be positive")
        self.env = env
        self.frame_stack = frame_stack
        self.action_repeat = action_repeat
        self.frames: deque[np.ndarray] = deque(maxlen=frame_stack)

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        processed = preprocess_frame(observation)
        self.frames.clear()
        self.frames.extend(processed.copy() for _ in range(self.frame_stack))
        return self._state(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        terminated = False
        truncated = False
        recent: list[np.ndarray] = []
        info: dict[str, Any] = {}
        for _ in range(self.action_repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            recent.append(np.asarray(observation))
            if terminated or truncated:
                break
        pooled = np.maximum(recent[-1], recent[-2]) if len(recent) >= 2 else recent[-1]
        self.frames.append(preprocess_frame(pooled))
        return self._state(), total_reward, bool(terminated), bool(truncated), info

    def _state(self) -> np.ndarray:
        return np.stack(tuple(self.frames), axis=0)

    def close(self) -> None:
        self.env.close()


def make_env(env_id: str, seed: int | None = None) -> DQNEnv:
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
    env = gym.make(env_id, frameskip=1, repeat_action_probability=0.0)
    wrapped = DQNEnv(env)
    if seed is not None:
        wrapped.reset(seed=seed)
        wrapped.action_space.seed(seed)
    return wrapped
