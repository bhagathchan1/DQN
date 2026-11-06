"""Gymnasium wrappers matching the linked project's online Atari data pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np


class NoopResetEnv:
    """Randomize episode starts with 1..noop_max action-zero steps."""

    def __init__(self, env: Any, noop_max: int = 30) -> None:
        self.env = env
        self.noop_max = noop_max
        self.action_space = env.action_space
        if noop_max <= 0:
            raise ValueError("noop_max must be positive")
        meanings = getattr(env.unwrapped, "get_action_meanings", lambda: ["NOOP"])()
        if meanings and meanings[0] != "NOOP":
            raise ValueError("Atari action 0 must be NOOP")

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        rng = self.env.unwrapped.np_random
        noops = int(rng.integers(1, self.noop_max + 1))
        for _ in range(noops):
            observation, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                observation, info = self.env.reset()
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()

    def render(self) -> np.ndarray | None:
        return self.env.render()


class MaxAndSkipEnv:
    """Repeat each chosen action and max-pool the final two observations."""

    def __init__(self, env: Any, skip: int = 4) -> None:
        self.env = env
        self.skip = skip
        self.action_space = env.action_space

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        return self.env.reset(**kwargs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observations: list[np.ndarray] = []
        total_reward = 0.0
        terminated = truncated = False
        info: dict[str, Any] = {}
        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            observations.append(np.asarray(observation))
            total_reward += float(reward)
            if terminated or truncated:
                break
        pooled = (
            np.maximum(observations[-2], observations[-1])
            if len(observations) >= 2
            else observations[-1]
        )
        info = dict(info)
        info['emulator_frames'] = len(observations)
        return pooled, total_reward, bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()

    def render(self) -> np.ndarray | None:
        return self.env.render()


class EpisodicLifeEnv:
    """Treat losing an Atari life as a terminal transition for training."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.action_space = env.action_space
        self.lives = 0
        self.was_real_done = True

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def _lives(self, info: dict[str, Any] | None = None) -> int:
        info = info or {}
        if "ale.lives" in info:
            return int(info["ale.lives"])
        if "lives" in info:
            return int(info["lives"])
        ale = getattr(self.env.unwrapped, "ale", None)
        return int(ale.lives()) if ale is not None else 0

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if self.was_real_done:
            observation, info = self.env.reset(**kwargs)
        else:
            observation, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)
                self.was_real_done = True
        self.lives = self._lives(info)
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = bool(terminated or truncated)
        lives = self._lives(info)
        life_lost = 0 < lives < self.lives
        self.lives = lives
        if life_lost and not self.was_real_done:
            terminated = True
            info = dict(info)
            info["episodic_life"] = True
            info["ale.lives"] = lives
        return observation, reward, bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()

    def render(self) -> np.ndarray | None:
        return self.env.render()


class FireResetEnv:
    """Issue FIRE and the follow-up action after reset when the game requires it."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.action_space = env.action_space
        meanings = getattr(env.unwrapped, "get_action_meanings", lambda: [])()
        self.enabled = len(meanings) >= 3 and meanings[1] == "FIRE"

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        if not self.enabled:
            return observation, info
        for action in (1, 2):
            observation, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                observation, info = self.env.reset()
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()

    def render(self) -> np.ndarray | None:
        return self.env.render()


class SignClipRewardEnv:
    """Clip training rewards to their sign and preserve the raw value in info."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.action_space = env.action_space

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        return self.env.reset(**kwargs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["raw_reward"] = float(reward)
        return observation, float(np.sign(reward)), terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def render(self) -> np.ndarray | None:
        return self.env.render()
