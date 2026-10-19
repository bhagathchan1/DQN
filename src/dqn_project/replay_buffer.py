"""Uniform experience replay with an explicit, shared batch contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ReplayBatch:
    """A sampled transition batch.

    states/next_states: uint8 [B, C, 84, 84]
    actions: int64 [B]
    rewards: float32 [B]
    terminated: bool [B], true only for true MDP terminal transitions
    weights: optional float32 [B] importance weights
    indices: optional int64 [B] replay slots for priority updates
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminated: np.ndarray
    weights: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None


class ReplayBuffer:
    """Fixed-size circular buffer storing complete frame stacks."""

    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...] = (4, 84, 84),
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.state_shape = state_shape
        self.rng = np.random.default_rng(seed)
        self.states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> int:
        state_array = self._validate_state(state, "state")
        next_state_array = self._validate_state(next_state, "next_state")
        index = self.position
        self.states[index] = state_array
        self.next_states[index] = next_state_array
        self.actions[index] = action
        self.rewards[index] = reward
        self.terminated[index] = terminated
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return index

    def _validate_state(self, value: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != self.state_shape:
            raise ValueError(f"{name} must have shape {self.state_shape}, got {array.shape}")
        if array.dtype == np.uint8:
            return array
        if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1.0:
            array = np.rint(array * 255.0)
        if array.size and (array.min() < 0 or array.max() > 255):
            raise ValueError(f"{name} values must be in [0, 255]")
        return array.astype(np.uint8)

    def sample(self, batch_size: int, beta: float | None = None) -> ReplayBatch:
        del beta  # shared signature with prioritized replay
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size < batch_size:
            raise ValueError(f"need {batch_size} transitions, buffer contains {self.size}")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return ReplayBatch(
            states=self.states[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_states=self.next_states[indices].copy(),
            terminated=self.terminated[indices].copy(),
            weights=np.ones(batch_size, dtype=np.float32),
            indices=indices.astype(np.int64),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """No-op compatibility hook for uniform replay."""
        del indices, priorities
