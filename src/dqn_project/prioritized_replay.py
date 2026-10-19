"""Proportional prioritized replay (project extension, not from Mnih et al. 2015)."""

from __future__ import annotations

import numpy as np

from .replay_buffer import ReplayBatch, ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritization: P(i) = p_i**alpha / sum_j p_j**alpha."""

    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...] = (4, 84, 84),
        alpha: float = 0.6,
        priority_epsilon: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        super().__init__(capacity, state_shape, seed)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if priority_epsilon <= 0:
            raise ValueError("priority_epsilon must be positive")
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, *args: object, **kwargs: object) -> int:
        index = super().add(*args, **kwargs)
        self.priorities[index] = self.max_priority
        return index

    def sampling_probabilities(self) -> np.ndarray:
        if self.size == 0:
            return np.empty(0, dtype=np.float64)
        scaled = np.power(self.priorities[: self.size].astype(np.float64), self.alpha)
        total = scaled.sum()
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("invalid replay priorities")
        return scaled / total

    def sample(self, batch_size: int, beta: float | None = 0.4) -> ReplayBatch:
        if batch_size <= 0 or self.size < batch_size:
            raise ValueError(f"need {batch_size} transitions, buffer contains {self.size}")
        if beta is None:
            beta = 0.4
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        probabilities = self.sampling_probabilities()
        indices = self.rng.choice(self.size, size=batch_size, replace=False, p=probabilities)
        weights = np.power(self.size * probabilities[indices], -beta)
        weights /= weights.max()
        return ReplayBatch(
            states=self.states[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_states=self.next_states[indices].copy(),
            terminated=self.terminated[indices].copy(),
            weights=weights.astype(np.float32),
            indices=indices.astype(np.int64),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float32)
        if indices.shape != priorities.shape:
            raise ValueError("indices and priorities must have identical shapes")
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("priority index outside populated buffer")
        if np.any(~np.isfinite(priorities)) or np.any(priorities < 0):
            raise ValueError("priorities must be finite and non-negative")
        adjusted = priorities + self.priority_epsilon
        self.priorities[indices] = adjusted
        self.max_priority = max(self.max_priority, float(adjusted.max(initial=0.0)))
