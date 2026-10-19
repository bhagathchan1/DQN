"""Exploration schedule and epsilon-greedy action selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EpsilonSchedule:
    start: float = 1.0
    end: float = 0.1
    decay_steps: int = 1_000_000

    def __post_init__(self) -> None:
        if not 0 <= self.end <= self.start <= 1:
            raise ValueError("require 0 <= end <= start <= 1")
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be positive")

    def value(self, step: int) -> float:
        if step >= self.decay_steps:
            return self.end
        fraction = max(step, 0) / self.decay_steps
        return self.start + fraction * (self.end - self.start)


@torch.no_grad()
def select_action(
    network: torch.nn.Module,
    state: np.ndarray,
    action_count: int,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device | str = "cpu",
) -> int:
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if rng.random() < epsilon:
        return int(rng.integers(action_count))
    state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
    return int(network(state_tensor).argmax(dim=1).item())
