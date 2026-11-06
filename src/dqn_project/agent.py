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
    warmup_steps: int = 0
    terminal_end: float | None = None
    terminal_decay_steps: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.end <= self.start <= 1:
            raise ValueError("require 0 <= end <= start <= 1")
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be positive")
        if not 0 <= self.warmup_steps < self.decay_steps:
            raise ValueError("warmup_steps must be non-negative and less than decay_steps")
        if self.terminal_end is not None:
            if not 0 <= self.terminal_end <= self.end:
                raise ValueError("terminal_end must be between zero and end")
            if self.terminal_decay_steps is None or self.terminal_decay_steps <= self.decay_steps:
                raise ValueError("terminal_decay_steps must be greater than decay_steps")

    def value(self, step: int) -> float:
        step = max(step, 0)
        if step < self.warmup_steps:
            return self.start
        if step < self.decay_steps:
            fraction = (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
            return self.start + fraction * (self.end - self.start)
        if self.terminal_end is not None:
            if step >= int(self.terminal_decay_steps):
                return self.terminal_end
            fraction = (step - self.decay_steps) / (
                int(self.terminal_decay_steps) - self.decay_steps
            )
            return self.end + fraction * (self.terminal_end - self.end)
        return self.end


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
