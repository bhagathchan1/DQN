"""Paper-aligned convolutional Q-network."""

from __future__ import annotations

import torch
from torch import nn


class DQN(nn.Module):
    """DQN for stacked 84x84 grayscale frames.

    Inputs may be uint8 in [0, 255] or floating point in [0, 1].
    The output contains one unconstrained Q-value per legal action.
    """

    def __init__(self, action_count: int, input_channels: int = 4) -> None:
        super().__init__()
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        self.action_count = action_count
        self.input_channels = input_channels
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_count),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 4:
            raise ValueError("states must have shape [batch, channels, height, width]")
        if states.shape[1] != self.input_channels or states.shape[2:] != (84, 84):
            raise ValueError(
                f"expected [batch, {self.input_channels}, 84, 84], got {tuple(states.shape)}"
            )
        states = states.to(dtype=torch.float32)
        if not states.is_floating_point():  # pragma: no cover - conversion above is explicit
            raise TypeError("states must be numeric")
        if states.numel() and states.detach().max() > 1.0:
            states = states / 255.0
        return self.head(self.features(states))
