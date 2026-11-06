"""Paper-aligned convolutional Q-network."""

from __future__ import annotations

import torch
from torch import nn


class DQN(nn.Module):
    """DQN for stacked 84x84 grayscale frames.

    Inputs may be uint8 in [0, 255] or floating point in [0, 1].
    The output contains one unconstrained Q-value per legal action.
    """

    def __init__(
        self,
        action_count: int,
        input_channels: int = 4,
        initializer: str = "pytorch",
    ) -> None:
        super().__init__()
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        self.action_count = action_count
        self.input_channels = input_channels
        if initializer not in {"pytorch", "keras_variance_scaling"}:
            raise ValueError("unsupported initializer")
        self.initializer = initializer
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
        self.apply(self._initialize_keras if initializer == "keras_variance_scaling" else self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _initialize_keras(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            # TensorFlow VarianceScaling(scale=2, mode='fan_in') uses a
            # truncated normal whose variance is corrected for truncation.
            std = (2.0 / fan_in) ** 0.5 / 0.87962566103423978
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 4:
            raise ValueError("states must have shape [batch, channels, height, width]")
        if states.shape[1] != self.input_channels or states.shape[2:] != (84, 84):
            raise ValueError(
                f"expected [batch, {self.input_channels}, 84, 84], got {tuple(states.shape)}"
            )
        # Use the input contract, not the brightest pixel: dark uint8 frames must
        # still be normalized, without a GPU-to-host synchronization per forward.
        byte_input = states.dtype == torch.uint8
        states = states.to(dtype=torch.float32)
        if byte_input:
            states = states / 255.0
        return self.head(self.features(states))
