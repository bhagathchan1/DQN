"""DQN optimization, target synchronization, and checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .model import DQN
from .optimizer import NatureRMSprop
from .replay_buffer import ReplayBatch


@dataclass(frozen=True)
class UpdateMetrics:
    loss: float
    mean_q: float
    mean_target: float
    td_errors: np.ndarray


class DQNLearner:
    def __init__(
        self,
        action_count: int,
        input_channels: int = 4,
        gamma: float = 0.99,
        learning_rate: float = 2.5e-4,
        device: str | torch.device = "cpu",
        optimizer: str = "rmsprop",
        gradient_clip: float | None = 10.0,
    ) -> None:
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        self.device = torch.device(device)
        self.gamma = gamma
        self.online = DQN(action_count, input_channels).to(self.device)
        self.target = DQN(action_count, input_channels).to(self.device)
        self.optimizer_name = optimizer
        self.gradient_clip = gradient_clip
        if optimizer == 'nature_rmsprop':
            self.optimizer = NatureRMSprop(self.online.parameters(), lr=learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(
                self.online.parameters(), lr=learning_rate, alpha=0.95, eps=0.01)
        else:
            raise ValueError('unknown optimizer')
        self.update_count = 0
        self.sync_target()
        self.target.eval()

    @torch.no_grad()
    def compute_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        next_q = self.target(next_states).max(dim=1).values
        return rewards + self.gamma * (~terminated).float() * next_q

    def update(self, batch: ReplayBatch) -> UpdateMetrics:
        states = torch.as_tensor(batch.states, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(batch.rewards, device=self.device, dtype=torch.float32)
        next_states = torch.as_tensor(batch.next_states, device=self.device)
        terminated = torch.as_tensor(batch.terminated, device=self.device, dtype=torch.bool)
        if batch.weights is None:
            weights = torch.ones_like(rewards)
        else:
            weights = torch.as_tensor(batch.weights, device=self.device, dtype=torch.float32)

        predicted = self.online(states).gather(1, actions[:, None]).squeeze(1)
        targets = self.compute_targets(rewards, next_states, terminated)
        td_errors = targets - predicted
        per_item_loss = nn.functional.smooth_l1_loss(predicted, targets, reduction="none")
        weighted = weights * per_item_loss
        loss = weighted.sum() if self.optimizer_name == 'nature_rmsprop' else weighted.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite DQN loss")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=self.gradient_clip)
        self.optimizer.step()
        self.update_count += 1
        return UpdateMetrics(
            loss=float(loss.detach().item()),
            mean_q=float(predicted.detach().mean().item()),
            mean_target=float(targets.detach().mean().item()),
            td_errors=td_errors.detach().abs().cpu().numpy(),
        )

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimizer_name": self.optimizer_name,
                "update_count": self.update_count,
                "gamma": self.gamma,
                "action_count": self.online.action_count,
                "input_channels": self.online.input_channels,
                "metadata": metadata or {},
            },
            path,
        )

    def load(self, path: str | Path, load_optimizer: bool = True) -> dict[str, Any]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["target"])
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.update_count = int(payload.get("update_count", 0))
        return dict(payload.get("metadata", {}))
