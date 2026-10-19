"""Checkpoint evaluation using the same preprocessing as training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import select_action
from .config import load_config
from .environment import make_env
from .learner import DQNLearner


def evaluate_policy(
    network: torch.nn.Module | None,
    env_id: str,
    episodes: int,
    epsilon: float,
    seed: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    env = make_env(env_id)
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    lengths: list[int] = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            length = 0
            while not done:
                if network is None:
                    action = int(rng.integers(env.action_space.n))
                else:
                    action = select_action(
                        network, state, env.action_space.n, epsilon, rng, device
                    )
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                length += 1
                done = terminated or truncated
            rewards.append(total_reward)
            lengths.append(length)
    finally:
        env.close()
    return {
        "policy": "random" if network is None else "dqn",
        "episodes": episodes,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "episode_rewards": rewards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--random-baseline", action="store_true")
    parser.add_argument("--output", default="results/metrics/evaluation.json")
    args = parser.parse_args()
    config = load_config(args.config)
    requested_device = str(config["training"].get("device", "cpu"))
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested_device
    network = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        learner = DQNLearner(
            action_count=int(checkpoint["action_count"]),
            input_channels=int(checkpoint.get("input_channels", 4)),
            gamma=float(checkpoint.get("gamma", config["training"]["gamma"])),
            device=device,
        )
        learner.load(args.checkpoint, load_optimizer=False)
        network = learner.online
    result = evaluate_policy(
        network,
        config["environment"]["id"],
        int(config["evaluation"]["episodes"]),
        float(config["evaluation"]["epsilon"]),
        int(config["run"]["seed"]),
        device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
