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
    max_steps_per_episode: int | None = None,
    render: bool = False,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps_per_episode is not None and max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be positive when provided")
    env = make_env(env_id, render_mode="human") if render else make_env(env_id)
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
                done = (
                    terminated
                    or truncated
                    or (
                        max_steps_per_episode is not None
                        and length >= max_steps_per_episode
                    )
                )
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
        "episode_lengths": lengths,
        "max_steps_per_episode": max_steps_per_episode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--random-baseline", action="store_true")
    parser.add_argument("--output", default="results/metrics/evaluation.json")
    parser.add_argument("--render", action="store_true", help="Show live gameplay")
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
    checkpoint_metadata: dict[str, Any] = {}
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
        checkpoint_metadata = dict(checkpoint.get("metadata", {}))
        saved_env = checkpoint_metadata.get("environment_id")
        if saved_env and saved_env != config["environment"]["id"]:
            raise ValueError(f"checkpoint environment {saved_env} does not match config")
    result = evaluate_policy(
        network,
        config["environment"]["id"],
        int(config["evaluation"]["episodes"]),
        float(config["evaluation"]["epsilon"]),
        int(config["evaluation"].get("seed", config["run"]["seed"])),
        device,
        int(config["evaluation"].get("max_steps_per_episode", 4500)),
        render=args.render,
    )
    result.update(
        {
            "run_id": checkpoint_metadata.get(
                "run_id", f"breakout_random_seed{config['run']['seed']}"
            ),
            "seed": int(config["run"]["seed"]),
            "evaluation_seed": int(config["evaluation"].get("seed", config["run"]["seed"])),
            "environment_id": str(config["environment"]["id"]),
            "config_id": Path(args.config).stem,
            "replay_type": (
                str(config["replay"]["type"]) if network is not None else "random"
            ),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "checkpoint_metadata": checkpoint_metadata,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
