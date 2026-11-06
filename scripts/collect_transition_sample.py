"""Collect a small reviewable transition sample with the linked Atari pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dqn_project.agent import select_action
from dqn_project.environment import make_env
from dqn_project.learner import DQNLearner


def collect_sample(
    env_id: str,
    steps: int,
    seed: int,
    output: Path,
    checkpoint: Path | None = None,
    epsilon: float = 0.0,
    device: str = "auto",
) -> tuple[Path, Path, Path]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    network = None
    checkpoint_metadata: dict[str, object] = {}
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        learner = DQNLearner(
            action_count=int(payload["action_count"]),
            input_channels=int(payload.get("input_channels", 4)),
            gamma=float(payload.get("gamma", 0.99)),
            device=device,
            optimizer=str(payload.get("optimizer_name", "rmsprop")),
            optimizer_epsilon=float(payload.get("optimizer_epsilon", 0.01)),
            gradient_clip_mode=str(payload.get("gradient_clip_mode", "global")),
            initializer=str(payload.get("initializer", "pytorch")),
            initialize_target_from_online=bool(
                payload.get("initialize_target_from_online", True)
            ),
        )
        checkpoint_metadata = learner.load(checkpoint, load_optimizer=False)
        learner.online.eval()
        network = learner.online

    env = make_env(env_id, pipeline="upstream", clip_rewards=True)
    env.action_space.seed(seed)
    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    raw_rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    episode_ids: list[int] = []
    episode = 0
    try:
        state, _ = env.reset(seed=seed)
        for _ in range(steps):
            if network is None:
                action = int(env.action_space.sample())
            else:
                action = select_action(
                    network, state, env.action_space.n, epsilon, rng, device
                )
            successor, reward, terminated, truncated, info = env.step(action)
            states.append(state.copy())
            next_states.append(successor.copy())
            actions.append(action)
            rewards.append(float(reward))
            raw_rewards.append(float(info.get("raw_reward", reward)))
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            episode_ids.append(episode)
            state = successor
            if terminated or truncated:
                episode += 1
                state, _ = env.reset()
    finally:
        env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        states=np.stack(states).astype(np.uint8),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        raw_rewards=np.asarray(raw_rewards, dtype=np.float32),
        next_states=np.stack(next_states).astype(np.uint8),
        terminated=np.asarray(terminated_flags, dtype=np.bool_),
        truncated=np.asarray(truncated_flags, dtype=np.bool_),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
    )
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "environment_id": env_id,
                "pipeline": "upstream Atari preprocessing port",
                "policy": (
                    "uniform random actions"
                    if checkpoint is None
                    else "checkpoint epsilon-greedy"
                ),
                "epsilon": epsilon if checkpoint is not None else 1.0,
                "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
                "checkpoint_metadata": checkpoint_metadata,
                "device": device,
                "seed": seed,
                "transitions": steps,
                "state_shape": [4, 84, 84],
                "state_dtype": "uint8",
                "reward_fields": {
                    "rewards": "sign-clipped training reward",
                    "raw_rewards": "unmodified reward returned by the game",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    sample_image = output.with_name(output.stem + "_first_state.png")
    # Current project tensors are channel-first; tile the four grayscale frames.
    first = np.concatenate(states[0], axis=1)
    Image.fromarray(first, mode="L").save(sample_image)
    return output, metadata_path, sample_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="ALE/Breakout-v5")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("data/breakout_sample.npz"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional DQN checkpoint; omit to collect uniform-random actions",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration rate when collecting from a checkpoint (default: greedy)",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    args = parser.parse_args()
    paths = collect_sample(
        args.env,
        args.steps,
        args.seed,
        args.output,
        checkpoint=args.checkpoint,
        epsilon=args.epsilon,
        device=args.device,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
