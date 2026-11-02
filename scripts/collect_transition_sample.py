"""Collect a small reviewable transition sample with the linked Atari pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dqn_project.environment import make_env


def collect_sample(
    env_id: str,
    steps: int,
    seed: int,
    output: Path,
) -> tuple[Path, Path, Path]:
    if steps <= 0:
        raise ValueError("steps must be positive")
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
            action = int(env.action_space.sample())
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
                "policy": "uniform random actions",
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
    args = parser.parse_args()
    paths = collect_sample(args.env, args.steps, args.seed, args.output)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
