"""Watch a saved DQN checkpoint play an Atari game in a local display window."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .agent import select_action
from .environment import make_env
from .learner import DQNLearner


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def play(
    checkpoint: str | Path,
    env_id: str = "ALE/Breakout-v5",
    episodes: int = 1,
    epsilon: float = 0.0,
    seed: int = 7,
    device: str = "auto",
    pipeline: str = "upstream",
    max_actions: int | None = None,
) -> None:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if max_actions is not None and max_actions <= 0:
        raise ValueError("max_actions must be positive when provided")

    checkpoint = Path(checkpoint)
    target_device = _device(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    learner = DQNLearner(
        action_count=int(payload["action_count"]),
        input_channels=int(payload.get("input_channels", 4)),
        gamma=float(payload.get("gamma", 0.99)),
        device=target_device,
        optimizer=str(payload.get("optimizer_name", "rmsprop")),
        optimizer_epsilon=float(payload.get("optimizer_epsilon", 0.01)),
        gradient_clip_mode=str(payload.get("gradient_clip_mode", "global")),
        initializer=str(payload.get("initializer", "pytorch")),
        initialize_target_from_online=bool(
            payload.get("initialize_target_from_online", True)
        ),
    )
    learner.load(checkpoint, load_optimizer=False)
    learner.online.eval()

    env = make_env(
        env_id,
        pipeline=pipeline,
        clip_rewards=False,
        render_mode="human",
    )
    rng = np.random.default_rng(seed)
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            reward_total = 0.0
            action_count = 0
            done = False
            while not done:
                action = select_action(
                    learner.online,
                    state,
                    env.action_space.n,
                    epsilon,
                    rng,
                    target_device,
                )
                state, reward, terminated, truncated, info = env.step(action)
                reward_total += reward
                action_count += 1
                life_boundary = bool(
                    pipeline == "upstream"
                    and terminated
                    and info.get("episodic_life")
                    and int(info.get("ale.lives", 0)) > 0
                )
                if life_boundary:
                    state, _ = env.reset()
                    continue
                done = terminated or truncated or (
                    max_actions is not None and action_count >= max_actions
                )
            print(
                f"Episode {episode + 1}: score={reward_total:.1f}, "
                f"actions={action_count}"
            )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env", default="ALE/Breakout-v5")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument(
        "--pipeline", choices=("project", "upstream"), default="upstream"
    )
    parser.add_argument("--max-actions", type=int)
    args = parser.parse_args()
    play(
        checkpoint=args.checkpoint,
        env_id=args.env,
        episodes=args.episodes,
        epsilon=args.epsilon,
        seed=args.seed,
        device=args.device,
        pipeline=args.pipeline,
        max_actions=args.max_actions,
    )


if __name__ == "__main__":
    main()
