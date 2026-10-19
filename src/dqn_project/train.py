"""Runnable DQN training loop for uniform or prioritized replay."""

from __future__ import annotations

import argparse
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .agent import EpsilonSchedule, select_action
from .config import load_config
from .environment import make_env
from .learner import DQNLearner
from .logging_utils import MetricsLogger
from .prioritized_replay import PrioritizedReplayBuffer
from .replay_buffer import ReplayBuffer


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def train(config_path: str) -> Path:
    config = load_config(config_path)
    run_cfg = config["run"]
    env_cfg = config["environment"]
    train_cfg = config["training"]
    replay_cfg = config["replay"]
    seed = int(run_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = _device(str(train_cfg.get("device", "auto")))

    env = make_env(env_cfg["id"])
    action_count = int(env.action_space.n)
    learner = DQNLearner(
        action_count,
        gamma=float(train_cfg["gamma"]),
        learning_rate=float(train_cfg["learning_rate"]),
        device=device,
    )
    replay_type = replay_cfg["type"]
    common = {
        "capacity": int(replay_cfg["capacity"]),
        "state_shape": (4, 84, 84),
        "seed": seed,
    }
    if replay_type == "prioritized":
        replay: ReplayBuffer = PrioritizedReplayBuffer(
            **common,
            alpha=float(replay_cfg["alpha"]),
            priority_epsilon=float(replay_cfg["priority_epsilon"]),
        )
    elif replay_type == "uniform":
        replay = ReplayBuffer(**common)
    else:
        raise ValueError("replay.type must be 'uniform' or 'prioritized'")

    schedule = EpsilonSchedule(
        start=float(train_cfg["epsilon_start"]),
        end=float(train_cfg["epsilon_end"]),
        decay_steps=int(train_cfg["epsilon_decay_steps"]),
    )
    run_id = str(run_cfg["id"])
    metrics_path = Path("results/metrics") / f"{run_id}.csv"
    logger = MetricsLogger(metrics_path)
    state, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_length = 0
    episode = 0
    latest_loss = float("nan")
    start = time.perf_counter()
    total_steps = int(train_cfg["total_environment_steps"])
    batch_size = int(train_cfg["batch_size"])
    learning_starts = int(train_cfg["learning_starts"])
    train_frequency = int(train_cfg["train_frequency"])
    target_update_frequency = int(train_cfg["target_update_frequency"])
    checkpoint_frequency = int(train_cfg["checkpoint_frequency"])
    beta_start = float(replay_cfg.get("beta_start", 0.4))

    try:
        for step in range(1, total_steps + 1):
            epsilon = schedule.value(step - 1)
            if step <= learning_starts:
                action = int(rng.integers(action_count))
            else:
                action = select_action(learner.online, state, action_count, epsilon, rng, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            clipped_reward = float(np.sign(reward)) if train_cfg.get("clip_rewards", True) else reward
            replay.add(state, action, clipped_reward, next_state, terminated)
            state = next_state
            episode_reward += reward
            episode_length += 1

            if step >= learning_starts and step % train_frequency == 0 and len(replay) >= batch_size:
                progress = step / total_steps
                beta = beta_start + progress * (1.0 - beta_start)
                batch = replay.sample(batch_size, beta=beta)
                update = learner.update(batch)
                latest_loss = update.loss
                if batch.indices is not None:
                    replay.update_priorities(batch.indices, update.td_errors)
                if learner.update_count % target_update_frequency == 0:
                    learner.sync_target()

            if terminated or truncated:
                logger.log(
                    run_id=run_id,
                    seed=seed,
                    environment_steps=step,
                    frames=step * 4,
                    episode=episode,
                    episode_reward=episode_reward,
                    episode_length=episode_length,
                    loss=latest_loss,
                    epsilon=epsilon,
                    wall_time_seconds=time.perf_counter() - start,
                    config_id=Path(config_path).stem,
                    replay_type=replay_type,
                    event="episode",
                )
                episode += 1
                episode_reward = 0.0
                episode_length = 0
                state, _ = env.reset()

            if checkpoint_frequency > 0 and step % checkpoint_frequency == 0:
                learner.save(
                    Path("checkpoints") / f"{run_id}_step_{step}.pt",
                    metadata={
                        "run_id": run_id,
                        "environment_id": env_cfg["id"],
                        "environment_steps": step,
                        "config_path": config_path,
                        "git_revision": _git_revision(),
                        "seed": seed,
                    },
                )
    finally:
        env.close()

    final_path = Path("checkpoints") / f"{run_id}_final.pt"
    learner.save(
        final_path,
        metadata={
            "run_id": run_id,
            "environment_id": env_cfg["id"],
            "environment_steps": total_steps,
            "config_path": config_path,
            "git_revision": _git_revision(),
            "seed": seed,
        },
    )
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    print(train(args.config))


if __name__ == "__main__":
    main()
