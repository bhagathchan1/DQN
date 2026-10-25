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
from .frame_replay import FrameReplayBuffer


def _git_revision() -> str:
    repository = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repository.as_posix()}",
             "rev-parse", "--short", "HEAD"], cwd=repository,
            text=True, stderr=subprocess.DEVNULL
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

    env = make_env(env_cfg["id"], noop_max=int(env_cfg.get('noop_max', 0)))
    action_count = int(env.action_space.n)
    learner = DQNLearner(
        action_count,
        gamma=float(train_cfg["gamma"]),
        learning_rate=float(train_cfg["learning_rate"]),
        device=device,
        optimizer=train_cfg.get('optimizer', 'rmsprop'),
        gradient_clip=train_cfg.get('gradient_clip', 10.0),
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
        if replay_cfg.get('storage') == 'frames':
            replay = FrameReplayBuffer(**common, directory=replay_cfg.get('directory'))
        else:
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
    emulator_frames = 0
    saved_checkpoints = []
    best_validation = -float('inf')

    try:
        for step in range(1, total_steps + 1):
            offset = learning_starts if train_cfg.get('epsilon_after_warmup', False) else 0
            epsilon = schedule.value(max(0, step - 1 - offset))
            if step <= learning_starts:
                action = int(rng.integers(action_count))
            else:
                action = select_action(learner.online, state, action_count, epsilon, rng, device)
            next_state, reward, terminated, truncated, info = env.step(action)
            emulator_frames += info.get('emulator_frames', 4)
            clipped_reward = float(np.sign(reward)) if train_cfg.get("clip_rewards", True) else reward
            learning_terminal = terminated or (train_cfg.get('terminal_on_life_loss', False) and info.get('life_lost', False))
            replay.add(state, action, clipped_reward, next_state, learning_terminal)
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
                    frames=emulator_frames,
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
                if isinstance(replay, FrameReplayBuffer):
                    replay.start_episode()

            log_frequency = int(train_cfg.get('log_frequency', 10000))
            if step % log_frequency == 0:
                print(f'step={step}/{total_steps} frames={emulator_frames} '
                      f'updates={learner.update_count} loss={latest_loss:.5f} '
                      f'epsilon={epsilon:.3f}', flush=True)

            if checkpoint_frequency > 0 and step % checkpoint_frequency == 0:
                checkpoint_path = Path("checkpoints") / f"{run_id}_step_{step}.pt"
                learner.save(
                    checkpoint_path,
                    metadata={
                        "run_id": run_id,
                        "environment_id": env_cfg["id"],
                        "environment_steps": step,
                        "config_path": config_path,
                        "git_revision": _git_revision(),
                        "seed": seed,
                        "config": config,
                    },
                )
                saved_checkpoints.append(checkpoint_path)
                keep = int(train_cfg.get('checkpoint_keep', 0))
                if keep and len(saved_checkpoints) > keep:
                    saved_checkpoints.pop(0).unlink()

            validation_frequency = int(train_cfg.get('validation_frequency', 0))
            if validation_frequency and step % validation_frequency == 0:
                from .evaluate import evaluate_policy
                evaluation = config['evaluation']
                score = evaluate_policy(
                    learner.online, env_cfg['id'], int(evaluation['episodes']),
                    float(evaluation['epsilon']), (seed + 1000000) % 2**32, device,
                    int(evaluation.get('max_steps_per_episode', 4500)),
                    noop_max=int(env_cfg.get('noop_max', 0)))
                logger.log(run_id=run_id, seed=seed, environment_steps=step,
                           frames=emulator_frames, event='validation', replay_type=replay_type,
                           evaluation_reward_mean=score['reward_mean'],
                           evaluation_reward_std=score['reward_std'])
                print(f"validation step={step} mean={score['reward_mean']:.3f}", flush=True)
                if score['reward_mean'] > best_validation:
                    best_validation = score['reward_mean']
                    learner.save(Path('checkpoints') / f'{run_id}_best.pt', metadata={
                        'run_id': run_id, 'environment_id': env_cfg['id'],
                        'environment_steps': step, 'seed': seed, 'config': config,
                        'selection': 'highest mean validation return',
                        'validation_reward_mean': best_validation,
                        'git_revision': _git_revision()})
    finally:
        env.close()
        if isinstance(replay, FrameReplayBuffer):
            replay.close()

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
            "config": config,
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
