"""Generate learning-curve plots and summary tables from saved CSV logs."""

from __future__ import annotations

import argparse
from pathlib import Path


def plot_metrics(inputs: list[str], output: str) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install analysis extras: pip install -e '.[analysis]'") from exc
    frames = []
    for path in inputs:
        frame = pd.read_csv(path)
        required = {"run_id", "environment_steps", "episode_reward", "replay_type"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frames.append(frame[frame["event"] == "episode"].copy())
    data = pd.concat(frames, ignore_index=True)
    data["episode_reward"] = pd.to_numeric(data["episode_reward"], errors="coerce")
    data = data.dropna(subset=["episode_reward"])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for run_id, group in data.groupby("run_id", sort=True):
        group = group.sort_values("environment_steps")
        smoothed = group["episode_reward"].rolling(20, min_periods=1).mean()
        replay = group["replay_type"].iloc[0]
        axis.plot(group["environment_steps"], smoothed, label=f"{run_id} ({replay})")
    axis.set_xlabel("Environment steps (action repeats)")
    axis.set_ylabel("Episode reward (20-episode rolling mean)")
    axis.set_title("DQN training reward")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    summary = data.groupby(["run_id", "replay_type"])["episode_reward"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    summary.to_csv(path.with_suffix(".summary.csv"))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="results/plots/training_reward.png")
    args = parser.parse_args()
    print(plot_metrics(args.inputs, args.output))


if __name__ == "__main__":
    main()
