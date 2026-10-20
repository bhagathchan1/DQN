"""Aggregate matched DQN experiment logs and render reproducible result figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "random": "#6B7280",
    "uniform": "#2563EB",
    "prioritized": "#D97706",
}
LABELS = {
    "random": "Random",
    "uniform": "Uniform DQN",
    "prioritized": "Prioritized DQN",
}


def load_training_logs(metrics_dir: Path) -> pd.DataFrame:
    paths = sorted(metrics_dir.glob("breakout_*_seed*.csv"))
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame.loc[frame["event"] == "episode"].copy()
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no matched training logs found in {metrics_dir}")
    data = pd.concat(frames, ignore_index=True)
    data["environment_steps"] = pd.to_numeric(data["environment_steps"])
    data["episode_reward"] = pd.to_numeric(data["episode_reward"])
    return data


def load_evaluations(metrics_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(metrics_dir.glob("*_evaluation.json")):
        if path.name.startswith("smoke_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {"run_id", "seed", "replay_type", "episode_rewards"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"{path} missing fields: {sorted(missing)}")
        for episode, reward in enumerate(payload["episode_rewards"]):
            rows.append(
                {
                    "run_id": payload["run_id"],
                    "seed": int(payload["seed"]),
                    "policy": payload["replay_type"],
                    "episode": episode,
                    "reward": float(reward),
                    "source_file": path.name,
                }
            )
    if not rows:
        raise FileNotFoundError(f"no completed evaluation JSON files found in {metrics_dir}")
    return pd.DataFrame(rows)


def summarize_evaluations(
    evaluations: pd.DataFrame, metrics_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = (
        evaluations.groupby(["policy", "seed"], as_index=False)["reward"]
        .agg(episodes="count", mean="mean", std="std", minimum="min", maximum="max")
        .sort_values(["policy", "seed"])
    )
    aggregate = (
        per_seed.groupby("policy", as_index=False)["mean"]
        .agg(seeds="count", mean_of_seed_means="mean", std_across_seed_means="std")
    )
    wide = per_seed.pivot(index="seed", columns="policy", values="mean")
    if {"uniform", "prioritized", "random"}.issubset(wide.columns):
        paired = pd.DataFrame(
            {
                "seed": wide.index,
                "uniform_minus_random": wide["uniform"] - wide["random"],
                "prioritized_minus_random": wide["prioritized"] - wide["random"],
                "prioritized_minus_uniform": wide["prioritized"] - wide["uniform"],
            }
        ).reset_index(drop=True)
        if metrics_dir is not None:
            paired.to_csv(metrics_dir / "paired_seed_differences.csv", index=False)
    pooled = (
        evaluations.groupby("policy", as_index=False)["reward"]
        .agg(pooled_episodes="count", pooled_mean="mean", pooled_std="std")
    )
    return per_seed, aggregate.merge(pooled, on="policy", how="left")


def plot_training(training: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    for axis, policy in zip(axes, ["uniform", "prioritized"], strict=True):
        subset = training.loc[training["replay_type"] == policy]
        for seed, group in subset.groupby("seed", sort=True):
            ordered = group.sort_values("environment_steps")
            rolling = ordered["episode_reward"].rolling(50, min_periods=10).mean()
            axis.plot(
                ordered["environment_steps"],
                rolling,
                linewidth=1.5,
                alpha=0.82,
                label=f"Seed {int(seed)}",
            )
        axis.set_title(LABELS[policy])
        axis.set_xlabel("Environment decisions")
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Reward, rolling mean over 50 episodes")
    figure.suptitle("Breakout training reward under matched 250,000-decision budgets")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_evaluation(per_seed: pd.DataFrame, output: Path) -> None:
    policies = [p for p in ["random", "uniform", "prioritized"] if p in set(per_seed.policy)]
    positions = np.arange(len(policies))
    means = [per_seed.loc[per_seed.policy == p, "mean"].mean() for p in policies]
    errors = [per_seed.loc[per_seed.policy == p, "mean"].std(ddof=1) for p in policies]
    errors = [0.0 if np.isnan(value) else value for value in errors]

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.bar(
        positions,
        means,
        yerr=errors,
        capsize=5,
        color=[COLORS[p] for p in policies],
        edgecolor="#1F2937",
        linewidth=0.8,
        alpha=0.85,
    )
    for position, policy in zip(positions, policies, strict=True):
        values = per_seed.loc[per_seed.policy == policy].sort_values("seed")
        offsets = np.linspace(-0.09, 0.09, len(values))
        axis.scatter(
            position + offsets,
            values["mean"],
            s=44,
            color="white",
            edgecolor="#111827",
            linewidth=1.0,
            zorder=3,
        )
        for x, (_, row) in zip(position + offsets, values.iterrows(), strict=True):
            axis.annotate(
                str(int(row["seed"])),
                (x, row["mean"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axis.set_xticks(positions, [LABELS[p] for p in policies])
    axis.set_ylabel("Mean evaluation reward per episode")
    axis.set_title("Held-out Breakout evaluation across fixed seeds")
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    axis.text(
        0.01,
        -0.19,
        "Bars show the mean of three seed means. Whiskers show SD across seeds; points label seeds.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#4B5563",
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--plots-dir", default="results/plots")
    args = parser.parse_args()
    metrics_dir = Path(args.metrics_dir)
    plots_dir = Path(args.plots_dir)

    training = load_training_logs(metrics_dir)
    evaluations = load_evaluations(metrics_dir)
    per_seed, aggregate = summarize_evaluations(evaluations, metrics_dir)
    per_seed.to_csv(metrics_dir / "evaluation_by_seed.csv", index=False)
    aggregate.to_csv(metrics_dir / "evaluation_summary.csv", index=False)
    plot_training(training, plots_dir / "training_reward.png")
    plot_evaluation(per_seed, plots_dir / "evaluation_comparison.png")
    if {"uniform", "prioritized", "random"}.issubset(per_seed.policy):
        paired = per_seed.pivot(index="seed", columns="policy", values="mean")
        delta = paired["prioritized"] - paired["uniform"]
        print("\nPaired prioritized-minus-uniform seed differences:")
        print(delta.rename("reward difference").to_string())
        print(f"Mean paired difference: {delta.mean():.3f} rewards per episode")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
