"""Build a reviewed snapshot for the local DQN experiment report app."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results/metrics"
OUTPUT = ROOT / "results/reviewed_experiment_snapshot.json"


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def convert_training_value(key: str, value: str) -> object:
    if key in {"run_id", "replay_type", "config_id", "event"}:
        return value
    if key in {"seed", "environment_steps", "frames", "episode", "episode_length"}:
        return int(value)
    if not value:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def metric(label: str, definition: str, component_ids: list[str]) -> dict[str, object]:
    return {
        "label": label,
        "definition": definition,
        "componentIds": component_ids,
    }


def main() -> None:
    evaluation_rows: list[dict[str, object]] = []
    for path in sorted(METRICS.glob("breakout_*_seed*_evaluation.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        for episode, reward in enumerate(result["episode_rewards"], start=1):
            evaluation_rows.append(
                {
                    "policy": result["replay_type"],
                    "seed": int(result["seed"]),
                    "episode": episode,
                    "reward": float(reward),
                    "length": int(result["episode_lengths"][episode - 1]),
                }
            )

    seed_rows = [
        {
            "policy": row["policy"],
            "seed": int(row["seed"]),
            "episodes": int(row["episodes"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "minimum": float(row["minimum"]),
            "maximum": float(row["maximum"]),
        }
        for row in read_csv(METRICS / "evaluation_by_seed.csv")
    ]

    training_rows: list[dict[str, object]] = []
    for path in sorted(METRICS.glob("breakout_*_seed*.csv")):
        for raw in read_csv(path):
            if raw.get("event") != "episode":
                continue
            row = {key: convert_training_value(key, value) for key, value in raw.items()}
            row["seed_label"] = f"Seed {row['seed']}"
            row["series_label"] = f"{str(row['replay_type']).title()} · seed {row['seed']}"
            training_rows.append(row)

    for replay_type in ("uniform", "prioritized"):
        for seed in (7, 17, 29):
            run_rows = sorted(
                (row for row in training_rows if row["replay_type"] == replay_type and row["seed"] == seed),
                key=lambda row: int(row["environment_steps"]),
            )
            rewards: list[float] = []
            for row in run_rows:
                rewards.append(float(row["episode_reward"]))
                row["rolling_reward_50"] = sum(rewards[-50:]) / len(rewards[-50:])

    for row in seed_rows:
        row["policy_label"] = str(row["policy"]).title()
        row["seed_label"] = f"Seed {row['seed']}"

    files = [
        f"results/metrics/{path.name}"
        for path in sorted(METRICS.glob("breakout_*_seed*.csv"))
    ]
    files.extend(
        f"results/metrics/{path.name}"
        for path in sorted(METRICS.glob("breakout_*_seed*_evaluation.json"))
    )
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    snapshot = {
        "surface": "report",
        "title": "Breakout DQN Reproduction Results",
        "status": "observed",
        "generatedAt": generated_at,
        "report": {"asOf": generated_at[:10]},
        "filters": [],
        "queries": {
            "evaluation": {
                "rows": evaluation_rows,
                "source": {
                    "label": "Held-out ALE evaluation episode outputs",
                    "files": files,
                    "filters": [
                        "ALE/Breakout-v5",
                        "10 episodes per policy and seed",
                        "Seeds 7, 17 and 29",
                        "Epsilon 0.05 for learned policies",
                        "Maximum 4,500 selected actions per episode",
                    ],
                    "assumptions": [
                        "The three training seeds were selected before the full runs.",
                        "The six DQN checkpoints used matched 250,000-decision budgets and differ only by replay type within each seed.",
                        "The random policy uses the same evaluation wrapper and reset seeds.",
                        "Three training seeds provide limited evidence about algorithm-level differences.",
                    ],
                    "metricDefinitions": [
                        metric(
                            "Evaluation episode reward",
                            "Unclipped game score accumulated during one held-out episode.",
                            ["evaluation-by-seed", "evaluation-episodes"],
                        ),
                        metric(
                            "Evaluation episode length",
                            "Selected actions before game termination or the 4,500-action cap.",
                            ["evaluation-episodes"],
                        ),
                    ],
                },
            },
            "seed_summary": {
                "rows": seed_rows,
                "source": {
                    "label": "Per-seed summaries computed from held-out episode outputs",
                    "files": ["results/metrics/evaluation_by_seed.csv"],
                    "metricDefinitions": [
                        metric(
                            "Mean evaluation reward",
                            "Arithmetic mean of ten held-out episode rewards within one policy and training seed.",
                            ["evaluation-by-seed"],
                        ),
                        metric(
                            "Across-seed spread",
                            "Standard deviation across three per-seed mean rewards; this describes seed variation and is not a confidence interval.",
                            ["evaluation-by-seed"],
                        ),
                    ],
                },
                "methods": [
                        {
                            "language": "Python",
                            "code": "Group the ten recorded evaluation episode rewards by replay policy and training seed; compute count, mean, sample standard deviation, minimum and maximum.",
                        }
                    ],
            },
            "training": {
                "rows": training_rows,
                "source": {
                    "label": "Episode-level training logs for six matched DQN runs",
                    "files": files,
                    "filters": ["Training horizon: 250,000 selected-action decisions per run"],
                    "metricDefinitions": [
                        metric(
                            "Training episode reward",
                            "Unclipped ALE score recorded during training under the current epsilon-greedy behavior policy.",
                            ["uniform-training", "prioritized-training"],
                        ),
                        metric(
                            "Environment decisions",
                            "Selected actions. Each selected action repeats across four emulator frames in the configured wrapper.",
                            ["uniform-training", "prioritized-training"],
                        ),
                    ],
                },
                "methods": [
                        {
                            "language": "Python",
                            "code": "Plot raw training episode rewards against environment_steps, separately by replay method and seed. The training logs contain episode reward, not evaluation reward.",
                        }
                    ],
            },
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    app_data = ROOT / "results/report_app/src/data.json"
    if app_data.exists():
        current = json.loads(app_data.read_text(encoding="utf-8"))
        snapshot["id"] = current["id"]
        snapshot["buildStatus"] = "complete"
        app_data.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
