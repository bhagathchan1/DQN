import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dqn_project.agent import EpsilonSchedule
from dqn_project.config import load_config
from dqn_project.logging_utils import METRIC_FIELDS, MetricsLogger
from dqn_project.plot_results import plot_metrics


class AgentAndLoggingTests(unittest.TestCase):
    def test_epsilon_schedule(self) -> None:
        schedule = EpsilonSchedule(1.0, 0.1, 100)
        self.assertEqual(schedule.value(0), 1.0)
        self.assertAlmostEqual(schedule.value(50), 0.55)
        self.assertEqual(schedule.value(200), 0.1)

    def test_default_config_loads(self) -> None:
        config = load_config("configs/default.yaml")
        self.assertEqual(config["environment"]["id"], "ALE/Breakout-v5")

    def test_metric_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            logger = MetricsLogger(path)
            logger.log(run_id="test", event="episode", episode_reward=1.5)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(list(rows[0]), METRIC_FIELDS)
            self.assertEqual(rows[0]["run_id"], "test")
            self.assertEqual(float(rows[0]["episode_reward"]), 1.5)

    def test_plot_is_generated_from_logged_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.csv"
            logger = MetricsLogger(metrics)
            for step, reward in [(10, 1.0), (20, 2.0), (30, 3.0)]:
                logger.log(
                    run_id="uniform_seed7",
                    replay_type="uniform",
                    event="episode",
                    environment_steps=step,
                    episode_reward=reward,
                )
            output = plot_metrics([str(metrics)], str(Path(directory) / "curve.png"))
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
