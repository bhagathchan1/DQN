import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_results import summarize_evaluations  # noqa: E402


class ResultAnalysisTests(unittest.TestCase):
    def test_seed_and_pooled_summaries(self) -> None:
        evaluations = pd.DataFrame(
            [
                {"policy": "uniform", "seed": 7, "reward": 1.0},
                {"policy": "uniform", "seed": 7, "reward": 3.0},
                {"policy": "uniform", "seed": 17, "reward": 5.0},
                {"policy": "uniform", "seed": 17, "reward": 7.0},
            ]
        )
        per_seed, aggregate = summarize_evaluations(evaluations)
        self.assertEqual(per_seed["mean"].tolist(), [2.0, 6.0])
        self.assertEqual(int(aggregate.loc[0, "seeds"]), 2)
        self.assertEqual(float(aggregate.loc[0, "mean_of_seed_means"]), 4.0)
        self.assertEqual(int(aggregate.loc[0, "pooled_episodes"]), 4)


if __name__ == "__main__":
    unittest.main()
