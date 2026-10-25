import unittest
from unittest.mock import patch

import numpy as np

from dqn_project.evaluate import evaluate_policy


class _ActionSpace:
    n = 2


class _EndlessEnv:
    action_space = _ActionSpace()

    def reset(self, **kwargs):
        return np.zeros((4, 84, 84), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((4, 84, 84), dtype=np.uint8), 1.0, False, False, {}

    def close(self):
        pass


class EvaluationTests(unittest.TestCase):
    def test_render_mode_is_forwarded(self) -> None:
        with patch("dqn_project.evaluate.make_env", return_value=_EndlessEnv()) as factory:
            evaluate_policy(None, "fake", 1, 0.0, 7, max_steps_per_episode=1, render=True)
            factory.assert_called_once_with("fake", render_mode="human")

    def test_episode_cap_is_enforced_and_reported(self) -> None:
        with patch("dqn_project.evaluate.make_env", return_value=_EndlessEnv()):
            result = evaluate_policy(
                network=None,
                env_id="fake",
                episodes=2,
                epsilon=0.0,
                seed=7,
                max_steps_per_episode=3,
            )
        self.assertEqual(result["episode_rewards"], [3.0, 3.0])
        self.assertEqual(result["episode_lengths"], [3, 3])
        self.assertEqual(result["max_steps_per_episode"], 3)


if __name__ == "__main__":
    unittest.main()
