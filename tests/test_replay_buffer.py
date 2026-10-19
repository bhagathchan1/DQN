import unittest

import numpy as np

from dqn_project.prioritized_replay import PrioritizedReplayBuffer
from dqn_project.replay_buffer import ReplayBuffer


def state(value: int) -> np.ndarray:
    return np.full((4, 84, 84), value, dtype=np.uint8)


class ReplayBufferTests(unittest.TestCase):
    def test_capacity_wraparound_and_sample_contract(self) -> None:
        replay = ReplayBuffer(3, seed=4)
        for value in range(5):
            replay.add(state(value), value, float(value), state(value + 1), value % 2 == 0)
        self.assertEqual(len(replay), 3)
        self.assertNotIn(0, replay.states[:, 0, 0, 0])
        batch = replay.sample(2)
        self.assertEqual(batch.states.shape, (2, 4, 84, 84))
        self.assertEqual(batch.states.dtype, np.uint8)
        self.assertEqual(batch.actions.dtype, np.int64)
        self.assertEqual(batch.rewards.dtype, np.float32)
        self.assertEqual(batch.terminated.dtype, np.bool_)
        self.assertTrue(np.all(batch.weights == 1.0))

    def test_float_state_conversion(self) -> None:
        replay = ReplayBuffer(1)
        replay.add(np.ones((4, 84, 84), dtype=np.float32), 0, 0, state(0), False)
        self.assertEqual(int(replay.states[0, 0, 0, 0]), 255)

    def test_prioritized_probabilities_weights_and_updates(self) -> None:
        replay = PrioritizedReplayBuffer(4, alpha=1.0, seed=3)
        for value in range(4):
            replay.add(state(value), 0, 0, state(value), False)
        replay.update_priorities(np.arange(4), np.array([1, 2, 3, 4], dtype=np.float32))
        probabilities = replay.sampling_probabilities()
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(np.all(probabilities > 0))
        self.assertGreater(probabilities[-1], probabilities[0])
        batch = replay.sample(3, beta=0.4)
        self.assertEqual(batch.indices.shape, (3,))
        self.assertTrue(np.all(batch.weights > 0))
        self.assertLessEqual(float(batch.weights.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
