import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dqn_project.learner import DQNLearner
from dqn_project.prioritized_replay import PrioritizedReplayBuffer
from dqn_project.replay_buffer import ReplayBatch


def sample_batch(batch_size: int = 2) -> ReplayBatch:
    rng = np.random.default_rng(2)
    return ReplayBatch(
        states=rng.integers(0, 256, (batch_size, 4, 84, 84), dtype=np.uint8),
        actions=np.arange(batch_size, dtype=np.int64) % 3,
        rewards=np.array([1.0, -1.0], dtype=np.float32)[:batch_size],
        next_states=rng.integers(0, 256, (batch_size, 4, 84, 84), dtype=np.uint8),
        terminated=np.array([True, False], dtype=np.bool_)[:batch_size],
        weights=np.ones(batch_size, dtype=np.float32),
        indices=np.arange(batch_size, dtype=np.int64),
    )


class LearnerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.learner = DQNLearner(action_count=3)

    def test_terminal_and_nonterminal_targets(self) -> None:
        for parameter in self.learner.target.parameters():
            parameter.data.zero_()
        self.learner.target.head[-1].bias.data.fill_(2.0)
        rewards = torch.tensor([1.0, 1.0])
        next_states = torch.zeros(2, 4, 84, 84)
        terminated = torch.tensor([True, False])
        targets = self.learner.compute_targets(rewards, next_states, terminated)
        self.assertAlmostEqual(targets[0].item(), 1.0, places=5)
        self.assertAlmostEqual(targets[1].item(), 1.0 + 0.99 * 2.0, places=5)

    def test_update_is_finite_and_changes_online_parameters(self) -> None:
        before = [p.detach().clone() for p in self.learner.online.parameters()]
        metrics = self.learner.update(sample_batch())
        self.assertTrue(np.isfinite(metrics.loss))
        self.assertEqual(metrics.td_errors.shape, (2,))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, self.learner.online.parameters())))

    def test_target_sync_and_checkpoint_round_trip(self) -> None:
        with torch.no_grad():
            next(self.learner.online.parameters()).add_(1.0)
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(self.learner.online.parameters(), self.learner.target.parameters())))
        self.learner.sync_target()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(self.learner.online.parameters(), self.learner.target.parameters())))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            self.learner.save(path, {"test": True})
            restored = DQNLearner(action_count=3)
            metadata = restored.load(path)
            self.assertEqual(metadata, {"test": True})
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(self.learner.online.parameters(), restored.online.parameters())))

    def test_prioritized_replay_is_learner_compatible(self) -> None:
        replay = PrioritizedReplayBuffer(4, seed=5)
        rng = np.random.default_rng(6)
        for action in range(4):
            replay.add(
                rng.integers(0, 256, (4, 84, 84), dtype=np.uint8),
                action % 3,
                float(action == 0),
                rng.integers(0, 256, (4, 84, 84), dtype=np.uint8),
                action == 3,
            )
        batch = replay.sample(2, beta=0.4)
        metrics = self.learner.update(batch)
        replay.update_priorities(batch.indices, metrics.td_errors)
        self.assertTrue(np.all(replay.priorities[batch.indices] > 0))


if __name__ == "__main__":
    unittest.main()
