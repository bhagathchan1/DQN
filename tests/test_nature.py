import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dqn_project.frame_replay import FrameReplayBuffer
from dqn_project.optimizer import NatureRMSprop
from dqn_project.environment import DQNEnv


class NatureTests(unittest.TestCase):
    def test_frame_replay_matches_full_stacks_across_resets_and_wrap(self):
        replay = FrameReplayBuffer(12, seed=3)
        expected = {}
        for index in range(40):
            if index % 7 == 0:
                state = np.full((4, 84, 84), index, np.uint8)
                replay.start_episode()
            successor = np.concatenate([state[1:], np.full((1, 84, 84), index+1, np.uint8)])
            replay.add(state, index, float(index), successor, index % 7 == 6)
            expected[index] = (state.copy(), successor.copy())
            state = successor
            batch = replay.sample(len(replay))
            for i, action in enumerate(batch.actions):
                np.testing.assert_array_equal(batch.states[i], expected[action][0])
                np.testing.assert_array_equal(batch.next_states[i], expected[action][1])
        self.assertLessEqual(len(replay.starts), 2)
        self.assertEqual(replay.frames.nbytes, 12 * 84 * 84)

    def test_disk_backing(self):
        with tempfile.TemporaryDirectory() as temp:
            replay = FrameReplayBuffer(8, directory=temp)
            state = np.zeros((4, 84, 84), np.uint8)
            replay.add(state, 0, 0, state, False)
            self.assertEqual(replay.sample(1).states.shape, (1, 4, 84, 84))
            replay.close()
            self.assertEqual((Path(temp) / 'frames.uint8').stat().st_size, 8*84*84)

    def test_nature_rmsprop_equation_and_state(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = NatureRMSprop([parameter])
        parameter.grad = torch.tensor([2.0])
        optimizer.step()
        expected = 1 - 0.00025 * 2 / np.sqrt(0.05 * 4 - (0.05*2)**2 + 0.01)
        self.assertAlmostEqual(parameter.item(), expected, places=6)
        self.assertAlmostEqual(optimizer.state[parameter]['mean'].item(), 0.1, places=6)

    def test_noops_seeded_and_life_loss_does_not_end_game(self):
        class Fake:
            def reset(self, **kwargs):
                self.steps = 0
                return np.zeros((84, 84), np.uint8), {'lives': 3}
            def step(self, action):
                self.steps += 1
                return np.full((84, 84), self.steps, np.uint8), 0, False, False, {'lives': 2 if self.steps >= 31 else 3}
        wrapped = DQNEnv(Fake(), noop_max=30)
        first, _ = wrapped.reset(seed=123)
        second, _ = wrapped.reset(seed=123)
        np.testing.assert_array_equal(first, second)
        wrapped.env.steps = 30
        _, _, terminated, _, info = wrapped.step(0)
        self.assertTrue(info['life_lost'])
        self.assertFalse(terminated)
        self.assertEqual(info['emulator_frames'], 4)
