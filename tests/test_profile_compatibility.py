import unittest
from pathlib import Path

import numpy as np
import torch

from dqn_project.config import load_config
from dqn_project.environment import DQNEnv
from dqn_project.learner import DQNLearner
from dqn_project.replay_buffer import ReplayBuffer
from dqn_project.upstream_data import MaxAndSkipEnv


class ProfileCompatibilityTests(unittest.TestCase):
    def test_both_profiles_can_optimize(self):
        root = Path(__file__).resolve().parents[1]
        replay = ReplayBuffer(8)
        state = np.zeros((4, 84, 84), np.uint8)
        for _ in range(2):
            replay.add(state, 0, 1, state, True)
        for filename in ['nature.yaml', 'upstream_breakout.yaml']:
            config = load_config(root / 'configs' / filename)['training']
            learner = DQNLearner(2, optimizer=config['optimizer'],
                                 gradient_clip=config.get('gradient_clip', 10.0),
                                 gradient_clip_mode=config.get('gradient_clip_mode', 'global'),
                                 initializer=config.get('initializer', 'pytorch'))
            before = learner.online.head[-1].bias.detach().clone()
            result = learner.update(replay.sample(2))
            self.assertTrue(np.isfinite(result.loss))
            self.assertFalse(torch.equal(before, learner.online.head[-1].bias))

    def test_nested_skip_preserves_emulator_frame_count(self):
        class Raw:
            action_space = object()
            def reset(self, **kwargs):
                return np.zeros((84, 84), np.uint8), {}
            def step(self, action):
                return np.zeros((84, 84), np.uint8), 1, False, False, {}
        env = DQNEnv(MaxAndSkipEnv(Raw(), skip=4), action_repeat=1)
        env.reset()
        _, reward, _, _, info = env.step(0)
        self.assertEqual(reward, 4)
        self.assertEqual(info['emulator_frames'], 4)
