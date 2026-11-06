import unittest

import numpy as np

from dqn_project.pong_environment import PongEnvironment


class FakeParallel:
    possible_agents = ['first_0', 'second_0']
    action_mapping = np.array([0, 1, 3, 4, 11, 12])

    def __init__(self, stop_at=None, truncate=False):
        self.unwrapped = self
        self.stop_at = stop_at
        self.truncate = truncate
        self.closed = False
        self.actions = []

    def action_space(self, agent):
        return type('Space', (), {'n': 6})()

    def reset(self, seed=None):
        self.frame = 0
        self.seed = seed
        image = np.zeros((210, 160, 3), np.uint8)
        return dict.fromkeys(self.possible_agents, image), {}

    def step(self, actions):
        self.actions.append(actions)
        self.frame += 1
        image = np.full((210, 160, 3), self.frame, np.uint8)
        stop = self.stop_at == self.frame
        # A serve penalty need not give the other player positive reward.
        return (dict.fromkeys(self.possible_agents, image),
                {'first_0': -1.0, 'second_0': 0.0},
                dict.fromkeys(self.possible_agents, stop and not self.truncate),
                dict.fromkeys(self.possible_agents, stop and self.truncate), {})

    def close(self):
        self.closed = True


class PongEnvironmentTests(unittest.TestCase):
    def test_machine_transitions_repeat_four_raw_frames(self):
        raw = FakeParallel()
        game = PongEnvironment(_parallel_env=raw)
        state = game.reset(7)
        self.assertEqual(state.shape, (4, 84, 84))
        self.assertEqual(state.dtype, np.uint8)
        successor, reward, term, trunc, info = game.step(2, 3)
        self.assertEqual(raw.frame, 4)
        self.assertEqual(info['emulator_frames'], 4)
        self.assertEqual(info['human_reward'], -4)
        self.assertEqual(reward, 0)
        self.assertFalse(term or trunc)
        self.assertEqual(raw.actions[0], {'first_0': 2, 'second_0': 3})
        self.assertTrue(np.all(successor[-1] == 4))
        self.assertTrue(np.all(successor[:-1] == 0))
        self.assertEqual(game.render().shape, (210, 160, 3))
        self.assertIsNone(info['human_goals'])
        self.assertEqual(info['score_mode'], 'return')

    def test_truncation_is_not_real_terminal(self):
        game = PongEnvironment(_parallel_env=FakeParallel(stop_at=2, truncate=True))
        game.reset(1)
        state, _, term, trunc, info = game.step(0, 0)
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(info['emulator_frames'], 2)
        self.assertTrue(np.all(state[-1] == 2))
        with self.assertRaises(RuntimeError):
            game.step(0, 0)

    def test_controls_resolve_actual_action_codes(self):
        game = PongEnvironment(_parallel_env=FakeParallel())
        self.assertEqual(game.action_for_keys({'up'}), 2)
        self.assertEqual(game.action_for_keys({'down'}), 3)
        self.assertEqual(game.action_for_keys({'up', 'fire'}), 4)
        self.assertEqual(game.action_for_keys({'down', 'fire'}), 5)
        self.assertEqual(game.action_for_keys({'up', 'down'}), 0)
        self.assertEqual(game.action_for_keys({'fire'}), 1)
        game.reset(1)
        with self.assertRaises(ValueError):
            game.step(99, 0)
        game.close()
        self.assertTrue(game.env.closed)

    def test_unknown_mapping_is_rejected(self):
        raw = FakeParallel()
        raw.action_mapping = np.array([0, 1])
        with self.assertRaises(RuntimeError):
            PongEnvironment(_parallel_env=raw)
        self.assertTrue(raw.closed)
