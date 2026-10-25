import unittest
from unittest.mock import patch

import numpy as np

from dqn_project.arena import ArenaRound, keyboard_action, paired_scores


class ArenaTests(unittest.TestCase):
    def test_controls_and_combined_fire(self):
        meanings = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        self.assertEqual(keyboard_action(meanings, right=True, fire=True), 4)
        self.assertEqual(keyboard_action(meanings, left=True, right=True), 0)
        self.assertEqual(keyboard_action(meanings, up=True), 0)

    def test_pairing_excludes_aborts_other_checkpoints_and_caps(self):
        base = dict(game='game', checkpoint='a', seed=1, max_steps=10,
                    epsilon=0.05, noop_max=30, completed=True)
        rows = [dict(base, mode='human', score=3), dict(base, mode='ai', score=5),
                dict(base, mode='ai', score=100, completed=False),
                dict(base, mode='ai', score=200, checkpoint='b'),
                dict(base, mode='ai', score=300, max_steps=20)]
        self.assertEqual(paired_scores(rows, 'game', 'a'), [(1, 3, 5)])

    def test_human_cap_and_raw_score(self):
        class Fake:
            action_space = type('Space', (), {'n': 2})()
            def __init__(self):
                self.env = self
                self.unwrapped = self
            def reset(self, **kwargs):
                return np.zeros((4, 84, 84), np.uint8), {}
            def get_action_meanings(self):
                return ['NOOP', 'FIRE']
            def step(self, action):
                return np.zeros((4, 84, 84), np.uint8), 5, False, False, {'emulator_frames': 4}
            def close(self):
                pass
        with patch('dqn_project.arena.make_env', return_value=Fake()):
            game = ArenaRound('fake', 'human', 1, max_steps=2)
            game.step(1)
            game.step(1)
            game.step(1)
            self.assertEqual(game.score, 10)
            self.assertEqual(game.result()['reason'], 'time_limit')
            self.assertEqual(game.steps, 2)
            game.close()
