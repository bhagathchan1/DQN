import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from dqn_project import training_data
from dqn_project.pong_server import create_app
from dqn_project.training_curves import smooth


class FakeRound:
    """Ten decisions; rewards at decisions 3 (+1) and 7 (-1); the game ends at 10."""
    def __init__(self, env_id, mode, seed, checkpoint=None, max_steps=600, epsilon=0.05):
        self.env_id, self.mode, self.checkpoint, self.epsilon = env_id, mode, checkpoint, epsilon
        self.meanings = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        self.state = np.zeros((4, 84, 84), dtype=np.uint8)
        self.steps, self.score, self.done, self.reason = 0, 0.0, False, ''

    def step(self, action=0):
        self.steps += 1
        self.last_action = self.steps % 4
        self.last_reward = 1.0 if self.steps == 3 else -1.0 if self.steps == 7 else 0.0
        self.score += self.last_reward
        self.state = np.roll(self.state, -1, axis=0)
        self.state[-1] = self.steps
        if self.steps >= 10:
            self.done, self.reason = True, 'game_over'

    def close(self):
        pass


class SmoothingTests(unittest.TestCase):
    def test_trailing_mean_and_thinning(self):
        points = [(i, float(i % 2)) for i in range(1000)]
        result = smooth(points, window=10, max_points=50)
        self.assertLessEqual(len(result), 51)
        self.assertEqual(result[0], [0, 0.0])
        self.assertEqual(result[-1][0], 999, 'the last point is always kept')
        self.assertAlmostEqual(result[-1][1], 0.5)

    def test_empty(self):
        self.assertEqual(smooth([]), [])


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        training_data._cache.clear()

    def test_pick_keeps_reward_moments_and_fills_with_quiet_ones(self):
        rewards = [0.0] * 100
        rewards[10], rewards[50] = 1.0, -1.0
        picked = training_data._pick(rewards, shown=8)
        self.assertEqual(len(picked), 8)
        self.assertIn(10, picked)
        self.assertIn(50, picked)
        self.assertEqual(picked, sorted(picked))

    def test_sample_records_real_transition_fields(self):
        data = training_data.sample_experience('breakout', 'random', seed=3, round_factory=FakeRound)
        self.assertEqual((data['decisions'], data['positive'], data['negative']), (10, 1, 1))
        self.assertEqual(data['epsilon'], 1.0)
        self.assertEqual(sum(item['count'] for item in data['actions']), 10)
        by_step = {item['step']: item for item in data['transitions']}
        self.assertEqual(by_step[3]['reward'], 1.0)
        self.assertEqual(by_step[7]['training_reward'], -1.0)
        self.assertTrue(by_step[3]['frames'].startswith('data:image/png;base64,'))
        self.assertTrue(by_step[10]['game_over'] if 10 in by_step else True)
        # Cached: the same request does not replay the game.
        self.assertIs(training_data.sample_experience('breakout', 'random', seed=3, round_factory=FakeRound), data)

    def test_sample_validates_inputs(self):
        with self.assertRaises(ValueError):
            training_data.sample_experience('tetris', 'random', round_factory=FakeRound)
        with self.assertRaises(ValueError):
            training_data.sample_experience('pong', 'random', seed=-1, round_factory=FakeRound)


class RecordsTests(unittest.TestCase):
    def test_paging_and_summary_on_a_real_log(self):
        first = training_data.records('pong_lab_dueling', 0, 5)
        self.assertEqual(len(first['rows']), 5)
        self.assertEqual(len(first['columns']), len(first['rows'][0]))
        self.assertGreater(first['summary']['games'], 1000)
        last = training_data.records('pong_lab_dueling', first['total'] - 2, 5)
        self.assertEqual(len(last['rows']), 2)

    def test_unknown_run_and_bad_paging_are_rejected(self):
        with self.assertRaises(ValueError):
            training_data.run_path('../../etc/passwd')
        with self.assertRaises(ValueError):
            training_data.records('pong_nature', 0, 1000)


class FakePong:
    def __init__(self):
        self.process = None
        self.frame_bytes = None
    def snapshot(self):
        return {'status': 'ready'}
    def close(self):
        pass


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.client = TestClient(create_app(Path(self.directory.name), controller=FakePong()))

    def tearDown(self):
        self.directory.cleanup()

    def test_runs_records_and_download(self):
        runs = self.client.get('/api/training/runs').json()
        self.assertIn('breakout_uniform', [run['id'] for run in runs])
        page = self.client.get('/api/training/records', params={'run': 'pong_lab_nature', 'limit': 3})
        self.assertEqual(page.status_code, 200)
        self.assertEqual(len(page.json()['rows']), 3)
        download = self.client.get('/api/training/records/pong_lab_nature.csv')
        self.assertEqual(download.status_code, 200)
        self.assertIn('attachment', download.headers['content-disposition'])
        self.assertTrue(download.text.startswith('event,step,game'))

    def test_bad_requests(self):
        self.assertEqual(self.client.get('/api/training/records', params={'run': 'nope'}).status_code, 422)
        self.assertEqual(self.client.get('/api/training/records/nope.csv').status_code, 404)
        self.assertEqual(self.client.get('/api/training/experience', params={'game': 'tetris'}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
