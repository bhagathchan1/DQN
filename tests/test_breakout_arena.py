import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from dqn_project.breakout_arena import BreakoutArena, discover_models, human_action

BREAKOUT_ACTIONS = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']


class KeyMappingTests(unittest.TestCase):
    def test_breakout_keys(self):
        self.assertEqual(human_action(BREAKOUT_ACTIONS, set()), 0)
        self.assertEqual(human_action(BREAKOUT_ACTIONS, {'left'}), 3)
        self.assertEqual(human_action(BREAKOUT_ACTIONS, {'right'}), 2)
        self.assertEqual(human_action(BREAKOUT_ACTIONS, {'left', 'right'}), 0)
        # No LEFTFIRE in Breakout: holding Space still launches the ball.
        self.assertEqual(human_action(BREAKOUT_ACTIONS, {'left', 'fire'}), 1)
from dqn_project.pong_server import create_app


class FakeRound:
    """Stands in for ArenaRound: scores one point per FIRE (human) or per step (DQN)."""
    created = []
    default_limit = 5

    def __init__(self, env_id, mode, seed, checkpoint=None, max_steps=4500, epsilon=0.05):
        self.env_id, self.mode, self.seed, self.checkpoint = env_id, mode, seed, checkpoint
        self.max_steps, self.epsilon = max_steps, epsilon
        self.meanings = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        self.actions = []
        self.score, self.steps, self.done, self.reason = 0.0, 0, False, ''
        self.limit = FakeRound.default_limit
        FakeRound.created.append(self)

    def step(self, action=0):
        self.actions.append(action)
        self.steps += 1
        if self.mode == 'ai' or action == 1:
            self.score += 1
        self.done = self.steps >= self.limit
        if self.done:
            self.reason = 'game_over'

    def frame(self):
        return np.zeros((210, 160, 3), dtype=np.uint8)

    def lives(self):
        return 5

    def result(self):
        return dict(timestamp_utc=f'2026-11-14T00:00:{len(FakeRound.created):02d}', game=self.env_id,
                    mode=self.mode, seed=self.seed, score=self.score, decisions=self.steps,
                    emulator_frames=self.steps * 4, max_steps=self.max_steps, epsilon=self.epsilon,
                    noop_max=30, checkpoint=self.checkpoint or '', checkpoint_training_steps=None,
                    completed=self.done, reason=self.reason or 'aborted')

    def close(self):
        pass


def wait_finished(arena, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if arena.snapshot()['status'] != 'playing':
            return arena.snapshot()
        time.sleep(0.01)
    raise AssertionError('round did not finish')


class BreakoutArenaTests(unittest.TestCase):
    def setUp(self):
        FakeRound.created.clear()
        FakeRound.default_limit = 5
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.models = root / 'models'
        self.models.mkdir()
        for name in ['breakout_uniform_10M_seed7_final', 'breakout_upstream_10M_seed7_final']:
            (self.models / f'{name}.pt').write_bytes(b'not loaded by the fake round')
        self.results = root / 'scores.csv'
        self.arena = BreakoutArena(self.results, models_dir=self.models, round_factory=FakeRound, period=0.001)

    def tearDown(self):
        self.arena.close()
        self.directory.cleanup()

    def test_models_are_listed_by_id_with_uniform_first(self):
        models = discover_models(self.models)
        self.assertEqual([m['kind'] for m in models], ['uniform', 'upstream'])
        self.assertTrue(all('path' in m and 'detail' in m for m in models))

    def test_human_then_dqn_on_same_seed_are_paired_and_saved(self):
        self.arena.period = 0.03  # slow enough that the keys arrive mid-round
        self.arena.start('human', 11)
        self.arena.set_keys(['fire', 'left'])
        human = wait_finished(self.arena)
        self.assertTrue(human['completed'])
        fires = FakeRound.created[0].actions.count(1)
        self.assertGreaterEqual(fires, 1, 'FIRE (with LEFT) maps to the FIRE action')
        self.assertEqual(human['score'], float(fires))
        model = discover_models(self.models)[0]['id']
        self.arena.start('ai', 11, model, speed=4)
        wait_finished(self.arena)
        ai_round = FakeRound.created[1]
        self.assertEqual((ai_round.seed, ai_round.max_steps), (11, 4500), 'DQN gets the same seed and limit')
        self.assertTrue(ai_round.checkpoint.endswith(f'{model}.pt'))
        pairs = self.arena.snapshot()['pairs']
        self.assertEqual(pairs[0]['seed'], 11)
        self.assertEqual((pairs[0]['human'], pairs[0]['dqn']), (float(fires), 5.0))
        self.assertEqual(pairs[0]['winner'], 'draw' if fires == 5 else 'dqn')
        # Results persist and reload in a new arena.
        reloaded = BreakoutArena(self.results, models_dir=self.models, round_factory=FakeRound)
        self.assertEqual(len(reloaded.pairs()), 1)

    def test_dqn_round_requires_a_known_model(self):
        with self.assertRaisesRegex(ValueError, 'trained Breakout'):
            self.arena.start('ai', 3, 'missing-model')

    def test_invalid_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            self.arena.start('watch', 3)
        with self.assertRaises(ValueError):
            self.arena.start('human', -1)
        with self.assertRaises(ValueError):
            self.arena.start('human', 3, speed=3)

    def test_human_round_stops_without_input_and_is_not_paired(self):
        FakeRound.default_limit = 10**9
        self.arena.start('human', 5)
        self.arena.last_input -= 10  # simulate a lost key heartbeat
        state = wait_finished(self.arena)
        self.assertFalse(state['completed'])
        self.assertEqual(state['reason'], 'input_timeout')
        self.assertEqual(self.arena.pairs(), [])

    def test_only_one_round_at_a_time_and_abort(self):
        FakeRound.default_limit = 10**9
        self.arena.start('human', 8)
        self.arena.set_keys([])  # keep the heartbeat alive
        with self.assertRaisesRegex(ValueError, 'already running'):
            self.arena.start('human', 9)
        state = self.arena.abort()
        self.assertEqual(state['status'], 'finished')
        self.assertFalse(state['completed'])


class FakePong:
    def __init__(self, status='ready'):
        self.state = {'status': status}
        self.process = None
        self.frame_bytes = None
    def snapshot(self):
        return dict(self.state)
    def request(self, action, payload=None):
        return self.snapshot()
    def close(self):
        pass


class BreakoutEndpointTests(unittest.TestCase):
    def setUp(self):
        FakeRound.created.clear()
        FakeRound.default_limit = 5
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        (root / 'breakout_uniform_10M_seed7_final.pt').write_bytes(b'x')
        self.arena = BreakoutArena(root / 'scores.csv', models_dir=root, round_factory=FakeRound, period=0.001)
        self.pong = FakePong()
        self.client = TestClient(create_app(root, controller=self.pong, breakout=self.arena))

    def tearDown(self):
        self.arena.close()
        self.directory.cleanup()

    def test_models_hide_server_paths(self):
        models = self.client.get('/api/breakout/models').json()
        self.assertEqual(models[0]['id'], 'breakout_uniform_10M_seed7_final')
        self.assertNotIn('path', models[0])

    def test_round_lifecycle_over_http(self):
        response = self.client.post('/api/breakout/rounds', json={'mode': 'ai', 'seed': 4,
                                                                 'model': 'breakout_uniform_10M_seed7_final'})
        self.assertEqual(response.status_code, 200, response.text)
        wait_finished(self.arena)
        state = self.client.get('/api/breakout/state').json()
        self.assertEqual(state['status'], 'finished')
        frame = self.client.get('/api/breakout/frame.jpg')
        self.assertEqual(frame.headers['content-type'], 'image/jpeg')

    def test_rejects_bad_requests(self):
        self.assertEqual(self.client.post('/api/breakout/rounds', json={'mode': 'ai', 'seed': 1, 'model': '../x'}).status_code, 422)
        self.assertEqual(self.client.post('/api/breakout/rounds', json={'mode': 'ai', 'seed': 1, 'model': 'nope'}).status_code, 409)

    def test_games_do_not_run_at_the_same_time(self):
        self.pong.state['status'] = 'playing'
        response = self.client.post('/api/breakout/rounds', json={'mode': 'human', 'seed': 1})
        self.assertEqual(response.status_code, 409)
        self.pong.state['status'] = 'ready'
        FakeRound.default_limit = 10**9
        self.client.post('/api/breakout/rounds', json={'mode': 'human', 'seed': 1})
        self.assertEqual(self.client.post('/api/matches').status_code, 409)


if __name__ == '__main__':
    unittest.main()
