import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import json

from dqn_project.pong_server import create_app, discover_pretrained


class FakeController:
    def __init__(self, root):
        self.root = Path(root)
        self.process = None
        self.frame_bytes = b'jpeg'
        self.state = {'id': 'test', 'status': 'ready', 'history': []}
        self.inputs = []
        self.aborts = []
        self.cancelled = False
        self.payloads = []
        self.models = [{'id': 'pong_lab_dueling_best_learner', 'network': 'dueling', 'label': 'Trained',
                        'path': str(self.root / 'pong_lab_dueling_best_learner.pt'), 'epsilon': 0.05}]
    def snapshot(self):
        return self.state.copy()
    def request(self, action, payload=None):
        self.payloads.append((action, payload))
        if action == 'start':
            self.state['status'] = 'playing'
        return self.snapshot()
    def sessions(self):
        return []
    def checkpoints(self):
        return []
    def pretrained(self):
        return self.models
    def keys(self, keys):
        self.inputs.append(keys)
    def stop_input(self, reason):
        self.aborts.append(reason)
    def cancel_learning(self):
        self.cancelled = True
    def close(self):
        pass


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.control = FakeController(self.temp.name)
        self.client = TestClient(create_app(controller=self.control))
    def tearDown(self):
        self.client.close()
        self.temp.cleanup()
    def test_http_contract_and_remote_origin_rejected(self):
        self.assertEqual(self.client.get('/api/health').status_code, 200)
        self.assertEqual(self.client.get('/api/session').json()['status'], 'ready')
        self.assertEqual(self.client.post('/api/matches').json()['status'], 'playing')
        self.assertEqual(self.client.post('/api/sessions').status_code, 409)
        self.assertEqual(self.client.get('/api/session', headers={'origin': 'https://outside.example'}).status_code, 403)
        self.assertEqual(self.client.post('/api/training/cancel').json()['status'], 'cancelling')
        self.assertTrue(self.control.cancelled)
    def test_path_and_frame_contract(self):
        response = self.client.post('/api/sessions', json={'session_id': '../escape'})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.get('/api/frame.jpg').headers['content-type'], 'image/jpeg')
        self.assertEqual(self.client.get('/api/unknown').status_code, 404)
    def test_pretrained_models_are_listed_without_paths(self):
        models = self.client.get('/api/pretrained').json()
        self.assertEqual([m['id'] for m in models], ['pong_lab_dueling_best_learner'])
        self.assertNotIn('path', models[0])

    def test_session_from_pretrained_model_sets_checkpoint_network_and_epsilon(self):
        response = self.client.post('/api/sessions', json={
            'config': {'network': 'nature', 'seed': 3}, 'pretrained': 'pong_lab_dueling_best_learner'})
        self.assertEqual(response.status_code, 200)
        action, payload = self.control.payloads[-1]
        self.assertEqual(action, 'create')
        config = payload['config']
        self.assertEqual(config['checkpoint'], self.control.models[0]['path'])
        self.assertEqual(config['network'], 'dueling')
        self.assertEqual((config['epsilon_start'], config['epsilon_end'], config['seed']), (0.05, 0.05, 3))

    def test_unknown_model_or_arbitrary_checkpoint_rejected(self):
        self.assertEqual(self.client.post('/api/sessions', json={'pretrained': 'nope'}).status_code, 404)
        response = self.client.post('/api/sessions', json={'config': {'checkpoint': '/tmp/evil.pt'}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.control.payloads, [])

    def test_discover_pretrained_reads_network_from_args(self):
        folder = Path(self.temp.name) / 'models'
        folder.mkdir()
        (folder / 'pong_lab_nature_best_learner.pt').write_bytes(b'')
        (folder / 'pong_lab_nature_args.json').write_text(json.dumps({'network': 'nature', 'steps': 3000000}))
        (folder / 'pong_lab_dueling_best_learner.pt').write_bytes(b'')
        models = {m['id']: m for m in discover_pretrained(folder)}
        self.assertEqual(models['pong_lab_nature_best_learner']['network'], 'nature')
        self.assertIn('3M-step', models['pong_lab_nature_best_learner']['label'])
        self.assertEqual(models['pong_lab_dueling_best_learner']['network'], 'dueling')

    def test_control_input_disconnect_and_blur(self):
        with self.client.websocket_connect('/ws/play', headers={'origin': 'http://127.0.0.1:8765'}) as websocket:
            websocket.send_json({'keys': ['up', 'fire']})
            websocket.send_json({'keys': [], 'blur': True})
        self.assertIn(['up', 'fire'], self.control.inputs)
        self.assertIn('focus_lost', self.control.aborts)
        self.assertIn('disconnected', self.control.aborts)
