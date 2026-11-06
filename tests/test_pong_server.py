import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from dqn_project.pong_server import create_app


class FakeController:
    def __init__(self, root):
        self.root = Path(root)
        self.process = None
        self.frame_bytes = b'jpeg'
        self.state = {'id': 'test', 'status': 'ready', 'history': []}
        self.inputs = []
        self.aborts = []
        self.cancelled = False
    def snapshot(self):
        return self.state.copy()
    def request(self, action, payload=None):
        if action == 'start':
            self.state['status'] = 'playing'
        return self.snapshot()
    def sessions(self):
        return []
    def checkpoints(self):
        return []
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
    def test_control_input_disconnect_and_blur(self):
        with self.client.websocket_connect('/ws/play', headers={'origin': 'http://127.0.0.1:8765'}) as websocket:
            websocket.send_json({'keys': ['up', 'fire']})
            websocket.send_json({'keys': [], 'blur': True})
        self.assertIn(['up', 'fire'], self.control.inputs)
        self.assertIn('focus_lost', self.control.aborts)
        self.assertIn('disconnected', self.control.aborts)
