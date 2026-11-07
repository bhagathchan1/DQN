import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dqn_project.learner import UpdateMetrics
from dqn_project.optimizer import NatureRMSprop
from dqn_project.pong_session import PongSession, validate_config


class FakeEnvironment:
    action_count = 2
    spec = {'family': 'test-pong', 'actions': ['NOOP', 'UP'], 'machine_side': 'right'}

    def reset(self, seed):
        self.state = np.zeros((4, 84, 84), dtype=np.uint8)
        self.steps = 0
        self.actions = []
        return self.state.copy()

    def step(self, human_action, machine_action):
        self.steps += 1
        self.actions.append((human_action, machine_action))
        self.state = np.concatenate([self.state[1:], np.full((1, 84, 84), self.steps, np.uint8)])
        reward = 2.0 if machine_action == 1 else -1.0
        return self.state.copy(), reward, False, False, {'human_reward': -reward, 'emulator_frames': 4}

    def action_for_keys(self, keys):
        return int('up' in keys)

    def render(self):
        return np.zeros((210, 160, 3), np.uint8)

    def close(self):
        pass


class TinyNetwork(torch.nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.linear = torch.nn.Linear(1, actions)

    def forward(self, states):
        return self.linear(states.float().mean(dim=(1, 2, 3))[:, None] / 255)


class TinyLearner:
    def __init__(self, actions):
        self.device = 'cpu'
        self.online = TinyNetwork(actions)
        self.target = copy.deepcopy(self.online)
        self.optimizer = NatureRMSprop(self.online.parameters())
        self.update_count = 0
        self.actions = actions

    def compute_targets(self, rewards, states, terminated):
        with torch.no_grad():
            return rewards + 0.99 * (~terminated).float() * self.target(states).max(1).values

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def update(self, batch):
        q = self.online(torch.as_tensor(batch.states)).gather(1, torch.as_tensor(batch.actions)[:, None]).squeeze(1)
        target = self.compute_targets(torch.as_tensor(batch.rewards), torch.as_tensor(batch.next_states), torch.as_tensor(batch.terminated))
        loss = torch.nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        return UpdateMetrics(loss.item(), q.mean().item(), target.mean().item(), (target-q).detach().abs().numpy())

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(online=self.online.state_dict(), target=self.target.state_dict(),
                        optimizer=self.optimizer.state_dict(), update_count=self.update_count,
                        action_count=self.actions, optimizer_name='nature_rmsprop', metadata=metadata or {}), path)

    def load(self, path, load_optimizer=True):
        value = torch.load(path, weights_only=False)
        self.online.load_state_dict(value['online'])
        self.target.load_state_dict(value['target'])
        if load_optimizer:
            self.optimizer.load_state_dict(value['optimizer'])
        self.update_count = value['update_count']
        return value['metadata']


class PongSessionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = dict(match_seconds=0.8, replay_capacity=32, min_replay=8, batch_size=4,
                           updates_per_match=2, target_sync=3, seed=7)
        self.sessions = []

    def tearDown(self):
        for session in self.sessions:
            session.close()
        self.temp.cleanup()

    def session(self, config=None, session_id=None):
        value = PongSession(self.root, config=self.config if config is None and session_id is None else config,
                            session_id=session_id, env_factory=FakeEnvironment, learner_factory=TinyLearner)
        self.sessions.append(value)
        return value

    def complete(self, session):
        session.start_match()
        session.set_keys({'up'})
        while session.status == 'playing':
            session.tick()

    def test_three_games_collect_then_learn_and_compare(self):
        session = self.session()
        before = copy.deepcopy(session.learner.online.state_dict())
        session.start_match()
        for _ in range(5):
            session.tick()
        self.assertEqual(len(session.replay), 0)  # only staging while playing
        self.assertEqual(session.learner.update_count, 0)
        self.assertTrue(all(torch.equal(before[key], value) for key, value in session.learner.online.state_dict().items()))
        while session.status == 'playing':
            session.tick()
        self.assertEqual(session.status, 'learning')
        self.assertIsNone(session.comparison)
        count = session.replay.count
        session.learn()
        self.assertEqual(session.active_version, 1)
        self.assertEqual(session.learner.update_count, 2)
        self.assertEqual(session.training['updates_completed'], 2)
        self.assertNotEqual(session.training['probe_before'], session.training['probe_after'])
        for expected_version in [2, 3]:
            self.complete(session)
            self.assertEqual(session.match['policy_version'], expected_version-1)
            session.learn()
            self.assertEqual(session.active_version, expected_version)
        self.assertEqual(session.committed_decisions, count*3)
        self.assertEqual(len(session.history()), 3)
        self.assertEqual(session.comparison['previous_policy_version'], 1)
        self.assertEqual(session.comparison['current_policy_version'], 2)
        self.assertEqual(session.learner.update_count, 6)
        # Snapshot payload is strictly serializable, including metrics and Q-values.
        import json
        json.dumps(session.snapshot(), allow_nan=False)

    def test_abort_does_not_commit_and_previous_valid_game_is_preserved(self):
        session = self.session()
        self.complete(session)
        session.learn()
        count = session.replay.count
        session.start_match()
        session.tick()
        session.abort_match('disconnect')
        self.assertEqual(session.replay.count, count)
        self.assertFalse(session.last_result['eligible'])
        self.complete(session)
        self.assertEqual(session.comparison['previous_match_id'], 'm000001')
        self.assertEqual(session.match['index'], 3)

    def test_cancel_keeps_active_model_then_retry_promotes_once(self):
        session = self.session()
        self.complete(session)
        original = copy.deepcopy(session.learner.online.state_dict())
        session.learn(cancelled=lambda: session.training['updates_completed'] >= 1)
        self.assertEqual(session.status, 'cancelled')
        self.assertEqual(session.active_version, 0)
        self.assertEqual(session.learner.update_count, 0)
        for key, value in original.items():
            torch.testing.assert_close(value, session.learner.online.state_dict()[key])
        replay_count = session.replay.count
        session.retry_learning()
        session.learn()
        self.assertEqual(session.active_version, 1)
        self.assertEqual(session.replay.count, replay_count)
        session.learn()
        self.assertEqual(session.active_version, 1)
        self.assertEqual(session.learner.update_count, 2)

    def test_failed_candidate_keeps_original_optimizer(self):
        session = self.session()
        self.complete(session)
        with patch.object(TinyLearner, 'update', side_effect=FloatingPointError('test update')):
            session.learn()
        self.assertEqual(session.status, 'failed')
        self.assertEqual(session.active_version, 0)
        self.assertEqual(session.learner.update_count, 0)
        self.assertEqual(len(session.learner.optimizer.state), 0)

    def test_restore_preserves_replay_optimizer_targets_rng_and_compacts_payloads(self):
        session = self.session()
        for _ in range(3):
            self.complete(session)
            session.learn()
        session.close()
        restored = self.session(config=None, session_id=session.id)
        self.assertEqual(restored.active_version, 3)
        self.assertEqual(restored.learner.update_count, 6)
        self.assertEqual(restored.replay.count, session.replay.count)
        np.testing.assert_array_equal(restored.replay.frames, session.replay.frames)
        self.assertEqual(restored.rng.bit_generator.state, session.rng.bit_generator.state)
        self.assertEqual(restored.replay.rng.bit_generator.state, session.replay.rng.bit_generator.state)
        for key, value in session.learner.target.state_dict().items():
            torch.testing.assert_close(value, restored.learner.target.state_dict()[key])
        self.assertEqual(len(list((session.directory/'replay').glob('snapshot-*.npz'))), 1)
        self.assertLessEqual(restored.replay.frames.shape[0], self.config['replay_capacity'])
        self.assertTrue(restored.learner.optimizer.state)

    def test_insufficient_replay_retains_version_and_records_zero_updates(self):
        session = self.session({**self.config, 'min_replay': 24})
        self.complete(session)
        self.assertEqual(session.status, 'insufficient_data')
        session.learn()
        self.assertEqual(session.active_version, 0)
        self.assertEqual(session.training['updates_completed'], 0)

    def test_time_cap_bootstraps_and_machine_actions_not_human_actions_are_stored(self):
        session = self.session()
        self.complete(session)
        self.assertTrue(session.replay_truncated[session.replay.count-1])
        self.assertFalse(session.replay.terminated[:session.replay.count].any())
        expected_actions = [machine for human, machine in session.env.actions]
        self.assertEqual(session.replay.actions[:session.replay.count].tolist(), expected_actions)
        self.assertNotEqual(expected_actions, [human for human, machine in session.env.actions])

    def test_invalid_checkpoint_rejected_before_import(self):
        checkpoint = self.root / 'legacy.pt'
        torch.save(dict(action_count=2, optimizer_name='nature_rmsprop', metadata={'environment_id': 'ALE/Pong-v5'}), checkpoint)
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            self.session({**self.config, 'checkpoint': str(checkpoint)})

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_config({'min_replay': 2})
        with self.assertRaises(ValueError):
            validate_config({'updates_per_match': 0})

    def test_network_setting_defaults_and_validates(self):
        self.assertEqual(validate_config()['network'], 'nature')
        self.assertEqual(validate_config({'network': 'dueling'})['network'], 'dueling')
        with self.assertRaises(ValueError):
            validate_config({'network': 'resnet'})


if __name__ == '__main__':
    unittest.main()
