"""Durable Pong sessions: collect during play, learn between complete matches.

The API controller owns the clock and input heartbeat. This module deliberately
contains no web or UI code, so the same state machine can be integration tested.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .frame_replay import FrameReplayBuffer
from .learner import DQNLearner


DEFAULT_CONFIG = {
    'match_seconds': 120.0, 'replay_capacity': 20000, 'min_replay': 256,
    'batch_size': 32, 'updates_per_match': 450, 'gamma': 0.99,
    'learning_rate': 0.00025, 'target_sync': 1000,
    'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decisions': 10000,
    'seed': 7, 'device': 'auto', 'network': 'nature',
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(_clean(data), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance():
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in ['torch', 'numpy', 'Pillow', 'pettingzoo', 'ale-py', 'multi-agent-ale-py']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    repository = Path(__file__).resolve().parents[2]
    git = ['git', '-c', f'safe.directory={repository.as_posix()}']
    try:
        revision = subprocess.check_output([*git, 'rev-parse', 'HEAD'], cwd=repository,
                                           text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output([*git, 'status', '--porcelain'], cwd=repository,
                                             text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = 'unknown', None
    return dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                packages=packages, git_revision=revision, git_dirty=dirty,
                cuda_available=torch.cuda.is_available(), recorded_utc=_utc())


def validate_config(config: dict | None = None) -> dict:
    supplied = dict(config or {})
    if 'updates_per_game' in supplied:
        supplied.setdefault('updates_per_match', supplied.pop('updates_per_game'))
    unknown = set(supplied) - set(DEFAULT_CONFIG) - {'checkpoint'}
    if unknown:
        raise ValueError(f'Unknown session settings: {sorted(unknown)}')
    result = {**DEFAULT_CONFIG, **supplied}
    for key in ['replay_capacity', 'min_replay', 'batch_size', 'updates_per_match', 'target_sync', 'epsilon_decisions']:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 8 <= result['replay_capacity'] <= 100000:
        raise ValueError('replay_capacity must be between 8 and 100000')
    if not result['batch_size'] <= result['min_replay'] <= result['replay_capacity'] - 4:
        raise ValueError('Require batch_size <= min_replay <= replay_capacity - 4')
    if result['updates_per_match'] > 2048 or result['batch_size'] > 256:
        raise ValueError('Demo updates must be <= 2048 and batch_size <= 256')
    for key in ['match_seconds', 'gamma', 'learning_rate', 'epsilon_start', 'epsilon_end']:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
    if not 0 < result['match_seconds'] <= 600 or result['learning_rate'] <= 0:
        raise ValueError('Require 0 < match_seconds <= 600 and positive learning_rate')
    if not 0 <= result['gamma'] <= 1 or not 0 <= result['epsilon_end'] <= result['epsilon_start'] <= 1:
        raise ValueError('Gamma and epsilon settings must be within [0,1]')
    if isinstance(result['seed'], bool) or not isinstance(result['seed'], int) or not 0 <= result['seed'] < 2**31:
        raise ValueError('seed must be an integer in [0,2**31)')
    if result['network'] not in {'nature', 'dueling'}:
        raise ValueError('network must be nature or dueling')
    if result['device'] not in {'cpu', 'cuda', 'auto'}:
        raise ValueError('device must be cpu, cuda or auto')
    requested = result['device']
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but this PyTorch installation has no available CUDA device.')
    result['device'] = 'cuda' if requested == 'auto' and torch.cuda.is_available() else 'cpu' if requested == 'auto' else requested
    return result


class PongSession:
    """One machine learner and its bounded replay, retained across human games."""

    def __init__(self, root: Path, config: dict | None = None, session_id: str | None = None,
                 *, env_factory=None, learner_factory=None):
        self.id = session_id or uuid.uuid4().hex[:16]
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', self.id):
            raise ValueError('Invalid session ID')
        self.directory = Path(root).resolve() / self.id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / 'session.json'
        saved = json.loads(self.manifest_path.read_text(encoding='utf-8')) if self.manifest_path.is_file() else None
        if saved and config:
            raise ValueError('Restore uses its saved configuration; create a new session to change settings.')
        if saved:
            # The device is where computation runs, not part of the learning protocol:
            # a restored session uses the GPU when one is available now.
            restored = {**saved['config'], 'device': 'auto'}
            self.config = validate_config(restored)
        else:
            self.config = validate_config(config)
        self._learner_factory = learner_factory
        if env_factory is None:
            from .pong_environment import PongEnvironment
            env_factory = PongEnvironment
        self.env = env_factory()
        self.spec = _clean(self.env.spec)
        self._keys: set[str] = set()
        self._state = None
        self.latest_state = None
        self.action_qs = []
        self._last_frame = None
        self._stage = []
        self._initial_state = None
        self._cancel_requested = False
        self._closed = False
        self._events_seq = 0
        self.error = None
        self.match = None
        self.training = None
        self.last_result = None
        self.comparison = None
        self._history = []
        self._committed_matches = set()
        self.committed_decisions = 0
        self.active_version = 0
        self.status = 'ready'
        self.origin = 'from_scratch'
        self._checkpoint = None
        self._checkpoint_hash = None
        self._match_started_perf = None
        self._replay_file = None
        self._replay_revision = 0
        self._phase_rng = None
        self.created_at = saved.get('created_at', _utc()) if saved else _utc()
        self.provenance = saved.get('provenance', _provenance()) if saved else _provenance()
        random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        self.rng = np.random.default_rng(self.config['seed'])
        self.replay = FrameReplayBuffer(self.config['replay_capacity'], seed=self.config['seed'])
        self.replay_match_indices = np.zeros(self.config['replay_capacity'], dtype=np.int64)
        self.replay_policy_versions = np.zeros(self.config['replay_capacity'], dtype=np.int64)
        self.replay_raw_rewards = np.zeros(self.config['replay_capacity'], dtype=np.float32)
        self.replay_truncated = np.zeros(self.config['replay_capacity'], dtype=np.bool_)
        try:
            self.learner = self._make_learner()
            if saved:
                self._restore(saved)
            else:
                if self.config.get('checkpoint'):
                    self._import(self.config['checkpoint'])
                self._checkpoint = self._save_learner(self.learner, self.active_version)
                self._checkpoint_hash = _sha(self.directory / self._checkpoint)
                self._save_replay()
                self._persist()
                _json(self.directory / 'config.json', self.config)
                self._event('session_created', {'origin': self.origin})
            self._state = self.env.reset(seed=self.config['seed'])
            self.latest_state = self._state.copy()
            self._last_frame = self.env.render()
        except BaseException:
            self.env.close()
            raise

    def _make_learner(self):
        if self._learner_factory:
            return self._learner_factory(self.env.action_count)
        return DQNLearner(self.env.action_count, gamma=self.config['gamma'],
                          learning_rate=self.config['learning_rate'], device=self.config['device'],
                          optimizer='nature_rmsprop', gradient_clip=None,
                          architecture=self.config['network'])

    def _import(self, path):
        payload = torch.load(Path(path), map_location='cpu', weights_only=False)
        metadata = payload.get('metadata', {})
        if metadata.get('environment_spec') != self.spec or payload.get('action_count') != self.env.action_count:
            raise ValueError('Checkpoint is incompatible: two-player environment, side, action map and preprocessing must match.')
        if payload.get('architecture', 'nature') != self.config['network']:
            raise ValueError(f"Checkpoint uses the {payload.get('architecture', 'nature')} network; "
                             f"this session uses {self.config['network']}.")
        if payload.get('optimizer_name') != 'nature_rmsprop':
            raise ValueError('Only compatible Nature RMSProp Pong session checkpoints can be imported.')
        source = metadata.get('config', {})
        for name in ['gamma', 'learning_rate', 'target_sync']:
            if name in source and source[name] != self.config[name]:
                raise ValueError(f'Imported checkpoint {name} differs from the selected session configuration.')
        if payload.get('gamma', self.config['gamma']) != self.config['gamma'] or payload.get('optimizer_epsilon', 0.01) != 0.01:
            raise ValueError('Imported checkpoint optimizer/discount settings are incompatible.')
        self.learner.load(path, load_optimizer=True)
        self.origin = 'imported_checkpoint'

    def _event(self, kind, data=None):
        self._events_seq += 1
        record = {'seq': self._events_seq, 'time': _utc(), 'type': kind, 'data': _clean(data or {})}
        with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    def _save_learner(self, learner, version):
        relative = f'checkpoints/v{version}/learner.pt'
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        learner.save(temporary, metadata={'session_id': self.id, 'version': version,
                                         'environment_spec': self.spec, 'config': self.config,
                                         'origin': self.origin, 'provenance': self.provenance,
                                         'saved_utc': _utc()})
        os.replace(temporary, path)
        return relative

    def _save_replay(self):
        """Save only populated slots, including boundary frames and provenance."""
        self._replay_revision += 1
        relative = f'replay/snapshot-{self._replay_revision}.npz'
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        size = min(self.replay.count, self.replay.capacity)
        starts = sorted(self.replay.starts)
        with path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, frames=self.replay.frames[:size], actions=self.replay.actions[:size],
                                rewards=self.replay.rewards[:size], terminated=self.replay.terminated[:size],
                                starts=np.asarray(starts, dtype=np.int64),
                                start_states=np.stack([self.replay.starts[i] for i in starts]) if starts else np.empty((0, 4, 84, 84), np.uint8),
                                match_indices=self.replay_match_indices[:size],
                                policy_versions=self.replay_policy_versions[:size],
                                raw_rewards=self.replay_raw_rewards[:size], truncated=self.replay_truncated[:size])
        os.replace(path.with_suffix('.tmp'), path)
        self._replay_file = relative

    def _manifest(self):
        return dict(schema_version=1, id=self.id, created_at=self.created_at, saved_at=_utc(),
                    config=self.config, environment_spec=self.spec, provenance=self.provenance,
                    origin=self.origin, status=self.status,
                    active_version=self.active_version, checkpoint=self._checkpoint,
                    checkpoint_hash=self._checkpoint_hash,
                    replay_file=self._replay_file, replay_revision=self._replay_revision,
                    replay_count=self.replay.count, replay_pending_start=self.replay.pending_start,
                    replay_rng=self.replay.rng.bit_generator.state, rng=self.rng.bit_generator.state,
                    torch_rng=torch.get_rng_state().tolist(),
                    committed_decisions=self.committed_decisions,
                    committed_matches=sorted(self._committed_matches),
                    history=self._history, match=self.match, training=self.training,
                    last_result=self.last_result, comparison=self.comparison,
                    events_seq=self._events_seq, error=self.error)

    def _persist(self):
        _json(self.manifest_path, self._manifest())
        # The manifest is replaced before obsolete replay snapshots are removed.
        if self._replay_file:
            keep = self.directory / self._replay_file
            for previous in (self.directory / 'replay').glob('snapshot-*.npz'):
                if previous != keep:
                    previous.unlink()

    def _restore(self, saved):
        if saved['environment_spec'] != self.spec:
            raise ValueError('Saved session environment differs from the installed two-player Pong adapter.')
        self.active_version = saved['active_version']
        self._checkpoint = saved['checkpoint']
        self._checkpoint_hash = saved.get('checkpoint_hash') or _sha(self.directory / self._checkpoint)
        self.learner.load(self.directory / self._checkpoint)
        self._replay_file = saved['replay_file']
        self._replay_revision = saved['replay_revision']
        with np.load(self.directory / self._replay_file, allow_pickle=False) as archive:
            size = len(archive['actions'])
            for name in ['frames', 'actions', 'rewards', 'terminated']:
                getattr(self.replay, name)[:size] = archive[name]
            self.replay.starts = {int(i): array.copy() for i, array in zip(archive['starts'], archive['start_states'])}
            for name, key in [('replay_match_indices', 'match_indices'), ('replay_policy_versions', 'policy_versions'),
                              ('replay_raw_rewards', 'raw_rewards'), ('replay_truncated', 'truncated')]:
                getattr(self, name)[:size] = archive[key]
        self.replay.count = saved['replay_count']
        self.replay.pending_start = saved['replay_pending_start']
        self.replay.rng.bit_generator.state = saved['replay_rng']
        self.rng.bit_generator.state = saved['rng']
        torch.set_rng_state(torch.tensor(saved['torch_rng'], dtype=torch.uint8))
        self.origin = saved.get('origin', 'restored_session')
        self.committed_decisions = saved['committed_decisions']
        self._committed_matches = set(saved['committed_matches'])
        self._history = saved['history']
        self.match, self.training = saved.get('match'), saved.get('training')
        self.last_result, self.comparison = saved.get('last_result'), saved.get('comparison')
        self._events_seq = saved.get('events_seq', 0)
        events_path = self.directory / 'events.jsonl'
        if events_path.is_file():
            # A crash between appending an event and replacing the manifest must
            # not reuse sequence numbers when an SSE client reconnects.
            with events_path.open(encoding='utf-8') as stream:
                for line in stream:
                    try:
                        self._events_seq = max(self._events_seq, int(json.loads(line)['seq']))
                    except (ValueError, KeyError):
                        pass  # ignore only an incomplete event line
        self.status, self.error = saved['status'], saved.get('error')
        if self.status == 'playing':
            self.status = 'interrupted'
            if self.match:
                result = {**self.match, 'status': 'aborted', 'valid': False, 'eligible': False,
                          'reason': 'server_restart', 'completed_at': _utc()}
                if not any(item['id'] == result['id'] for item in self._history):
                    self._history.append(result)
                self.last_result = result
            self.error = 'Previous game was interrupted; no staged experience was added.'
        elif self.status == 'learning':
            self.status = 'interrupted'
            self.training['status'] = 'interrupted'
            self.training['error'] = 'Learning was interrupted before model promotion. Retry from the active checkpoint.'
            self.error = self.training['error']
        self._persist()
        self._event('session_restored', {'status': self.status})

    def snapshot(self):
        return _clean(dict(id=self.id, status=self.status, active_version=self.active_version,
                           model_version=f'v{self.active_version}', origin=self.origin, config=self.config,
                           checkpoint_hash=self._checkpoint_hash,
                           environment_spec=self.spec, replay_size=len(self.replay),
                           committed_decisions=self.committed_decisions,
                           cumulative_updates=self.learner.update_count,
                           match=self.match, training=self.training, last_result=self.last_result,
                           comparison=self.comparison, history=self._history, error=self.error,
                           events_seq=self._events_seq, action_qs=self.action_qs))

    def history(self):
        return copy.deepcopy(self._history)

    def frame(self):
        return self._last_frame

    def set_keys(self, keys):
        self._keys = set(keys) & {'up', 'down', 'fire'}

    def start_match(self):
        if self.status in {'playing', 'learning'}:
            raise ValueError('Finish the current game or learning phase first.')
        index = len(self._history) + 1
        epsilon = max(self.config['epsilon_end'], self.config['epsilon_start'] -
                      (self.config['epsilon_start'] - self.config['epsilon_end']) *
                      self.committed_decisions / self.config['epsilon_decisions'])
        self._state = self.env.reset(seed=self.config['seed'] + index)
        self.latest_state = self._state.copy()
        self._initial_state = self._state.copy()
        self._last_frame = self.env.render()
        self._match_started_perf = time.perf_counter()
        self._stage = []
        self._keys.clear()
        self.error = None
        self.action_qs = []
        self.match = dict(id=f'm{index:06d}', index=index, policy_version=self.active_version,
                          checkpoint_hash=self._checkpoint_hash, origin=self.origin,
                          epsilon=float(epsilon), started_at=_utc(), seed=self.config['seed'] + index,
                          elapsed_seconds=0.0, remaining_seconds=self.config['match_seconds'],
                          human_return=0.0, machine_return=0.0, decisions=0, emulator_frames=0,
                          human_goals=None, machine_goals=None, human_penalties=None,
                          machine_penalties=None, score_mode='return', status='playing',
                          wall_time_seconds=0.0, decision_fps=0.0, emulator_fps=0.0)
        self.status = 'playing'
        self._event('match_started', {'match_id': self.match['id'], 'policy_version': self.active_version})
        self._persist()

    def tick(self):
        if self.status != 'playing':
            return False
        with torch.no_grad():
            tensor = torch.as_tensor(self._state, device=self.learner.device).unsqueeze(0)
            values = self.learner.online(tensor).squeeze(0).detach().cpu().numpy()
        self.action_qs = values.tolist()
        machine_action = int(self.rng.integers(self.env.action_count)) if self.rng.random() < self.match['epsilon'] else int(values.argmax())
        human_action = self.env.action_for_keys(self._keys)
        successor, reward, terminated, truncated, info = self.env.step(human_action, machine_action)
        frames = int(info.get('emulator_frames', 4))
        if frames <= 0:
            raise RuntimeError('Atari adapter returned a nonpositive emulator frame count')
        self.match['decisions'] += 1
        self.match['emulator_frames'] += frames
        self.match['elapsed_seconds'] = self.match['emulator_frames'] / 60
        self.match['remaining_seconds'] = max(0.0, self.config['match_seconds'] - self.match['elapsed_seconds'])
        wall = max(time.perf_counter() - self._match_started_perf, 1e-9)
        self.match['wall_time_seconds'] = wall
        self.match['decision_fps'] = self.match['decisions'] / wall
        self.match['emulator_fps'] = self.match['emulator_frames'] / wall
        self.match['human_return'] += float(info['human_reward'])
        self.match['machine_return'] += float(reward)
        self.match['score_mode'] = info.get('score_mode', 'return')
        for name in ['human_goals', 'machine_goals', 'human_penalties', 'machine_penalties']:
            if name in info:
                self.match[name] = info[name]
        capped = self.match['remaining_seconds'] <= 0
        # The staging buffer keeps only successors' newest frame, not two full stacks.
        self._stage.append(dict(frame=np.asarray(successor[-1], np.uint8).copy(), action=machine_action,
                                human_action=human_action, raw_reward=float(reward),
                                reward=float(np.sign(reward)), terminated=bool(terminated),
                                truncated=bool(truncated or (capped and not terminated))))
        self._state = successor
        self.latest_state = successor.copy()
        self._last_frame = self.env.render()
        self._events_seq += 1
        if terminated or truncated or capped:
            self._finish_match('terminated' if terminated else 'time_limit')
        return True

    def _finish_match(self, reason):
        if self.status != 'playing':
            return
        result = copy.deepcopy(self.match)
        result.update(status='completed', valid=True, eligible=True, reason=reason, completed_at=_utc())
        human, machine = result['human_return'], result['machine_return']
        result['outcome'] = 'machine' if machine > human else 'human' if human > machine else 'draw'
        result['outcome_label'] = 'return_leader' if result['score_mode'] == 'return' else 'winner'
        result['return_margin'] = machine - human
        previous = next((item for item in reversed(self._history) if item.get('valid')), None)
        self.comparison = None if previous is None else dict(
            previous_match_id=previous['id'], previous_index=previous['index'],
            previous_policy_version=previous['policy_version'], current_policy_version=result['policy_version'],
            machine_return_delta=machine - previous['machine_return'],
            human_return_delta=human - previous['human_return'],
            return_margin_delta=result['return_margin'] - previous['return_margin'],
            duration_delta=result['elapsed_seconds'] - previous['elapsed_seconds'],
            decision_fps_delta=result['decision_fps'] - previous.get('decision_fps', 0))
        self._history.append(result)
        self.last_result = result
        self.match = copy.deepcopy(result)
        if result['id'] not in self._committed_matches:
            self.replay.start_episode()
            state = self._initial_state
            for item in self._stage:
                successor = np.concatenate([state[1:], item['frame'][None]], axis=0)
                slot = self.replay.add(state, item['action'], item['reward'], successor, item['terminated'])
                self.replay_match_indices[slot] = result['index']
                self.replay_policy_versions[slot] = result['policy_version']
                self.replay_raw_rewards[slot] = item['raw_reward']
                self.replay_truncated[slot] = item['truncated']
                state = successor
            self.committed_decisions += len(self._stage)
            self._committed_matches.add(result['id'])
            self._save_replay()
        self._stage = []
        enough = len(self.replay) >= self.config['min_replay']
        self.training = dict(id=f'train-{result["id"]}', match_id=result['id'],
                             input_version=self.active_version, output_version=None,
                             updates_completed=0, total_updates=self.config['updates_per_match'],
                             metrics=[], probe_before=None, probe_after=None, elapsed_seconds=0.0,
                             status='queued' if enough else 'insufficient_data', error=None)
        self.status = 'learning' if enough else 'insufficient_data'
        self._record_training()
        self._event('match_completed', {'match_id': result['id'], 'replay_size': len(self.replay), 'status': self.status})
        self._persist()
        self._write_result(result)

    def abort_match(self, reason='aborted'):
        if self.status != 'playing':
            return
        result = copy.deepcopy(self.match)
        invalid = reason in {'invalid', 'slow_simulation', 'low_throughput', 'invalid_throughput'}
        result.update(status='invalid' if invalid else 'aborted', valid=False, eligible=False,
                      reason=str(reason), completed_at=_utc(), outcome=None)
        self._stage = []
        self._keys.clear()
        self.match = result
        self._history.append(result)
        self.last_result = result
        self.comparison = None
        self.status = 'ready'
        self._event('match_aborted', {'match_id': result['id'], 'reason': reason})
        self._persist()
        self._write_result(result)

    def _write_result(self, result):
        _json(self.directory / 'matches' / result['id'] / 'result.json', result)
        # Derived export: no transition payload is duplicated in old match folders.
        import csv
        with (self.directory / 'scores.csv').open('w', newline='', encoding='utf-8') as stream:
            fields = ['id', 'index', 'policy_version', 'status', 'human_return', 'machine_return',
                      'elapsed_seconds', 'epsilon', 'reason']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self._history)

    def _record_training(self):
        if self.training:
            for item in self._history:
                if item['id'] == self.training['match_id']:
                    item['training'] = {k: v for k, v in self.training.items() if k != 'metrics'}
                    if self.last_result and self.last_result['id'] == item['id']:
                        self.last_result['training'] = copy.deepcopy(item['training'])
                    if self.match and self.match['id'] == item['id']:
                        self.match['training'] = copy.deepcopy(item['training'])

    def _probe_batch(self):
        old = self.replay.rng
        try:
            index = self.last_result['index']
            self.replay.rng = np.random.default_rng(self.config['seed'] + 100000 + index)
            return self.replay.sample(self.config['batch_size'])
        finally:
            self.replay.rng = old

    @staticmethod
    def _diagnostic(learner, batch, targets):
        with torch.no_grad():
            states = torch.as_tensor(batch.states, device=learner.device)
            actions = torch.as_tensor(batch.actions, device=learner.device)
            q = learner.online(states).gather(1, actions[:, None]).squeeze(1)
            td = targets.to(learner.device) - q
            return dict(mean_q=float(q.mean().item()), td_error=float(td.abs().mean().item()),
                        loss=float(torch.nn.functional.smooth_l1_loss(q, targets.to(learner.device)).item()))

    def learn(self, cancelled: Callable[[], bool] = lambda: False, on_update: Callable | None = None):
        if self.status != 'learning' or not self.training or self.training['status'] == 'completed':
            return
        phase = self.training
        self._cancel_requested = False
        phase['status'] = 'running'
        self.error = None
        self._phase_rng = copy.deepcopy(self.replay.rng.bit_generator.state)
        self._event('learning_started', {'phase_id': phase['id'], 'input_version': self.active_version})
        self._persist()
        started = time.perf_counter()
        candidate = None
        old_version, old_checkpoint, old_learner = self.active_version, self._checkpoint, self.learner
        old_hash = self._checkpoint_hash
        committed = False
        try:
            candidate = copy.deepcopy(self.learner)
            batch = self._probe_batch()
            targets = self.learner.compute_targets(
                torch.as_tensor(batch.rewards, device=self.learner.device),
                torch.as_tensor(batch.next_states, device=self.learner.device),
                torch.as_tensor(batch.terminated, device=self.learner.device)).detach().clone()
            phase['probe_before'] = self._diagnostic(candidate, batch, targets)
            directory = self.directory / 'training' / phase['id']
            directory.mkdir(parents=True, exist_ok=True)
            # Only current probe tensors are retained; summaries and hashes remain durable.
            probe_path = self.directory / 'training' / 'latest_probe.npz'
            np.savez_compressed(probe_path, states=batch.states, actions=batch.actions, targets=targets.cpu().numpy())
            phase['probe_hash'] = _sha(probe_path)
            with (directory / 'metrics.jsonl').open('w', encoding='utf-8') as metrics_file:
                for index in range(self.config['updates_per_match']):
                    if cancelled() or self._cancel_requested:
                        raise InterruptedError('Learning cancelled; active model was retained.')
                    sampled = self.replay.sample(self.config['batch_size'])
                    result = candidate.update(sampled)
                    synced = candidate.update_count % self.config['target_sync'] == 0
                    if synced:
                        candidate.sync_target()
                    metric = dict(update=index + 1, cumulative_update=candidate.update_count,
                                  loss=result.loss, td_error=float(np.mean(result.td_errors)),
                                  mean_q=result.mean_q, target_synced=synced)
                    phase['metrics'].append(metric)
                    phase['updates_completed'] = index + 1
                    phase['elapsed_seconds'] = time.perf_counter() - started
                    metrics_file.write(json.dumps(_clean(metric), allow_nan=False) + '\n')
                    metrics_file.flush()
                    self._event('optimizer_update', metric)
                    if on_update:
                        on_update()
            if cancelled() or self._cancel_requested:
                raise InterruptedError('Learning cancelled before promotion; active model was retained.')
            phase['probe_after'] = self._diagnostic(candidate, batch, targets)
            new_version = old_version + 1
            checkpoint = self._save_learner(candidate, new_version)
            phase.update(status='completed', output_version=new_version,
                         checkpoint=checkpoint, checkpoint_hash=_sha(self.directory / checkpoint),
                         elapsed_seconds=time.perf_counter() - started)
            self.learner = candidate
            self.active_version = new_version
            self._checkpoint = checkpoint
            self._checkpoint_hash = phase['checkpoint_hash']
            self.status = 'summary'
            self._record_training()
            # Stage ancillary exports first; session.json is the authoritative
            # commit point and must never promote a partly saved candidate.
            _json(directory / 'summary.json', phase)
            self._write_result(self.last_result)
            self._event('candidate_saved', {'version': new_version, 'phase_id': phase['id']})
            self._persist()
            committed = True
            try:
                self._event('model_promoted', {'version': new_version, 'phase_id': phase['id']})
            except OSError:
                # The checkpoint and authoritative manifest are already durable;
                # a failed auxiliary event append must not roll them back.
                pass
        except BaseException as exc:
            if committed:
                raise
            self.learner, self.active_version, self._checkpoint = old_learner, old_version, old_checkpoint
            self._checkpoint_hash = old_hash
            self.replay.rng.bit_generator.state = self._phase_rng
            self.status = 'cancelled' if isinstance(exc, InterruptedError) else 'failed'
            phase.update(status=self.status, output_version=None, error=f'{type(exc).__name__}: {exc}',
                         elapsed_seconds=time.perf_counter() - started)
            self.error = phase['error']
            self._record_training()
            self._event('learning_' + self.status, {'phase_id': phase['id'], 'error': self.error})
            self._persist()
            _json(self.directory / 'training' / phase['id'] / 'summary.json', phase)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            if on_update:
                on_update()

    def cancel_learning(self):
        self._cancel_requested = True

    def retry_learning(self):
        if self.status not in {'failed', 'cancelled', 'interrupted'} or not self.training:
            raise ValueError('There is no uncommitted learning phase to retry.')
        if self.training['status'] not in {'failed', 'cancelled', 'interrupted'}:
            raise ValueError('The previous phase is already committed; start a new game instead.')
        if self.training['input_version'] != self.active_version:
            raise ValueError('The active model has changed; start a new game instead.')
        self.training.update(status='queued', error=None, updates_completed=0, metrics=[],
                             probe_before=None, probe_after=None, elapsed_seconds=0.0)
        self.error = None
        self.status = 'learning'
        self._event('learning_retry', {'phase_id': self.training['id']})
        self._persist()

    def close(self):
        if self._closed:
            return
        if self.status == 'playing':
            self.abort_match('session_closed')
        self._persist()
        self.env.close()
        self.replay.close()
        self._closed = True
