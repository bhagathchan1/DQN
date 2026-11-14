"""Breakout in the DeepQ Learning Lab: you play a round, then a trained DQN plays the same seed.

Breakout is single-player, so the two players take turns on identical starting
conditions (same seed, same no-op start, same decision limit) and the scores are
compared side by side. Rounds reuse :class:`dqn_project.arena.ArenaRound`, run in
a background thread at the Atari pace of 15 decisions per second (four emulator
frames each), and are appended to ``experiments/breakout_arena/scores.csv``.
"""
from __future__ import annotations

import csv
import io
import re
import threading
import time
from pathlib import Path

ENV_ID = 'ALE/Breakout-v5'
DECISION_PERIOD = 4 / 60
MAX_DECISIONS = 4500          # 5 minutes of game time per round, for both players
AI_EPSILON = 0.05             # the evaluation epsilon used for the trained Breakout runs
INPUT_TIMEOUT = 1.0           # seconds without a key heartbeat before a human round stops
SPEEDS = (1, 2, 4)            # watch the DQN at 1x, 2x or 4x
MODELS_DIR = Path(__file__).resolve().parents[2] / 'trained_models' / 'breakout'

_DESCRIPTIONS = {
    'uniform': ('Uniform replay', 'Nature DQN, 200k uniform replay, RMSProp, 10M steps'),
    'prioritized': ('Prioritized replay', 'Nature DQN, 200k prioritized replay (α 0.6), 10M steps'),
    'upstream': ('Upstream reference setup', 'Reference-code settings: Adam 1e-4, 10k replay, 10M steps'),
}


def human_action(meanings, keys) -> int:
    """Map held keys to Breakout's actions (NOOP, FIRE, RIGHT, LEFT).

    Breakout has no move-and-fire action, so Space (launch the ball) takes priority
    while held; otherwise left or right moves the paddle.
    """
    left, right = 'left' in keys, 'right' in keys
    candidates = []
    if 'fire' in keys:
        candidates.append('FIRE')
    if left != right:
        candidates.append('LEFT' if left else 'RIGHT')
    candidates.append('NOOP')
    return next((meanings.index(name) for name in candidates if name in meanings), 0)


def discover_models(directory: Path | None = None) -> list[dict]:
    """Trained Breakout checkpoints; the browser selects them by ID, never by path."""
    directory = Path(directory or MODELS_DIR)
    models = []
    for path in sorted(directory.glob('*.pt')):
        kind = next((key for key in ('prioritized', 'upstream', 'uniform') if key in path.stem), 'uniform')
        title, detail = _DESCRIPTIONS[kind]
        models.append({'id': path.stem, 'kind': kind, 'label': f'{title} · trained DQN',
                       'detail': detail, 'path': str(path.resolve())})
    # Uniform replay had the best training average (46.6), so offer it first.
    order = {'uniform': 0, 'prioritized': 1, 'upstream': 2}
    return sorted(models, key=lambda model: order[model['kind']])


class BreakoutArena:
    """One Breakout round at a time, stepped by a background thread."""

    def __init__(self, results_path: Path, models_dir: Path | None = None, round_factory=None,
                 period: float = DECISION_PERIOD):
        self.results_path = Path(results_path)
        self.models_dir = models_dir
        self._round_factory = round_factory
        self.period = period
        self.lock = threading.Lock()
        self.round = None
        self.thread = None
        self.keys: set[str] = set()
        self.last_input = 0.0
        self.frame_bytes: bytes | None = None
        self.state = {'status': 'idle'}
        self.history = self._load_history()

    # ----- models and history -------------------------------------------------
    def models(self) -> list[dict]:
        return discover_models(self.models_dir)

    def _load_history(self) -> list[dict]:
        if not self.results_path.is_file():
            return []
        rows = []
        try:
            with self.results_path.open(newline='', encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    rows.append(self._clean_row(row))
        except (OSError, ValueError, csv.Error):
            return []
        return rows[-200:]

    @staticmethod
    def _clean_row(row: dict) -> dict:
        def number(value, cast=float):
            try:
                return cast(value)
            except (TypeError, ValueError):
                return None
        completed = row.get('completed')
        return {'timestamp_utc': row.get('timestamp_utc'), 'mode': row.get('mode'),
                'seed': number(row.get('seed'), int), 'score': number(row.get('score')),
                'decisions': number(row.get('decisions'), int), 'model': row.get('model') or Path(str(row.get('checkpoint') or '')).stem,
                'completed': completed in (True, 'True', 'true', '1'), 'reason': row.get('reason')}

    def pairs(self) -> list[dict]:
        """Latest completed human and DQN score per (seed, model), newest first."""
        human: dict[int, dict] = {}
        machine: dict[tuple[int, str], dict] = {}
        for row in self.history:
            if not row['completed'] or row['seed'] is None:
                continue
            if row['mode'] == 'human':
                human[row['seed']] = row
            elif row['mode'] == 'ai':
                machine[(row['seed'], row['model'])] = row
        result = []
        for (seed, model), ai_row in machine.items():
            if seed in human:
                you, dqn = human[seed]['score'], ai_row['score']
                result.append({'seed': seed, 'model': model, 'human': you, 'dqn': dqn,
                               'winner': 'draw' if you == dqn else 'you' if you > dqn else 'dqn',
                               'time': max(str(human[seed]['timestamp_utc']), str(ai_row['timestamp_utc']))})
        return sorted(result, key=lambda row: row['time'], reverse=True)

    # ----- control --------------------------------------------------------------
    def playing(self) -> bool:
        with self.lock:
            return self.state.get('status') == 'playing'

    def start(self, mode: str, seed: int, model_id: str | None = None, speed: int = 1) -> dict:
        if mode not in {'human', 'ai'}:
            raise ValueError('mode must be human or ai')
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError('seed must be an integer in [0, 2**31)')
        if speed not in SPEEDS:
            raise ValueError(f'speed must be one of {SPEEDS}')
        model = None
        if mode == 'ai':
            model = next((item for item in self.models() if item['id'] == model_id), None)
            if model is None:
                raise ValueError('Choose one of the trained Breakout models.')
        with self.lock:
            if self.state.get('status') == 'playing':
                raise ValueError('A Breakout round is already running.')
        factory = self._round_factory
        if factory is None:
            import torch
            from .arena import ArenaRound
            torch.set_num_threads(1)
            factory = ArenaRound
        # The DQN plays exactly the round a human would: same seed, no-ops and limit.
        round_ = factory(ENV_ID, mode, seed, model['path'] if model else None,
                         max_steps=MAX_DECISIONS, epsilon=AI_EPSILON)
        with self.lock:
            self.round = round_
            self.keys = set()
            self.last_input = time.monotonic()
            self.state = {'status': 'playing', 'mode': mode, 'seed': seed, 'speed': speed if mode == 'ai' else 1,
                          'model': model['id'] if model else None, 'score': 0.0, 'decisions': 0,
                          'lives': self._lives(round_), 'max_decisions': MAX_DECISIONS, 'started': time.time()}
            self._encode_frame(round_)
            self.thread = threading.Thread(target=self._run, args=(round_,), name='breakout-round', daemon=True)
            self.thread.start()
            return self.snapshot_locked()

    def set_keys(self, keys) -> None:
        with self.lock:
            if self.state.get('status') == 'playing' and self.state.get('mode') == 'human':
                self.keys = {key for key in keys if key in {'left', 'right', 'fire'}}
                self.last_input = time.monotonic()

    def abort(self, reason: str = 'aborted') -> dict:
        with self.lock:
            if self.state.get('status') == 'playing':
                self.state['stop_reason'] = reason
        if self.thread is not None:
            self.thread.join(timeout=2)
        with self.lock:
            return self.snapshot_locked()

    def close(self) -> None:
        self.abort('server_stopped')

    # ----- state ----------------------------------------------------------------
    def snapshot_locked(self) -> dict:
        state = dict(self.state)
        state['history'] = list(reversed(self.history[-30:]))
        state['pairs'] = self.pairs()[:30]
        return state

    def snapshot(self) -> dict:
        with self.lock:
            return self.snapshot_locked()

    @staticmethod
    def _lives(round_):
        lives = getattr(round_, 'lives', None)
        return lives() if callable(lives) else None

    def _encode_frame(self, round_) -> None:
        try:
            from PIL import Image
            stream = io.BytesIO()
            Image.fromarray(round_.frame()).save(stream, 'JPEG', quality=85)
            self.frame_bytes = stream.getvalue()
        except Exception:  # noqa: BLE001 - a missing frame must not stop the round
            pass

    def _run(self, round_) -> None:
        next_tick = time.monotonic()
        reason = None
        try:
            while True:
                with self.lock:
                    stop = self.state.get('stop_reason')
                    mode, speed, keys = self.state['mode'], self.state['speed'], set(self.keys)
                    idle = time.monotonic() - self.last_input
                if stop:
                    reason = stop
                    break
                if mode == 'human' and idle > INPUT_TIMEOUT:
                    reason = 'input_timeout'
                    break
                action = human_action(round_.meanings, keys) if mode == 'human' else 0
                round_.step(action)
                with self.lock:
                    self.state.update(score=float(round_.score), decisions=int(round_.steps),
                                      lives=self._lives(round_))
                    self._encode_frame(round_)
                if round_.done:
                    break
                next_tick += self.period / speed
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -self.period:
                    next_tick = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            reason = f'error: {type(exc).__name__}: {exc}'
        result = round_.result()
        if reason:
            result.update(completed=False, reason=reason)
        model = self.state.get('model') or ''
        result['model'] = model
        try:
            round_.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            from .arena import append_result
            append_result(self.results_path, result)
        except Exception:  # noqa: BLE001 - keep the in-memory record even if disk fails
            pass
        with self.lock:
            self.history.append(self._clean_row({k: str(v) for k, v in result.items()}))
            self.state.update(status='finished', completed=bool(result['completed']),
                              reason=result['reason'], score=float(result['score']),
                              decisions=int(result['decisions']))
            self.state.pop('stop_reason', None)
            self.round = None


def safe_model_id(value) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r'[A-Za-z0-9_.-]{1,120}', value))
