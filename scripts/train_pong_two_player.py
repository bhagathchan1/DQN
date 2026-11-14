"""Pre-train a Pong Learning Lab model in the real two-player Pong environment.

The DQN plays the lab's machine side (PettingZoo ``second_0``) against a scripted
opponent on the human side (``first_0``) that presses the same key-mapped actions a
human would, with its skill (reaction dead-zone, hesitation, aim noise) re-drawn
every game so the learner sees many different "players".

Training can start from a single-player ``ALE/Pong-v5`` checkpoint (see
``configs/pong_nature.yaml``). In this two-player runtime ``second_0`` moves the
orange paddle on the LEFT, whereas single-player Pong trains the green paddle on
the RIGHT, so the single-player weights are mirrored left-to-right first. The
Nature network's convolutions tile 84 pixels exactly (84->20->9->7), so mirroring
the kernels and the flattened 7x7 feature map is an exact equivalence; this is
checked numerically at start-up.

The output checkpoints use the lab's own learner settings (Nature RMSProp, no
gradient clipping, gamma 0.99, learning rate 0.00025, target sync every 1000
updates) and record the installed two-player environment spec, so the lab's
"import checkpoint" check accepts them.

Example:
  python scripts/train_pong_two_player.py --network nature \
      --init runs/pong_nature/checkpoints/pong_nature_seed7_best.pt \
      --output runs/pong_lab_nature --steps 3000000
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from dqn_project.frame_replay import FrameReplayBuffer
from dqn_project.learner import DQNLearner
from dqn_project.pong_environment import PongEnvironment
from dqn_project.pong_session import _clean, _provenance, _utc, validate_config

FIELD_TOP, FIELD_BOTTOM = 34, 194
HUMAN_PADDLE_RGB = (92, 186, 92)   # first_0 (lab human): green paddle, right side
BALL_RGB = (236, 236, 236)
DECISIONS_PER_GAME = 1800          # the lab's default 120-second match at 15 decisions/s


def mirror_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Weights for the same network applied to horizontally flipped input frames."""
    mirrored = {}
    for name, tensor in state.items():
        tensor = tensor.clone()
        if tensor.ndim == 4:                                   # conv kernels [out,in,h,w]
            tensor = tensor.flip(-1)
        elif tensor.ndim == 2 and tensor.shape[1] == 64 * 7 * 7:  # first linear layer(s)
            tensor = tensor.reshape(tensor.shape[0], 64, 7, 7).flip(-1).reshape(tensor.shape[0], -1)
        mirrored[name] = tensor
    return mirrored


def _locate(frame: np.ndarray, rgb) -> float | None:
    mask = np.all(frame[FIELD_TOP:FIELD_BOTTOM] == rgb, axis=-1)
    rows = np.nonzero(mask)[0]
    return float(rows.mean()) + FIELD_TOP if len(rows) else None


class ScriptedOpponent:
    """Ball-tracking paddle on the human side, driven through the lab's key mapping."""

    def __init__(self, env: PongEnvironment, rng: np.random.Generator, skill: str = 'random'):
        self.env, self.rng = env, rng
        self.actions = {name: env.action_for_keys(keys) for name, keys in
                        [('up', {'up'}), ('down', {'down'}), ('fire', {'fire'}), ('noop', set())]}
        self.new_game(skill)

    def new_game(self, skill: str = 'random') -> None:
        if skill == 'hard':
            self.deadzone, self.hesitate, self.aim, self.random_rate = 2.0, 0.0, 0.0, 0.0
        else:
            self.deadzone = float(self.rng.uniform(1.0, 10.0))
            self.hesitate = float(self.rng.uniform(0.0, 0.35))
            self.aim = float(self.rng.normal(0.0, 6.0))
            self.random_rate = float(self.rng.uniform(0.0, 0.15))

    def act(self, frame: np.ndarray) -> int:
        if self.rng.random() < self.random_rate:
            return int(self.rng.choice(list(self.actions.values())))
        ball = _locate(frame, BALL_RGB)
        paddle = _locate(frame, HUMAN_PADDLE_RGB)
        if ball is None:                           # between points: serve
            return self.actions['fire']
        if paddle is None or self.rng.random() < self.hesitate:
            return self.actions['noop']
        offset = ball + self.aim - paddle
        if abs(offset) <= self.deadzone:
            return self.actions['noop']
        return self.actions['down'] if offset > 0 else self.actions['up']


def play_game(env, opponent, learner, rng, epsilon, seed, on_step=None):
    """One lab-length game. Returns the machine's return (its points minus the human's)."""
    state = env.reset(seed=seed)
    total = 0.0
    for decision in range(DECISIONS_PER_GAME):
        if rng.random() < epsilon:
            action = int(rng.integers(env.action_count))
        else:
            with torch.no_grad():
                q = learner.online(torch.as_tensor(state, device=learner.device)[None])
            action = int(q.argmax(1).item())
        successor, reward, terminated, truncated, _ = env.step(opponent.act(env.render()), action)
        total += reward
        if on_step:
            on_step(state, action, reward, successor, terminated)
        state = successor
        if terminated or truncated:
            break
    return total, decision + 1, bool(terminated or truncated)


def evaluate(learner, rng_seed, games):
    env = PongEnvironment()
    rng = np.random.default_rng(rng_seed)
    opponent = ScriptedOpponent(env, rng, skill='hard')
    try:
        returns = [play_game(env, opponent, learner, rng, 0.0, rng_seed + i)[0] for i in range(games)]
    finally:
        env.close()
    return float(np.mean(returns)), returns


def save(learner, path: Path, env_spec: dict, lab_config: dict, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    learner.save(temporary, metadata={
        'session_id': 'two_player_pretraining', 'version': 0,
        'environment_spec': env_spec, 'config': lab_config,
        'origin': 'two_player_pretraining', 'provenance': _provenance(),
        'saved_utc': _utc(), **extra})
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--network', choices=['nature', 'dueling'], default='nature')
    parser.add_argument('--init', help='single-player ALE/Pong-v5 checkpoint to mirror and start from')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=3_000_000, help='machine decisions to train for')
    parser.add_argument('--replay', type=int, default=500_000)
    parser.add_argument('--learning-starts', type=int, default=20_000)
    parser.add_argument('--epsilon-start', type=float, default=None,
                        help='default 0.2 with --init, 1.0 from scratch')
    parser.add_argument('--epsilon-end', type=float, default=0.02)
    parser.add_argument('--epsilon-decay', type=int, default=1_000_000)
    parser.add_argument('--eval-every', type=int, default=100_000)
    parser.add_argument('--eval-games', type=int, default=8)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lab_config = validate_config({'network': args.network, 'seed': args.seed, 'device': args.device})
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = PongEnvironment()
    env_spec = _clean(env.spec)
    learner = DQNLearner(env.action_count, gamma=lab_config['gamma'],
                         learning_rate=lab_config['learning_rate'], device=lab_config['device'],
                         optimizer='nature_rmsprop', gradient_clip=None, architecture=args.network)
    epsilon_start = args.epsilon_start if args.epsilon_start is not None else (0.2 if args.init else 1.0)

    if args.init:
        payload = torch.load(args.init, map_location='cpu', weights_only=False)
        if payload.get('architecture', 'nature') != args.network:
            raise SystemExit(f"--init is a {payload.get('architecture')} network, not {args.network}")
        mirrored = mirror_state_dict(payload['online'])
        learner.online.load_state_dict(mirrored)
        learner.sync_target()
        # Exactness check: mirrored network on flipped frames == original on original frames.
        reference = DQNLearner(env.action_count, architecture=args.network).online
        reference.load_state_dict(payload['online'])
        probe = torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8)
        with torch.no_grad():
            difference = (reference(probe) - learner.online.cpu()(probe.flip(-1))).abs().max().item()
        learner.online.to(learner.device)
        print(f'mirrored {args.init}: max |dQ| = {difference:.2e}', flush=True)
        if difference > 1e-3:
            raise SystemExit('mirroring check failed')

    replay = FrameReplayBuffer(args.replay, seed=args.seed)
    opponent = ScriptedOpponent(env, rng)
    (out / 'args.json').write_text(json.dumps({**vars(args), 'epsilon_start': epsilon_start,
                                               'lab_config': lab_config, 'environment_spec': env_spec},
                                              indent=2) + '\n')
    log = (out / 'metrics.csv').open('w', newline='')
    writer = csv.DictWriter(log, fieldnames=['event', 'step', 'game', 'return', 'decisions', 'epsilon',
                                             'loss', 'mean_q', 'eval_mean', 'wall_seconds'])
    writer.writeheader()
    step = game = 0
    best = -float('inf')
    losses, qs = [], []
    started = time.perf_counter()
    next_eval = args.eval_every

    def epsilon_now():
        fraction = min(1.0, max(0, step - args.learning_starts) / args.epsilon_decay)
        return epsilon_start + fraction * (args.epsilon_end - epsilon_start)

    def on_step(state, action, reward, successor, terminated):
        nonlocal step
        replay.add(state, action, float(np.sign(reward)), successor, terminated)
        step += 1
        if step >= args.learning_starts and step % 4 == 0:
            update = learner.update(replay.sample(32))
            losses.append(update.loss)
            qs.append(update.mean_q)
            if learner.update_count % lab_config['target_sync'] == 0:
                learner.sync_target()

    try:
        while step < args.steps:
            opponent.new_game()
            replay.start_episode()
            epsilon = epsilon_now()
            total, decisions, _ = play_game(env, opponent, learner, rng, epsilon,
                                            args.seed * 100_003 + game, on_step)
            game += 1
            writer.writerow(dict(event='game', step=step, game=game, decisions=decisions, epsilon=round(epsilon, 4),
                                 loss=np.mean(losses) if losses else '', mean_q=np.mean(qs) if qs else '',
                                 wall_seconds=round(time.perf_counter() - started), **{'return': total}))
            losses.clear()
            qs.clear()
            if game % 20 == 0:
                log.flush()
                print(f'game={game} step={step}/{args.steps} return={total:+.0f} eps={epsilon:.3f}', flush=True)
            if step >= next_eval or step >= args.steps:
                next_eval += args.eval_every
                mean, returns = evaluate(learner, 10_000 + args.seed, args.eval_games)
                writer.writerow(dict(event='eval', step=step, game=game, eval_mean=mean,
                                     wall_seconds=round(time.perf_counter() - started)))
                log.flush()
                print(f'eval step={step} mean={mean:+.2f} returns={returns}', flush=True)
                extra = {'training_step': step, 'eval_mean_return_vs_hard_scripted': mean,
                         'initialized_from': args.init}
                save(learner, out / 'latest' / 'checkpoints' / 'v0' / 'learner.pt', env_spec, lab_config, extra)
                if mean > best:
                    best = mean
                    save(learner, out / 'best' / 'checkpoints' / 'v0' / 'learner.pt', env_spec, lab_config, extra)
    finally:
        log.close()
        env.close()
        replay.close()
    print(f'done: best eval mean {best:+.2f}; checkpoints under {out}', flush=True)


if __name__ == '__main__':
    main()
