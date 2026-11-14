"""Verify the installed two-player Pong adapter and save actual Atari frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dqn_project.pong_environment import PongEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='tmp/two-player-pong-probe')
    parser.add_argument('--decisions', type=int, default=150)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    game = PongEnvironment()
    try:
        state = game.reset(seed=7)
        assert state.shape == (4, 84, 84) and state.dtype == np.uint8
        Image.fromarray(game.render()).save(output / 'reset.png')
        before = int(game.raw.frame)
        _, _, _, _, info = game.step(game.action_for_keys({'fire'}), game.action_for_keys({'fire'}))
        assert int(game.raw.frame) - before == 4, 'frame skip was applied more than once'
        assert info['emulator_frames'] == 4
        trace = []
        rng = np.random.default_rng(7)
        for index in range(args.decisions):
            keys = {'up', 'fire'} if index < 15 else {'down', 'fire'} if index < 30 else {'fire'}
            human_action = game.action_for_keys(keys)
            machine_action = int(rng.integers(game.action_count))
            _, reward, terminated, truncated, info = game.step(human_action, machine_action)
            trace.append(dict(index=index, human_action=human_action, machine_action=machine_action,
                              human_reward=info['human_reward'], machine_reward=reward,
                              emulator_frames=info['emulator_frames']))
            if index in {14, 29, args.decisions-1}:
                Image.fromarray(game.render()).save(output / f'decision-{index}.png')
            if terminated or truncated:
                break
        (output / 'probe.json').write_text(json.dumps(dict(
            spec=game.spec, observations=list(state.shape), trace=trace,
            total_emulator_frames=sum(row['emulator_frames'] for row in trace),
        ), indent=2), encoding='utf-8')
        print(json.dumps(dict(spec=game.spec, decisions=len(trace),
                              nonzero_rewards=[r for r in trace if r['human_reward'] or r['machine_reward']],
                              output=str(output.resolve())), indent=2))
    finally:
        game.close()


if __name__ == '__main__':
    main()
