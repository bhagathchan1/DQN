"""Exercise the real Atari UI without opening a window; save a review screenshot."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import torch
from dqn_project.play import ArenaUI
from dqn_project.arena import paired_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='tmp/arena-preview.png')
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ui = ArenaUI(SimpleNamespace(games=['Breakout'], checkpoint_dir=str(Path(args.checkpoint).parent),
                                checkpoint=args.checkpoint, seed=10007, max_steps=8, epsilon=0.05,
                                output=str(output.with_suffix('.csv'))))
    try:
        ui.draw()
        for mode in ['human', 'ai']:
            ui.start(mode)
            assert ui.round is not None, ui.status
            while not ui.round.done:
                ui.round.step(1)
            ui.last_frame = ui.round.frame()
            ui.last_score, ui.last_steps = ui.round.score, ui.round.steps
            ui.finish()
        assert len(paired_scores(ui.results, ui.game, ui.checkpoint)) == 1
        ui.draw()
        ui.pg.image.save(ui.screen, output)
        print(f'AI and scripted human-input rounds complete; screenshot: {output}')
    finally:
        ui.finish()
        ui.pg.quit()


if __name__ == '__main__':
    main()
