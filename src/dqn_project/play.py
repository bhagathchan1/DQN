"""Desktop Atari arena: play as a human, watch DQN, compare matched rounds."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .arena import ArenaRound, append_result, keyboard_action, paired_scores


class ArenaUI:
    def __init__(self, args):
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((1160, 800))
        pygame.display.set_caption('Atari Lab | Human + DQN')
        self.font = pygame.font.SysFont('Segoe UI', 20)
        self.small = pygame.font.SysFont('Segoe UI', 16)
        self.large = pygame.font.SysFont('Segoe UI', 36, bold=True)
        self.args = args
        self.games = args.games
        self.game_index = 0
        self.checkpoints = sorted(Path(args.checkpoint_dir).resolve().rglob('*.pt'))
        if args.checkpoint:
            selected = Path(args.checkpoint).resolve()
            self.checkpoints = [selected] + [p for p in self.checkpoints if p != selected]
        self.checkpoint_index = 0
        self.seed = args.seed
        self.round = None
        self.results = []
        self.last_frame = None
        self.status = 'Choose a game and checkpoint, then start a round.'
        self.buttons = []
        self.running = True
        self.last_score = 0.0
        self.last_steps = 0

    @property
    def game(self):
        return f'ALE/{self.games[self.game_index]}-v5'

    @property
    def checkpoint(self):
        return str(self.checkpoints[self.checkpoint_index]) if self.checkpoints else ''

    def start(self, mode):
        if self.round:
            return
        try:
            self.round = ArenaRound(self.game, mode, self.seed, self.checkpoint or None,
                                    self.args.max_steps, self.args.epsilon)
            self.last_frame = self.round.frame()
            self.last_score, self.last_steps = 0, 0
            trained = (f' ({self.round.checkpoint_steps:,} training decisions)'
                       if mode == 'ai' and self.round.checkpoint_steps is not None else '')
            self.status = f'{mode.upper()} playing{trained}. Arrows / WASD move; Space fires; Escape aborts.'
        except Exception as exc:
            self.round = None
            self.status = str(exc)

    def finish(self):
        if not self.round:
            return
        result = self.round.result()
        self.round.close()
        self.round = None
        self.results.append(result)
        append_result(self.args.output, result)
        matched = any(seed == self.seed for seed, _, _ in paired_scores(self.results, self.game, self.checkpoint))
        self.status = (f"{result['mode'].upper()}: {result['score']:g} points. " +
                       ('Pair complete. Choose Next seed for another comparison.' if matched and result['completed']
                        else 'Play the other mode on this seed to compare.' if result['completed']
                        else 'Aborted round saved; excluded from comparison.'))

    def text(self, value, x, y, color=(222, 232, 246), font=None):
        self.screen.blit((font or self.font).render(str(value), True, color), (x, y))

    def wrapped(self, value, x, y, width=510):
        line = ''
        for word in str(value).split():
            if self.small.size(line + word)[0] > width and line:
                self.text(line, x, y, font=self.small)
                y += 22
                line = ''
            line += word + ' '
        self.text(line, x, y, font=self.small)

    def button(self, label, x, y, width, action, enabled=True, accent=False):
        rect = self.pg.Rect(x, y, width, 43)
        color = (37, 109, 101) if accent else (38, 52, 73)
        if not enabled:
            color = (25, 33, 46)
        self.pg.draw.rect(self.screen, color, rect, border_radius=9)
        self.text(label, x+14, y+8, color=(225, 235, 245) if enabled else (99, 113, 132))
        if enabled:
            self.buttons.append((rect, action))

    def draw(self):
        self.screen.fill((12, 19, 31))
        self.buttons = []
        self.text('ATARI LAB', 30, 20, (99, 218, 194), self.small)
        self.text('Human + DQN', 28, 43, font=self.large)
        self.text('Same game. Same seed. Your turn.', 30, 96, (143, 161, 184))
        self.pg.draw.rect(self.screen, (4, 8, 14), (30, 142, 480, 630), border_radius=12)
        if self.last_frame is not None:
            surface = self.pg.surfarray.make_surface(np.transpose(self.last_frame, (1, 0, 2)))
            surface = self.pg.transform.scale(surface, (480, 630))
            self.screen.blit(surface, (30, 142))
        else:
            self.text('Ready, player?', 180, 414)
            self.text('Start a human or AI round', 150, 448, (143, 161, 184), self.small)
        idle = self.round is None
        self.button('Game: ' + self.games[self.game_index] + '  >', 550, 36, 575, 'game', idle)
        self.text('CHECKPOINT', 550, 98, (143, 161, 184), self.small)
        name = Path(self.checkpoint).name if self.checkpoint else 'No checkpoint found'
        if len(name) > 54:
            name = name[:51] + '...'
        self.button(name, 550, 122, 575, 'checkpoint', idle and bool(self.checkpoints))
        self.text('Click to cycle files. Checkpoint must match the game.', 550, 174, (143, 161, 184), self.small)
        self.text(f'Round seed {self.seed}   |   Limit {self.args.max_steps * 4 / 60:.1f}s game time', 550, 211)
        self.button('Play human', 550, 249, 180, 'human', idle, True)
        self.button('Watch AI', 745, 249, 180, 'ai', idle and bool(self.checkpoints), True)
        self.button('Next seed', 940, 249, 185, 'next', idle)
        self.button('Stop round', 550, 308, 180, 'stop', not idle)
        self.text(f'Score  {self.last_score:g}', 755, 312)
        self.text(f'{self.last_steps * 4 / 60:.1f}s played', 945, 312)
        self.text('MATCHED RESULTS', 550, 384, (99, 218, 194), self.small)
        for x, heading in [(550, 'Seed'), (690, 'Human'), (830, 'AI'), (970, 'AI - Human')]:
            self.text(heading, x, 417, font=self.small)
        pairs = paired_scores(self.results, self.game, self.checkpoint)
        for i, (seed, human, ai) in enumerate(pairs[-5:]):
            for x, value in [(550, seed), (690, f'{human:g}'), (830, f'{ai:g}'), (970, f'{ai-human:+g}')]:
                self.text(value, x, 446+i*26, font=self.small)
        if pairs:
            human = np.mean([p[1] for p in pairs])
            ai = np.mean([p[2] for p in pairs])
            self.text(f'{len(pairs)} paired rounds  |  Mean human {human:.1f}  /  AI {ai:.1f}', 550, 590)
        else:
            self.text('Complete both modes on the same seed to see a comparison.', 550, 458, (143, 161, 184), self.small)
        self.wrapped(self.status, 550, 637, 570)
        self.text('Arrows / WASD: move    Space: fire    Escape: abort', 550, 710, font=self.small)
        self.text('15 decisions/sec = 60 emulator frames/sec. No score clipping.', 550, 739, (143, 161, 184), self.small)
        self.text('Completed and aborted attempts are saved to CSV.', 550, 765, (143, 161, 184), self.small)
        self.pg.display.flip()

    def activate(self, action):
        if action in {'human', 'ai'}:
            self.start(action)
        elif action == 'stop':
            self.finish()
        elif action == 'next':
            self.seed += 1
        elif action == 'game':
            self.game_index = (self.game_index + 1) % len(self.games)
            self.last_frame = None
        elif action == 'checkpoint':
            self.checkpoint_index = (self.checkpoint_index + 1) % len(self.checkpoints)

    def run(self):
        clock = self.pg.time.Clock()
        try:
            while self.running:
                for event in self.pg.event.get():
                    if event.type == self.pg.QUIT:
                        self.running = False
                    elif event.type == self.pg.KEYDOWN and event.key == self.pg.K_ESCAPE:
                        self.finish()
                    elif event.type == self.pg.MOUSEBUTTONDOWN and event.button == 1:
                        for rect, action in self.buttons:
                            if rect.collidepoint(event.pos):
                                self.activate(action)
                                break
                if self.round and self.running:
                    keys = self.pg.key.get_pressed()
                    action = keyboard_action(self.round.meanings,
                        up=keys[self.pg.K_UP] or keys[self.pg.K_w], down=keys[self.pg.K_DOWN] or keys[self.pg.K_s],
                        left=keys[self.pg.K_LEFT] or keys[self.pg.K_a], right=keys[self.pg.K_RIGHT] or keys[self.pg.K_d],
                        fire=keys[self.pg.K_SPACE])
                    self.round.step(action)
                    self.last_frame = self.round.frame()
                    self.last_score, self.last_steps = self.round.score, self.round.steps
                    if self.round.done:
                        self.finish()
                self.draw()
                clock.tick(15 if self.round else 30)
        finally:
            self.finish()
            self.pg.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', nargs='+', default=['Breakout', 'Pong', 'SpaceInvaders', 'Seaquest', 'BeamRider'])
    parser.add_argument('--checkpoint-dir', default='experiments')
    parser.add_argument('--checkpoint', help='Select this checkpoint first')
    parser.add_argument('--seed', type=int, default=10007)
    parser.add_argument('--max-steps', type=int, default=4500)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--output', default='experiments/arena/scores.csv')
    args = parser.parse_args()
    if args.games == ['all']:
        from .games import NATURE_GAMES
        args.games = list(NATURE_GAMES)
    if args.seed < 0 or args.max_steps <= 0 or not 0 <= args.epsilon <= 1:
        parser.error('seed must be nonnegative, max-steps positive and epsilon in [0,1]')
    torch.set_num_threads(1)
    ArenaUI(args).run()


if __name__ == '__main__':
    main()
