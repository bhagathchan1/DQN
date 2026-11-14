# Trained models

GPU training runs from 2026-11-13 (seed 7). All checkpoints are PyTorch `.pt` files.

## Pong, single-player (10M environment steps)

| File | Network | Validation mean |
|---|---|---|
| `pong_single/pong_nature_seed7_best.pt` | Nature DQN | +19.1 |
| `pong_single/pong_dueling_seed7_best.pt` | Dueling DQN | +17.2 |

Configs: `configs/pong_nature.yaml`, `configs/pong_dueling.yaml`.

## Pong Learning Lab, two-player fine-tune (3M steps)

Initialised from the single-player best checkpoints above, then fine-tuned with
`scripts/train_pong_two_player.py` (epsilon 0.2 -> 0.02). Settings are in `*_args.json`.

| File | Network | Best eval mean (8 games) |
|---|---|---|
| `pong_two_player/pong_lab_nature_best_learner.pt` | Nature DQN | +21.0 |
| `pong_two_player/pong_lab_dueling_best_learner.pt` | Dueling DQN | +19.0 |

These are lab `learner.pt` checkpoints. The Pong Learning Lab UI lists every `.pt` in
`pong_two_player/` under **Opponent** in the setup card (the default). Choosing one creates a
session that imports it, uses its network, and plays at epsilon 0.05 instead of the fresh-model
1.0; it keeps learning from your games after each match. `dqn-lab --models-dir <folder>` points
the lab at another folder. Package-version differences (pettingzoo, multi-agent-ale-py) are
allowed and logged in the session's `events.jsonl`; sides, actions and preprocessing must match.
Note: in the installed pettingzoo `pong_v3`, `first_0` (lab "human") moves the right paddle and
`second_0` (DQN) the left one.

The `pong_single/` checkpoints come from single-player Gymnasium Pong and cannot be imported
directly; `scripts/train_pong_two_player.py --init` mirrors them into a two-player model.

## Breakout (10M environment steps, code from `main`)

| File | Replay | Last-100 training-episode average |
|---|---|---|
| `breakout/breakout_upstream_10M_seed7_final.pt` | uniform (upstream config) | 12.9 |
| `breakout/breakout_uniform_10M_seed7_final.pt` | uniform | 46.6 |
| `breakout/breakout_prioritized_10M_seed7_final.pt` | prioritized | 29.9 |

These Breakout numbers are averages from training episodes, which still include
exploration (epsilon about 0.07-0.1). They are not greedy evaluation scores.

## Metrics

`metrics/` holds the per-episode / evaluation CSV for every run.
