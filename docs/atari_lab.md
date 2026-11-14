# Atari training and human/AI comparison

## Setup and quick verification

From the repository folder on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[all]'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m dqn_project.pipeline --config configs/nature.yaml --games Breakout Pong --output experiments/nature-smoke --smoke --device cpu
```

Use `.venv/bin/python` on Linux/macOS. The smoke test uses only 64 decisions,
small replay and two capped evaluation episodes. It tests the optimizer, replay,
validation, checkpoint retention/reloading, and final/random evaluation. Its
scores say nothing about trained performance.

## Paper-scale training

```powershell
.\.venv\Scripts\python.exe -m dqn_project.pipeline --config configs/nature.yaml --games Breakout --output experiments/nature-breakout --seeds 7 --device cuda
```

Use `--device cpu` if your PyTorch has no CUDA support (as on the environment
tested here), or `auto`. A GPU-compatible PyTorch installation is needed for
CUDA. A short CPU smoke test is practical; the full experiment is a substantial
compute job. No full training job is started automatically by installing this code.

Multiple independent agents, one per game and seed:

```powershell
.\.venv\Scripts\python.exe -m dqn_project.pipeline --config configs/nature.yaml --games Breakout Pong SpaceInvaders Seaquest BeamRider --output experiments/nature-five --seeds 7 --device cuda
```

`--games all` expands to the paper's 49 games, using modern ALE names. All 49
IDs were checked against the installed ALE registry. Runs are sequential, and
each has its own weights, replay, logs, resolved config and final evaluation.
This is not a single agent trained jointly across games.

### Exact units and choices

`configs/nature.yaml` is a modern paper-aligned preset with these settings:

| Setting | Value |
| --- | --- |
| Budget | 50,000,000 action decisions per game, following the released DeepMind launcher |
| Action repeat | 4; up to 200,000,000 training emulator frames, excluding reset no-ops/validation |
| Replay | 1,000,000 single processed frames; stacks reconstructed during sampling |
| Learning starts | 50,000 decisions |
| Batch / update interval | 32 / every 4 decisions |
| Target copy | Every 10,000 optimizer updates, as specified in the paper's table |
| Epsilon | 1.0 to 0.1 over 1,000,000 decisions after warm-up |
| Optimizer | Centered RMSProp, mean/square decay 0.95, learning rate 0.00025, epsilon 0.01 inside square root |
| Loss | Huber; summed batch gradients for the Nature optimizer; no extra norm clipping |
| Rewards | Sign clipping for learning, raw scores for reporting |
| Life loss | Disables bootstrapping for that transition, without resetting the full game |
| Starts | 1 to 30 seeded no-op emulator steps after reset |
| Validation | Every 250,000 decisions, 30 episodes at epsilon 0.05, capped at 4,500 decisions each |
| Final testing | 30 episodes on training seed + 10,000; random baseline on the same seeds |

The paper uses the word “frames” in several contexts. This project deliberately
uses explicit action-decision counters. The budget follows
[DeepMind's released launcher](https://github.com/google-deepmind/dqn/blob/master/run_cpu),
which sets 50 million steps and action repeat four. If your course requires
**50 million raw emulator frames**, copy the YAML and set
`total_environment_steps: 12500000`; this is a different budget. Actual training
frame counts are logged, excluding reset no-ops and separate evaluation.

The target-copy interval follows Extended Data Table 1's parameter-update unit;
the released Lua learner instead checks its interaction-step counter. This
difference is explicit rather than hidden behind a shared “step” label.
The optimizer equation was independently implemented from the
[released learner](https://github.com/google-deepmind/dqn/blob/master/dqn/NeuralQLearner.lua).

### Storage, checkpoints and limitations

Replay uses a disk-backed uint8 array, about **6.6 GiB per game/seed**, plus small
metadata and episode-boundary stacks. Prefer an SSD; mapped pages can also occupy
RAM via the OS cache. The four oldest transitions are excluded when their history
has been overwritten. It uses one processed frame per transition instead of
duplicating eight frames in full current/next stacks.

The preset retains the latest two periodic checkpoints, the best validation
checkpoint and the final checkpoint. Only checkpoint paths created during the
current run are eligible for retention cleanup. Validation uses seed + 1,000,000,
separate from final testing. The pipeline's reported DQN score evaluates the
**final** checkpoint; `*_best.pt` is available for explicitly selected evaluation
or UI playback. Do not pick checkpoints using final test scores.

This implementation does not promise bitwise or score-level replication of 2015:
it uses modern ALE/PyTorch, bilinear luminance resizing, Kaiming initialization,
full-game logging, and continuous frame history across life loss. Validation uses
a fixed episode-count protocol, and random actions use the same four-frame repeat
as DQN (the paper's random baseline used six). Human UI controls also use four-frame
repeat. The UI is a local comparison tool, not the paper's professional-human
benchmark. Existing `default.yaml` remains a smaller course-budget preset.

Checkpoints save weights, optimizer state and configuration. Exact interrupted
training resume, including replay, RNG and ALE state, is **not implemented**.
Start each experiment in a new output folder. The pipeline refuses existing paths.
The saved checkpoint can still be tested or played after an interrupted run.

## Play and compare

```powershell
.\.venv\Scripts\python.exe -m dqn_project.play --checkpoint-dir experiments/nature-breakout --games Breakout
```

For a specific checkpoint:

```powershell
.\.venv\Scripts\python.exe -m dqn_project.play --checkpoint experiments/nature-breakout/Breakout/uniform_seed7/checkpoints/uniform_seed7_final.pt --games Breakout
```

The UI discovers local `.pt` files under `--checkpoint-dir` (defaults to
`experiments`). Click the checkpoint button to cycle files. It rejects an AI
checkpoint from another game. The game button cycles the requested games;
`--games all` enables the 49-game menu. A human can play without a checkpoint,
but an AI round requires one. Existing inherited checkpoints may be selected
explicitly; they are not presented as your new training results.

1. Select a game and checkpoint.
2. Click **Play human**. Hold arrows or WASD to move and Space to fire/start.
   Some Atari games map vertical movement to the joystick's LEFT/RIGHT actions
   (for example Pong); use left/right or A/D in those games.
3. Click **Watch AI** on the same seed, or play AI first.
4. Compare the raw returns, then click **Next seed** for another pair.

Both modes use the same action set, reset seed, random no-ops, action repeat and
five-minute game-time cap. The display targets 15 decisions/sec (60 emulator
frames/sec); slower hardware can stretch wall-clock time. Scores are not clipped.
AI uses epsilon 0.05 by default. `--epsilon 0` chooses greedy play; `--max-steps`
changes the cap for both modes. Escape/Stop/window-close saves an aborted attempt,
which is excluded from comparisons. There is no pause/reload advantage mid-round.

The table shows paired completed rounds from the current UI session, grouped by
game, checkpoint path, seed and protocol. Repeating a mode on the same seed replaces
its displayed score with the latest completed attempt; all attempts remain in
`experiments/arena/scores.csv`. That CSV includes mode, raw score, termination
reason, seed, caps, checkpoint path and training budget when available. The table
does not reload past sessions, preventing accidental pooling of unrelated tests.

This is a comparison of separate human and AI runs, not simultaneous multiplayer
or human-versus-agent competition inside a two-player Atari game.

## Validation performed

- Unit tests cover replay reconstruction/reset/wrap-around, disk storage, the
  optimizer equation, no-op seeding, life loss, controls and score pairing.
- Real Breakout and Pong smoke training/evaluation runs exercised the new preset.
- `scripts/smoke_arena.py` exercises real Atari with scripted human inputs and an
  AI checkpoint, and renders a screenshot without opening a window. It does not
  claim to measure a human player's skill.

```powershell
.\.venv\Scripts\python.exe scripts/smoke_arena.py --checkpoint experiments/nature-smoke/Breakout/uniform_seed7/checkpoints/uniform_seed7_final.pt
```
