# Module 4: Human-versus-agent UI

**Integration lead: Member 3. Contributors: all three members.**

## Purpose

Let a user play an Atari game and compete against a trained DQN agent's score.
The existing desktop UI runs separate human and AI rounds on the same game and
seed, then compares their scores. This is score competition through paired rounds.

## Responsibilities

| Member | Responsibility |
|---|---|
| Member 1 | Environment/rendering integration, game/action compatibility, keyboard mapping support and consistent preprocessing |
| Member 2 | Checkpoint discovery/loading, inference, model metadata and training-step provenance |
| Member 3 | Screen flow, controls, paired score display, CSV output, error handling and acceptance testing |

## Implementation and interfaces

- [play.py](../../src/dqn_project/play.py): Pygame window, controls and round flow.
- [arena.py](../../src/dqn_project/arena.py): round execution, action mapping, score persistence and pairing.
- [test_arena.py](../../tests/test_arena.py): arena behavior checks.
- [Atari Lab guide](../../docs/atari_lab.md): detailed gameplay and training instructions.

Input: game ID, seed, episode decision limit, evaluation epsilon and a matching
trained checkpoint. Output: rendered gameplay, human/AI scores, paired differences
and CSV history at `experiments/arena/scores.csv` by default.

Use the same game, checkpoint selection, seed, step limit, epsilon and no-op
settings for a pair. Compare completed rounds only; aborted attempts remain in
the CSV but are excluded from paired scores. The current screen summarizes
the current session's rounds; the CSV persists attempts across sessions.

## Launch and play

Install the `all` extra as described in the main README, including Pygame and Atari
dependencies. After Module 2 produces a checkpoint:

```powershell
python -m dqn_project.play --checkpoint-dir experiments/team-smoke --games Breakout --seed 10007
```

1. Select the game and its matching checkpoint.
2. Choose **Play human**; use arrows/WASD to move and Space to fire.
3. Choose **Watch AI** on the same seed and compare the completed scores.
4. Choose **Next seed** for another pair. Escape or **Stop round** aborts a round.

The smoke checkpoint checks integration; use a fully trained checkpoint for the
final competition demo.

## Acceptance criteria

- Human controls and AI inference both complete a round.
- Missing or mismatched checkpoints produce an understandable status message.
- Completed human/AI attempts on matching settings yield a score comparison.
- Aborted rounds are saved and excluded from comparison.
- Changing games/checkpoints does not mix incompatible comparisons.
- CSV records retain game, seed, checkpoint, score and completion status.

```powershell
python -m unittest discover -s tests -p test_arena.py -v
```

Member 3 also performs a manual launch/play/watch/abort/save check on a desktop
with available Atari ROMs; automated arena tests do not establish that the window
and keyboard work on the demo machine.
