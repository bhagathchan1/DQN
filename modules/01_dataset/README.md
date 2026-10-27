# Module 1: Dataset and experience generation

**Owner: Member 1**

## Purpose and responsibilities

Generate the agent's training experience by interacting with Atari environments.
Unlike supervised learning, this project collects transitions online as the
policy changes; it does not require a labeled image dataset before training.

- Configure environment creation, reset/step behavior, seeds and action spaces.
- Maintain grayscale resizing, max pooling, action repeat and frame stacking.
- Maintain uniform, prioritized and frame-based replay storage and sampling.
- Preserve terminal versus time-limit behavior and distinguish raw scores from
  clipped training rewards.
- Document game, preprocessing, collection budget and storage requirements.
- Support the UI's environment setup and keyboard-to-action mapping with Member 3.

## Implementation ownership

- [environment.py](../../src/dqn_project/environment.py): environment and preprocessing.
- [replay_buffer.py](../../src/dqn_project/replay_buffer.py): transition/batch contract and uniform replay.
- [prioritized_replay.py](../../src/dqn_project/prioritized_replay.py): priority sampling and updates.
- [frame_replay.py](../../src/dqn_project/frame_replay.py): frame storage.
- [games.py](../../src/dqn_project/games.py): game registry.

Member 2 owns the collection schedule and action selection inside the training
loop. Member 1 supplies the environment and storage components used by that loop.

## Inputs and outputs

Inputs: environment configuration, seed, action selected by the agent and replay
settings. Outputs: states as `numpy.uint8 [4,84,84]`, rewards, termination flags,
action counts and learner-compatible `ReplayBatch` values.

Store true `terminated` in replay. A `truncated` episode resets collection but
continues to bootstrap in the learning target. Follow [SPEC.md](../../docs/SPEC.md).

## Validation and deliverables

```powershell
python -m unittest discover -s tests -p test_environment.py -v
python -m unittest discover -s tests -p test_replay_buffer.py -v
python -m unittest discover -s tests -p test_nature.py -v
python scripts/environment_smoke_test.py --env ALE/Breakout-v5
```

Deliver the working collector components, documented preprocessing/storage choices
and test evidence. Completion requires correct state shape/dtype, valid actions,
replay capacity/wrap-around, sampling and priority feedback, and reproducible
seeded setup. The Atari smoke test requires installed environment dependencies
and available ROMs; synthetic tests provide fast interface checks.
