# DQN project specification

## Scope

The baseline reproduces the central mechanics of Mnih et al. (2015) on one visual
Atari environment when local runtime and ROM access permit. CartPole is not used for
paper-aligned results. The extension compares uniform replay against proportional
prioritized replay under an otherwise matched protocol.

## Interface contract

### State

`numpy.uint8` with shape `[4, 84, 84]`, channel-first and values in `[0, 255]`.
The four channels are consecutive processed grayscale frames, oldest first. Batched
states have shape `[B, 4, 84, 84]`. The network performs conversion to `[0, 1]`.

### Action and transition

Actions are integer indices from zero through `action_count - 1`. A stored transition
is `(state, action, clipped_reward, next_state, terminated)`. `truncated` ends a
collection episode but is not stored as terminal, so time-limit truncation continues
to bootstrap. This distinction must not be silently changed.

### Replay sample

`ReplayBatch` fields are:

| Field | Shape | dtype | Meaning |
|---|---:|---|---|
| `states` | `[B,4,84,84]` | `uint8` | current state |
| `actions` | `[B]` | `int64` | chosen action |
| `rewards` | `[B]` | `float32` | training reward |
| `next_states` | `[B,4,84,84]` | `uint8` | successor state |
| `terminated` | `[B]` | `bool` | true MDP terminal |
| `weights` | `[B]` | `float32` | one for uniform; IS weight for priority |
| `indices` | `[B]` | `int64` | replay slots for priority feedback |

Both replay implementations expose `add`, `sample(batch_size, beta)` and
`update_priorities(indices, priorities)`.

### Learner

`DQNLearner.update(batch)` returns scalar loss/means and absolute per-item TD errors.
The Bellman target is `r` when terminated and `r + gamma * max_a Q_target(s', a)`
otherwise. Target synchronization is an explicit operation controlled by the training
loop. Checkpoints contain both networks, optimizer state, update count and provenance.

### Metrics

CSV columns and units are defined once in `logging_utils.METRIC_FIELDS`. `frames`
means emulator frames (`environment_steps * 4` in the default wrapper), while
`environment_steps` means decisions/action repeats. Empty fields are expected for
event types to which a metric does not apply.

## Acceptance checklist

- [x] Environment reset/step contract and preprocessing tested with a fake image env.
- [x] Network output dimension is configurable and tested.
- [x] Uniform replay capacity, wrap-around, shapes and dtypes tested.
- [x] Terminal/non-terminal Bellman targets tested.
- [x] A finite optimization step changes online-network parameters.
- [x] Target synchronization and checkpoint round-trip tested.
- [x] Uniform and prioritized replay share a learner-compatible batch.
- [x] Priority probabilities, non-zero priorities, weights and updates tested.
- [x] CSV logs drive plots and contain run/config/replay provenance.
- [x] Real Atari environment reset/step smoke test on the execution platform.
- [x] Tiny collection/replay/update/checkpoint/evaluation pipeline smoke run.
- [ ] Random, uniform-DQN and prioritized-DQN training/evaluation runs completed.
- [ ] Results and presentation created only after real experiment logs exist.
