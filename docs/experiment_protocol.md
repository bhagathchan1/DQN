# Experiment protocol

## Research question

Can the implemented paper-aligned DQN baseline learn in `ALE/Breakout-v5` under a
course-scale budget, and how does proportional prioritized replay compare with uniform
replay when all other settings are held constant?

## Fixed protocol

- Environment: `ALE/Breakout-v5`, sticky actions disabled by construction, ALE frame
  skip set to one because the project wrapper repeats each selected action four times.
- Observation: max over the last two raw frames in an action repeat, luminance resize
  to 84 x 84, four processed frames stacked channel-first.
- Reward: raw reward for reporting; sign-clipped reward for training.
- Seeds: 7, 17 and 29, fixed before the full runs and matched across replay methods.
- Budget: 250,000 decisions / approximately 1,000,000 emulator frames per run.
- Exploration: epsilon from 1.0 to 0.1 over the run; evaluation epsilon 0.05.
- Learner: batch 32, gamma 0.99, RMSProp learning rate 0.00025, Huber loss, gradient
  norm clipping at 10, one update per four decisions after 20,000 warm-up decisions.
- Target network: hard synchronization every 10,000 learner updates.
- Replay: 10,000 complete stacked-state transitions. Uniform versus proportional
  priority (`alpha=0.6`, beta 0.4 annealed to 1.0), with all other settings matched.
- Evaluation: 10 episodes per policy and seed, capped at 4,500 selected actions
  (18,000 emulator frames, approximately five minutes). Report mean, standard
  deviation, minimum, maximum and individual returns. Use the same reset seeding and
  stopping rules for random, uniform and prioritized policies.

## Logging and provenance

Each run writes an append-only CSV with run ID, seed, environment steps, estimated
emulator frames, episode number/reward/length, evaluation mean/std where applicable,
loss, epsilon, wall time, config ID, replay type and event type. Checkpoints record
environment, budget, config path, seed and Git revision. Package versions and execution
hardware should be captured in the final report alongside the logs.

## Required comparisons

1. Random action baseline using the evaluation protocol.
2. Uniform-replay DQN.
3. Prioritized-replay DQN with the matched config and seeds.

Plot rolling training reward for diagnosis, but base the final comparison on held-out
evaluation episodes from saved checkpoints. Report per-seed results; do not select only
the best checkpoint or best seed without declaring that selection rule in advance.

## Deviations from Mnih et al. (2015)

- One game rather than 49; about one million emulator frames rather than 50 million.
- Ten pilot evaluation episodes rather than 30 games of up to five minutes.
- A much smaller replay capacity, storing complete frame stacks for clarity rather than
  a memory-efficient one-million-frame representation.
- Modern Gymnasium/ALE runtime and PyTorch implementation.
- Standard PyTorch RMSProp and gradient-norm clipping are not exact reproductions of
  every optimizer implementation detail in the paper.
- True terminations and time-limit truncations are distinguished explicitly.
- Prioritized replay is a project extension and must not be attributed as an algorithm
  specified in the 2015 paper.

These deviations make this a faithful educational implementation, not a numerical
reproduction of the paper's published scores.
