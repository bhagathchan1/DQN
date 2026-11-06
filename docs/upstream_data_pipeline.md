# Upstream-style Atari data pipeline

The linked TensorFlow project does not download a prebuilt dataset. During training,
the agent plays Atari and writes each online experience to a replay memory:

```text
(state, action, sign-clipped reward, next_state, done)
```

Its environment preparation uses random no-op starts (1–30), four-frame action
repetition with max-pooling over the final two observations, episodic-life boundaries,
FIRE startup actions when the game requires them, OpenCV grayscale/84×84 resizing,
sign-clipped training rewards, and four-frame stacks. The linked agent uses a
10,000-transition replay memory and samples batches during training.

This repository now exposes that Atari data pipeline through
`make_env(..., pipeline="upstream")`. Frames are returned as `uint8` channel-first
arrays `[4, 84, 84]` to fit this project's PyTorch model; pixel preprocessing and frame
ordering follow the linked pipeline. A small random-policy sample can be exported for
review with:

```bash
python scripts/collect_transition_sample.py --steps 256 --seed 7 --output data/breakout_sample.npz
```

The compressed archive contains `states`, `actions`, clipped `rewards`, `raw_rewards`,
`next_states`, `terminated`, `truncated`, and `episode_ids`. The companion JSON records
the environment and schema, and a PNG tiles the first four-frame state. By default,
the sample uses uniformly random actions and is for data-pipeline inspection.

To collect transitions from a trained DQN policy instead, pass a saved checkpoint:

```bash
python scripts/collect_transition_sample.py \
  --env ALE/Breakout-v5 \
  --steps 5000 \
  --seed 7 \
  --checkpoint checkpoints/breakout_upstream_port_seed7_episode_14000.pt \
  --epsilon 0.34 \
  --device auto \
  --output data/breakout_agent_sample.npz
```

This starts a fresh environment and records 5,000 new transitions using the checkpoint
policy (with epsilon-greedy exploration at the requested rate). It is genuine agent
policy experience, but it is a separate collection rollout, not a dump of the training
run's replay buffer. The original training replay is generated online, has a 10,000
transition capacity, and is not currently persisted as a dataset.

The upstream pipeline can also be selected in the environment section of a YAML
training configuration:

```yaml
environment:
  id: ALE/Breakout-v5
  pipeline: upstream
```

`configs/upstream_breakout.yaml` selects the upstream-compatible training profile,
including the optimizer, exploration schedule, update cadence, replay sampling,
episode checkpointing, greedy best-of-five evaluation, GIFs, and TensorBoard scalars.
The profile is a PyTorch/Gymnasium port, not the original TensorFlow/Gym runtime; RNG
sequences and pixel/runtime edge cases can therefore differ. Completed project runs
remain on their original configuration and need new runs to produce upstream-profile
results.
