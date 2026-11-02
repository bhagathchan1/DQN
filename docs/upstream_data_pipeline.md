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
the environment and schema, and a PNG tiles the first four-frame state. This sample is
for data-pipeline inspection only; the DQN training replay is generated online and is
not a static training file.

The upstream pipeline can also be selected in the environment section of a YAML
training configuration:

```yaml
environment:
  id: ALE/Breakout-v5
  pipeline: upstream
```

This matches the linked project's data preparation. The rest of the training protocol
still needs separate parity settings before the experiment can be called an exact
reproduction.
