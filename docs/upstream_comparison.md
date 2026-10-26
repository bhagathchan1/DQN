# Comparison with the linked GitHub implementation

The target repository is
[jihoonerd/Human-level-control-through-deep-reinforcement-learning](https://github.com/jihoonerd/Human-level-control-through-deep-reinforcement-learning).
This project's tested results are a compute-bounded, paper-oriented Breakout
reproduction with a prioritized-replay extension; they are not a reproduction of that
repository's exact training run or result curves.

## What aligns

| Element | Linked repository | This project |
|---|---|---|
| DQN network | 32×8×8/4, 64×4×4/2, 64×3×3/1 convolutional layers; 512-unit hidden layer; action-value outputs | Same layer dimensions and action-value head |
| Action repeat and state | Four-frame action repeat, max-pooling, four 84×84 grayscale frames | Four-frame repeat, max-pooling, four 84×84 luminance frames |
| Core learning | One-step target-network Q-learning, gamma 0.99, batch 32, clipped training rewards, Huber loss | Same broad mechanics |
| Uniform replay | 10,000 replay entries | 10,000 complete stacked-state transitions |

## Material differences

| Element | Linked repository | This project's completed runs | Consequence |
|---|---|---|---|
| Training budget | `training_frames = 10^7` selected steps, with four emulator frames per step (~40M emulator frames) | 250,000 selected steps/run (~1M emulator frames) | The upstream run is 40× longer; scores and learning curves are not comparable. |
| Optimizer | Adam, learning rate 0.0001, epsilon 1e-6 | PyTorch RMSProp, learning rate 0.00025 | The project setting follows the paper-oriented protocol more closely, not the GitHub code. |
| Target synchronization | Every 1,000 selected environment steps | Every 10,000 learner updates (one update per four selected steps) | Synchronization cadence differs substantially. |
| Reset wrappers | Random no-op reset (1–30), episodic-life handling, and FIRE reset where applicable | Seeded resets; no no-op, episodic-life, or FIRE wrapper | Initial states and episode boundaries differ. |
| Image resize | OpenCV RGB-to-gray and `INTER_AREA` resize | Pillow luminance conversion and bilinear resize | Pixel values can differ even for the same game state. |
| Exploration | 1.0 to 0.1 by 1M selected steps, then anneals toward 0.01 by 25M (the default 10M-step run ends before that) | 1.0 to 0.1 by 250,000 selected steps | Schedule and late-training exploration differ. |
| Evaluation | Greedy policy; five trials, then reports the best score; up to ten-minute safety cap | Epsilon 0.05; ten held-out episodes; reports mean/spread; five-minute cap | The project reports a more stable mean, but not the same evaluation protocol. |
| Scope | README shows Atlantis, Boxing, Breakout and Pong examples | Breakout only; three matched training seeds; prioritized replay added as an extension | The project does not claim the upstream multi-game coverage. |

## Audit conclusion

Nothing in the linked repository changes the validity of the already reported
course-scale comparison, but the two projects are not implementation-equivalent. If
exact parity with the GitHub repository is required, a separate upstream-parity config
and new runs are needed; existing checkpoints must not be relabeled as upstream runs.
The linked repository's original visual files are retained under
`assets/upstream_dqn/` with attribution and are labeled as upstream material in the
README and showcase deck.

## Inspected upstream files

- `dqn/agent/dqn_agent.py`: default step budget, optimizer, epsilon schedule, target
  cadence, and best-of-five evaluation.
- `dqn/utils/atari_wrappers.py`: no-op, episodic-life, FIRE, max-skip, and resize wrappers.
- `dqn/networks/dqn_network.py`: convolutional architecture.
- `README.md` and `LICENSE`: asset gallery and MIT notice.
