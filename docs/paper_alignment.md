# Alignment with Mnih et al. (2015)

This page describes the inherited course-budget configuration. The new
`configs/nature.yaml` adds full-budget multi-game runs, single-frame replay,
random no-op starts, life-loss learning terminals and the Nature optimizer.
See [Atari Lab](atari_lab.md) for exact units, the released-code budget and
remaining differences. In particular, the inherited “50 million emulator frames”
wording below should not be used to size the new preset: it uses 50 million
action decisions, matching the released DeepMind launcher, with action repeat four.

This project reproduces the central Deep Q-Network mechanism from
[Mnih et al., *Nature* 518, 529–533 (2015)](https://doi.org/10.1038/nature14236)
on one Atari game. It is an educational, compute-bounded reproduction rather
than a claim that the paper's 49-game, 50-million-frame experiment was rerun.

## Implemented directly

| Paper element | Project implementation |
|---|---|
| Visual state | Four consecutive 84 x 84 luminance frames |
| Flicker reduction | Pixelwise maximum over the final two frames in each action repeat |
| Action repeat | Four emulator frames per selected action |
| Network | 32 filters at 8 x 8/4, 64 at 4 x 4/2, 64 at 3 x 3/1, then 512 hidden units and one output per action |
| Value target | One-step Q-learning target with a separately held target network |
| Replay | Random minibatches from a finite circular replay memory |
| Discount | 0.99 |
| Training rewards | Positive rewards clipped to 1 and negative rewards to -1 |
| Exploration | Epsilon-greedy, annealed from 1.0 to 0.1 over one million emulator frames |
| Optimizer and loss | RMSProp with a Huber loss, which implements the paper's clipped-error gradient behavior |
| Evaluation policy | Epsilon 0.05 with unmodified game rewards and a five-minute episode cap |

The implementation also preserves true environment termination separately from a
time-limit truncation, so truncation does not silently remove Bellman bootstrapping.

## Deliberate compute and protocol deviations

| Dimension | Nature experiment | This project |
|---|---:|---:|
| Games | 49 | Breakout only |
| Training budget per run | 50 million emulator frames | 1 million emulator frames |
| Replay capacity | 1 million recent frames | 10,000 complete stacked-state transitions |
| Evaluation episodes | 30 | 10 per saved policy and seed |
| Training seeds | Not presented as a matched three-seed comparison | Fixed matched seeds 7, 17 and 29 |
| Learner frequency | Paper algorithm updates inside each agent step | One update per four selected actions after warm-up |
| Optimizer implementation | DeepMind's RMSProp variant | PyTorch RMSProp |
| Episode treatment | Life loss used as a training episode boundary | Full game termination used as the terminal signal |
| Initial evaluation conditions | Random no-op starts | Deterministic episode seeds plus epsilon-greedy action randomness |

These differences reduce runtime and memory use but also make the observed scores
incomparable with the paper's published Breakout score. The valid conclusions are
limited to whether learning appears within this project budget and whether the
prioritized-replay extension differs from uniform replay under the matched local
protocol.

## Extension outside the 2015 algorithm

The proportional prioritized replay configuration samples transitions according to
their absolute temporal-difference error and applies normalized importance-sampling
weights. It is evaluated as a clearly labeled extension. It is not attributed to the
2015 DQN paper.
