# Deep Q-Network (DQN): Reproduction and DeepQ Learning Lab

A PyTorch implementation of the core method from Mnih et al., *Human-level control
through deep reinforcement learning* (**Nature**, 2015), with two applications:

1. **A compute-bounded reproduction** of the DQN learning loop on Atari Breakout,
   including a matched-seed comparison of uniform and prioritized replay.
2. **DeepQ Learning Lab**, a browser demo with two games. In **Pong** you play two-player
   Atari Pong live against a DQN that learns from each completed game. In **Breakout** you
   take turns with a trained DQN on the same seed and compare scores. A **Trained models**
   page explains how every opponent was trained.

> **Scope.** This is an educational reproduction of the algorithm, not a rerun of the
> paper's 49-game, 50-million-frame benchmark. Results below are from short, local runs
> and are labelled with their budgets.

## Contents

- [The paper](#the-paper)
- [This implementation](#this-implementation)
- [Results](#results)
- [Quick start](#quick-start)
- [Branches](#branches)
- [Repository map](#repository-map)
- [Documentation](#documentation)
- [Limitations](#limitations)
- [Citation and attribution](#citation-and-attribution)

## The paper

**Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015).** *Human-level control through
deep reinforcement learning.* Nature 518, 529–533.
[Read the paper](https://www.nature.com/articles/nature14236) · [DOI: 10.1038/nature14236](https://doi.org/10.1038/nature14236)

The paper trains one agent per game from raw pixels and the game score, using the same
algorithm, network and hyperparameters across **49 Atari 2600 games**. The authors report
that the agent reaches more than 75% of a professional human tester's score on more than
half of the games (29 games). Scores are normalized so that random play is 0% and the
professional tester is 100%:

```
normalized score = 100 × (DQN score − random score) / (human score − random score)
```

The method has three parts that this repository implements:

| Component | What it does | Why |
|---|---|---|
| Convolutional Q-network | Maps four stacked 84 × 84 frames to one Q-value per action | Learns directly from pixels |
| Experience replay | Stores transitions in a buffer (1M frames in the paper) and trains on random minibatches | Breaks correlations between consecutive frames |
| Target network | A copy of the weights, refreshed every C = 10,000 updates, used for the bootstrap target | Keeps the regression target from moving every step |

The paper's update target is `y = r + γ · max_a′ Q(s′, a′; θ⁻)` (with `y = r` at episode
end), trained by minimizing `(y − Q(s, a; θ))²`. The paper reports an ablation in
which disabling replay or the target network hurts performance (Extended Data Table 3).

## This implementation

| Area | What is here |
|---|---|
| Learner | One-step DQN with uniform replay and a separate target network; Huber loss, gradient clipping, ε-greedy exploration |
| Network | Nature DQN (3 conv layers, 512 hidden units) and a Dueling variant (value and advantage streams) |
| Preprocessing | Grayscale, 84 × 84, max over the last two frames, 4-frame stack, action repeat 4 |
| Replay extension | Proportional prioritized replay with importance-sampling weights. It is an **extension**, not part of the 2015 paper |
| Training pipeline | Config-driven runs (`configs/`), multi-seed pipeline, checkpoints, evaluation and metrics CSVs |
| Interactive demo | DeepQ Learning Lab: FastAPI service and React/TypeScript UI. Pong: live two-player PettingZoo `pong_v3`, learning after each game. Breakout: same-seed turns against trained checkpoints |
| Tests | Unit and integration tests under `tests/` |

The implementation follows the paper in the mechanism, and deviates in budget, games and
schedule. The full comparison is in [`docs/paper_alignment.md`](docs/paper_alignment.md).

## Results

### Breakout: matched-seed replay comparison (short budget)

Three seeds (7, 17, 29) per replay method, each trained for a 250,000 selected-action budget
(about 1M emulator frames). Each policy was scored on 10 held-out episodes at ε = 0.05.
Source: [`results/reviewed_experiment_snapshot.json`](results/reviewed_experiment_snapshot.json).

| Policy | Seed 7 | Seed 17 | Seed 29 | Mean of seed means |
|---|---:|---:|---:|---:|
| Random (baseline) | 1.0 | 0.8 | 1.2 | 1.00 |
| Uniform replay | 2.4 | 1.9 | 6.2 | 3.50 |
| Prioritized replay | 4.5 | 0.8 | 1.5 | 2.27 |

- Uniform replay beat random on all three seeds.
- Prioritized replay's paired differences from uniform were +2.1, −1.1 and −4.7 points.
- With three seeds and 10 episodes each, the replay-method ranking is **not settled**.

Training curves and the evaluation comparison:

| Training reward over time | Evaluation score comparison |
|---|---|
| ![Training reward for the project's Breakout runs](results/plots/training_reward.png) | ![Evaluation score comparison for the project's Breakout runs](results/plots/evaluation_comparison.png) |

### Pong: single-player and two-player models

| Model | Budget | Result |
|---|---|---|
| Single-player Nature DQN (ALE/Pong-v5), seed 7 | 10M environment steps | Validation mean **+19.1** |
| Single-player Dueling DQN, seed 7 | 10M environment steps | Validation mean **+17.2** |
| Two-player lab fine-tune, Nature | 3M steps from the single-player weights | 8-game evaluation mean **+21.0** |
| Two-player lab fine-tune, Dueling | 3M steps from the single-player weights | 8-game evaluation mean **+19.0** |

Pong scores are points, and a game is won at 21. The single-player and two-player numbers
use different evaluation protocols, so they are not a head-to-head ranking. Checkpoints,
configs and metrics are in [`trained_models/`](trained_models/README.md).

### Breakout: 10M-step runs (training averages only)

| Run | Replay | Last-100 training-episode average |
|---|---|---:|
| Upstream-compatible profile, seed 7 | uniform | 12.9 |
| Uniform, seed 7 | uniform | 46.6 |
| Prioritized, seed 7 | prioritized | 29.9 |

These are averages over training episodes with exploration still on (ε ≈ 0.07–0.1). They are
**not** greedy evaluation scores. A greedy evaluation of these checkpoints is a planned next step.

### How the trained models were trained

All models are DQNs trained only from game rewards (no human examples), with seed 7 on a
GPU in November 2026. Every run uses 4 stacked 84 × 84 grayscale frames, a decision every
4 emulator frames, rewards clipped to −1/0/+1, γ = 0.99, batches of 32 and a Huber loss
against a frozen target network.

**Pong opponents (used live in the lab)** are trained in three stages:

1. **Pre-train on single-player Pong** (`ALE/Pong-v5`) against Atari's built-in opponent:
   10M decisions, Nature RMSProp (lr 0.00025), 1M-frame uniform replay, target copy every
   10k decisions, ε 1.0 → 0.1 over the first 1M. About 4.4 h (Nature) and 4.8 h (Dueling).
   Final validation +19.1 (Nature) and +17.2 (Dueling), 10 games at ε 0.05.
2. **Mirror the weights.** In two-player Pong the DQN controls the other paddle, so the
   network is flipped left-to-right. The convolutions tile 84 pixels exactly, so this is
   an exact transformation, and `scripts/train_pong_two_player.py` checks it numerically.
3. **Fine-tune in real two-player Pong** for 3M decisions (about 1,800 games, 2.3 h each)
   against a scripted ball-tracking paddle whose skill is redrawn every game: replay 500k,
   ε 0.2 → 0.02 over 1M, evaluation every 100k decisions over 8 greedy games against the
   hardest setting. Evaluation rose from −6.0 (Nature) and +1.0 (Dueling) after 100k
   decisions to a best of **+21.0** and **+19.0**.

In the lab, the chosen model plays you at ε 0.05 and keeps learning: 450 more updates after
every completed match.

**Breakout opponents (used in the lab's take-turns mode)** are three 10M-decision runs:
uniform replay (200k, RMSProp, target copy every 10k, 6.3 h), prioritized replay (same, with
α 0.6 and β 0.4 → 1, 7.2 h) and the upstream reference settings (Adam 0.0001, 10k replay,
target copy every 1k, episodes end at each lost life, 6.6 h). Their training averages are in
the table above.

The lab's **Trained models** page shows the same information. Files, configs and
per-episode metrics: [`trained_models/`](trained_models/README.md).

### Reference gameplay clips

These clips come from the upstream implementation this project builds on. **They are not
generated by this project's checkpoints.** Attribution is in
[`assets/upstream_dqn/ATTRIBUTION.md`](assets/upstream_dqn/ATTRIBUTION.md).

| Atlantis | Boxing |
|---|---|
| ![Atlantis gameplay reference](assets/upstream_dqn/atlantis.gif) | ![Boxing gameplay reference](assets/upstream_dqn/boxing.gif) |
| Breakout | Pong |
| ![Breakout gameplay reference](assets/upstream_dqn/breakout.gif) | ![Pong gameplay reference](assets/upstream_dqn/pong.gif) |

## Quick start

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[all]'
```

Atari environments need ALE support and ROMs under terms that apply to you. No ROMs are
included. Check the setup:

```bash
python scripts/environment_smoke_test.py --env ALE/Breakout-v5
```

### Train

```bash
dqn-train --config configs/default.yaml        # uniform replay, short budget
dqn-train --config configs/prioritized.yaml    # prioritized replay extension
dqn-train --config configs/upstream_breakout.yaml   # 10M-step profile; GPU recommended
```

Colab: [`notebooks/07_colab_gpu_breakout.ipynb`](notebooks/07_colab_gpu_breakout.ipynb) ·
[notebook instructions](notebooks/README.md). Multi-seed experiments: [pipeline guide](docs/pipeline.md).

### Evaluate and watch

```bash
dqn-evaluate --config configs/default.yaml \
  --checkpoint checkpoints/breakout_uniform_seed7_final.pt \
  --output results/metrics/uniform_evaluation.json

dqn-evaluate --config configs/default.yaml --random-baseline \
  --output results/metrics/random_evaluation.json

dqn-play --checkpoint checkpoints/your_checkpoint.pt --env ALE/Breakout-v5
```

Step-by-step viewing instructions: [`guides/watch-agent-play/README.md`](guides/watch-agent-play/README.md).

### DeepQ Learning Lab (Pong + Breakout)

One browser UI with three views, chosen at the top of the page:

- **Pong · live duel:** play two-player Pong against a trained (or fresh) DQN that learns
  after each game.
- **Breakout · take turns:** play a round (←/→ or A/D to move, Space to launch), then watch a
  trained Breakout DQN play the same seed at 1×, 2× or 4× speed. Both rounds get the same
  random no-op start, 5 lives and a 5-minute limit; results are saved to
  `experiments/breakout_arena/scores.csv`. Only one game runs at a time.
- **Trained models:** how each opponent was trained, with its results.

Setup differs by platform (WSL2 on Windows, native on Linux or macOS). Start with:

```bash
wsl bash scripts/run_wsl_lab.sh     # Windows: then open http://localhost:8765
dqn-lab                             # Linux / macOS, after pip install -e ".[env,lab,ui]" websockets
```

The full setup, controls, troubleshooting and design status are in
[`docs/pong_learning_lab.md`](docs/pong_learning_lab.md).

### Tests

```bash
python -m unittest discover -s tests -v
```

## Branches

The repository has four branches on GitHub: `main` and three feature branches. All three
feature branches are already merged into `main`, so `main` contains their work.

| Branch | Purpose | Status |
|---|---|---|
| [`main`](https://github.com/bhagathchan1/DQN/tree/main) | Integrated project: DQN reproduction, DeepQ Learning Lab (Pong + Breakout), trained models, documentation | Default branch; use this one |
| [`feature/pong-learning-lab`](https://github.com/bhagathchan1/DQN/tree/feature/pong-learning-lab) | Two-player Pong lab UI, GPU Pong configs, two-player fine-tune script, trained Pong models | Merged into `main` |
| [`feature/reproducible-training-pipeline`](https://github.com/bhagathchan1/DQN/tree/feature/reproducible-training-pipeline) | Atari transition collection from saved policies, upstream Atari profile in the training pipeline, Colab import fix | Merged into `main` |
| [`feature/trained-models`](https://github.com/bhagathchan1/DQN/tree/feature/trained-models) | GPU training runs, Dueling DQN option, 15 Hz cadence, session resume information | Merged into `main` |

Pull requests should target `main`. The feature branches are kept for history.

## Repository map

| Path | Contents |
|---|---|
| `src/dqn_project/` | Model, learner, Atari preprocessing, replay (uniform and prioritized), training, evaluation, play, Pong adapter and lab server |
| `configs/` | Training profiles: Nature, Dueling, uniform, prioritized, upstream Breakout, Pong |
| `scripts/` | Training and smoke-test entry points, Pong probes, WSL2 lab setup, report builders |
| `web/` | React/TypeScript/Vite UI for the DeepQ Learning Lab (Pong, Breakout, trained models) |
| `trained_models/` | Trained Pong and Breakout checkpoints, configs and metrics CSVs |
| `results/` | Recorded experiment snapshot, metrics and plots |
| `tests/` | Unit and integration tests |
| `notebooks/` | Colab GPU workflow |
| `guides/` | Setup guide for watching a trained agent play |
| `modules/` | Team module guides (dataset, training, testing, UI) |
| `assets/upstream_dqn/` | Attributed reference visuals from the upstream implementation |

## Documentation

| Document | Covers |
|---|---|
| [`docs/paper_alignment.md`](docs/paper_alignment.md) | Paper vs implementation, element by element, with deviations |
| [`docs/experiment_protocol.md`](docs/experiment_protocol.md) | Matched-seed Breakout protocol |
| [`docs/upstream_comparison.md`](docs/upstream_comparison.md) | Comparison with the upstream implementation |
| [`docs/upstream_data_pipeline.md`](docs/upstream_data_pipeline.md) | Upstream-compatible data pipeline |
| [`docs/atari_lab.md`](docs/atari_lab.md) | Nature preset and desktop arena |
| [`docs/pipeline.md`](docs/pipeline.md) | Isolated multi-seed experiments |
| [`docs/architecture_notes.md`](docs/architecture_notes.md) | Architecture notes |
| [`docs/runtime_validation.md`](docs/runtime_validation.md) | Runtime validation records |
| [`docs/pong_learning_lab.md`](docs/pong_learning_lab.md) | DeepQ Learning Lab (Pong mode): setup, controls, design and status |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`REQUIREMENTS.md`](REQUIREMENTS.md) | DeepQ Learning Lab (Pong mode) architecture and requirements |
| [`modules/README.md`](modules/README.md) | Team module allocation and handoffs |

## Limitations

- Breakout runs use a ~1M-frame budget, not the paper's 50M frames per game, and only one game.
- Three seeds and 10 evaluation episodes per policy give descriptive results, not confidence intervals.
- The 10M-step Breakout numbers are training averages with exploration on, not greedy scores.
- The Pong lab learns between games; the paper learns online during play.
- Pong lab rewards include serve penalties, so returns and verified goal counts are reported separately.
- Prioritized replay is an extension and is not attributed to the 2015 paper.

## Citation and attribution

```bibtex
@article{mnih2015humanlevel,
  title   = {Human-level control through deep reinforcement learning},
  author  = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and others},
  journal = {Nature},
  volume  = {518},
  pages   = {529--533},
  year    = {2015},
  doi     = {10.1038/nature14236}
}
```

The Atari environments are provided by the Arcade Learning Environment and the
multi-agent Pong environment by PettingZoo. Reference clips and the network illustration
are from the upstream implementation; see [`assets/upstream_dqn/ATTRIBUTION.md`](assets/upstream_dqn/ATTRIBUTION.md).
