# Running your own DQN experiments

This branch adds experiment orchestration to the existing PyTorch implementation.
The default is the uniform-replay DQN baseline from Mnih et al. (2015).
Prioritized replay remains an optional extension, not part of the original paper.

## Windows setup

Run from the repository directory with Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[all]'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

An isolated `.venv` has already been created in this checkout. If `python` is not
on PATH, use an installed Python executable to create it. Activation is optional.
Linux/macOS use `.venv/bin/python` in place of the Windows executable.

## First verify the complete pipeline

```powershell
.\.venv\Scripts\python.exe -m dqn_project.pipeline --config configs/default.yaml --output experiments/smoke --smoke --device cpu
```

This performs 64 real Atari action decisions, replay sampling and optimization,
saves a checkpoint, reloads it, and evaluates DQN and random policies for two
episodes capped at 100 decisions. It validates execution only; scores from this
run do not demonstrate learning. Each output path must be new.

## Train and test

```powershell
.\.venv\Scripts\python.exe -m dqn_project.pipeline --config configs/default.yaml --output experiments/uniform --seeds 7 17 29
.\.venv\Scripts\python.exe -m dqn_project.pipeline --config configs/prioritized.yaml --output experiments/prioritized --seeds 7 17 29
```

Runs execute sequentially. Each default run takes 250,000 action decisions,
approximately one million emulator frames. The full-state replay buffer alone
uses about 538 MiB at capacity 10,000; do not raise it to one million without
changing the storage design (that would require about 53 GiB for observations).
Choose `--device cpu` or `--device cuda` explicitly, or let `auto` select a device
available to your PyTorch installation. The runner defaults OMP/MKL threads to
one unless you already set those variables.

The pipeline takes a snapshot of the selected YAML and overrides only run ID,
training seed, held-out evaluation seed, and explicit CLI options. Smoke mode
also reduces budgets. Customize the environment and hyperparameters by copying
the YAML. There is no pretrained-weight dependency: training starts from random
weights. Checkpoints contain model/optimizer state, but this pipeline does not
resume replay or environment state after interruption; rerun in a new directory.

## Outputs

Each experiment has `manifest.json` (status, source revision, dirty-tree flag,
platform, seeds and errors), `requirements-lock.txt`, and `summary.csv` with
per-training-seed DQN/random test statistics. The lock file is an environment
snapshot from pip freeze, not a portable cross-platform dependency resolution.

Each seed directory contains:

- `config.yaml`: actual parameters used.
- `checkpoints/`: periodic and final model checkpoints.
- `results/metrics/`: raw training episode rewards and losses.
- `dqn_evaluation.json`, `random_evaluation.json`: raw test returns and summaries.
- `train.log`, `dqn.log`, `random.log`: stage stdout/stderr.
- `training_reward.png` and its summary: generated if training episodes finished.

Evaluation uses training seed + 10,000, with episode seeds incremented from there.
Both policies share evaluation seeds and caps. Evaluation reports raw rewards;
training clips rewards for learning only. Seed separation does not compensate for
the inherited lack of randomized no-op resets: see `paper_alignment.md` for this
and other protocol differences. These are course-scale runs, not reproduction of
the Nature scores. The checked-in results belong to the original repository;
your new runs live under ignored `experiments/` and do not reuse those results.

On failure inspect the manifest and the failing stage log. Existing output
directories are refused so experiments cannot silently overwrite each other.

## Live demonstration and independent evaluation

After completing a run:

```powershell
.\.venv\Scripts\python.exe -m dqn_project.evaluate --config experiments/uniform/uniform_seed7/config.yaml --checkpoint experiments/uniform/uniform_seed7/checkpoints/uniform_seed7_final.pt --output experiments/uniform/uniform_seed7/demo.json --render
```

Omit `--render` for headless testing. Evaluation rejects a checkpoint whose stored
game differs from the config. Use checkpoints you generated or trust.

## Connection to the paper and your notes

The supplied handwritten notes identify the key stabilizers: uniform experience
replay and a periodically copied target network. They also specify the 84x84x4
input, three convolutional layers and 512-unit hidden layer. These live in
`replay_buffer.py`, `learner.py`, `environment.py` and `model.py` respectively.
`train.py` connects interaction, epsilon-greedy actions, replay and updates;
`pipeline.py` now connects training, checkpoint testing, baseline and reporting.
The notes' final demo requirement is supported by `evaluate --render`.

The external [Jihoon Kim reference implementation](https://github.com/jihoonerd/Human-level-control-through-deep-reinforcement-learning)
uses TensorFlow 2 and the older Gym NoFrameskip interface. This branch reuses your
projectmate's existing PyTorch/Gymnasium code rather than introducing a second
framework. No source code from that reference repository was copied.
The [paper](https://doi.org/10.1038/nature14236) and local PDFs are the conceptual
references. See `paper_alignment.md` for the existing implementation's departures
from the original training/evaluation protocol.
