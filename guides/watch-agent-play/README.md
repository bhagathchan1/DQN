# Watch a trained DQN agent play Atari

This guide runs a saved DQN policy in Breakout. On a computer with a desktop display,
`dqn-play` opens a live game window and the checkpoint controls the actions. In Google
Colab, use the GIF workflow instead; Colab does not open a desktop game window.

The example checkpoint is the episode 14,000 Breakout checkpoint collected from the
Colab run. It is a local model file and is intentionally not committed to GitHub.

## 1. Get the project and checkpoint

If you already cloned the project, open a terminal in its root folder and update it:

```bash
git status --short --branch
git pull --ff-only origin main
```

Commit or otherwise preserve any local edits before pulling. If you do not have the
project yet, clone it:

```bash
git clone https://github.com/bhagathchan1/DQN.git
cd DQN
```

Download or copy the checkpoint into the project's `checkpoints/` directory. For the
checkpoint from the 14,000-episode Colab run, the expected path is:

```text
checkpoints/breakout_upstream_port_colab_seed7_episode_14000.pt
```

Check that the file is there before continuing:

macOS or Linux:

```bash
ls -lh checkpoints/breakout_upstream_port_colab_seed7_episode_14000.pt
```

Windows PowerShell:

```powershell
Get-Item .\checkpoints\breakout_upstream_port_colab_seed7_episode_14000.pt
```

Checkpoint files are ignored by Git. Transfer them separately through Drive or another
file-sharing method; do not add them to a Git commit.

## 2. Install the game and display dependencies

Use Python 3.10 or newer. From the project root, create and activate a virtual
environment, then install the environment extras. These include Gymnasium/ALE for the
Atari environment and Pygame for the live display.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[env]'
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[env]'
```

The game environment uses `ALE/Breakout-v5`. Atari ROM availability depends on the
installed ALE package and applicable ROM terms. The repository does not store ROM
files.

## 3. Start the live game

Run this from the project root with the virtual environment activated:

```bash
dqn-play \
  --checkpoint checkpoints/breakout_upstream_port_colab_seed7_episode_14000.pt \
  --env ALE/Breakout-v5 \
  --episodes 1 \
  --epsilon 0 \
  --device auto
```

A Breakout window should open. The agent selects actions from the checkpoint's learned
policy and plays one full game. The terminal prints the score and number of selected
actions when the episode ends. Press **Ctrl+C** in the terminal to stop early.

`--epsilon 0` makes the agent choose its highest-valued action each turn. To add
exploration, use a value such as `--epsilon 0.1`. For a shorter preview, add
`--max-actions 1000`; each selected action is repeated across emulator frames by the
Atari preprocessing pipeline.

`--device auto` selects CUDA when PyTorch detects an NVIDIA GPU, Apple MPS when
available, or CPU otherwise. A GPU is not required to watch the agent play.

## 4. If you are using Google Colab

Colab runs without a normal desktop display. Open
[`notebooks/07_colab_gpu_breakout.ipynb`](../../notebooks/07_colab_gpu_breakout.ipynb),
mount Drive, and run its evaluation cell after training. It evaluates the final
checkpoint, saves a GIF under `MyDrive/DQN/results/evaluations/`, and displays the GIF
in the notebook output.

If you already have a checkpoint in Drive and want to make a GIF without running
training again, set the `checkpoint` variable in the evaluation cell to that file and
run the cell. For example:

```python
checkpoint = DRIVE_DIR / 'checkpoints' / 'breakout_upstream_port_colab_seed7_episode_14000.pt'
```

## Troubleshooting

- **`dqn-play: command not found`** — activate the virtual environment and rerun
  `python -m pip install -e '.[env]'` from the project root. As an alternative, run
  `python -m dqn_project.play --checkpoint checkpoints/breakout_upstream_port_colab_seed7_episode_14000.pt --env ALE/Breakout-v5`.
- **`No module named dqn_project`** — confirm the terminal is in the project root and
  reinstall with `python -m pip install -e '.[env]'`.
- **No game window appears** — run the command on a computer with a desktop session.
  A headless SSH session and standard Colab runtime cannot show the live window; use
  the GIF option in Colab.
- **ALE or Breakout cannot be created** — confirm the environment installation
  completed, then run `python scripts/environment_smoke_test.py --pipeline upstream --env ALE/Breakout-v5`.
- **The agent plays poorly** — the checkpoint came from a partial training run. The
  live viewer shows what that checkpoint learned; it does not improve the policy.
