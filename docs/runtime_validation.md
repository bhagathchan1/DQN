# Runtime validation record

Validated on 19 October 2026 on the local macOS arm64 development machine.

## Installed environment

- Python 3.11.15
- NumPy 2.4.6
- PyTorch 2.14.1
- Gymnasium 1.4.0
- ALE-py 0.12.1
- Pillow 13.0.0
- PyYAML 6.0.3

## Checks completed

- All 15 automated tests passed in the project `.venv`.
- `ALE/Breakout-v5` reset and step succeeded with four legal actions.
- Processed states and successor states were `uint8` arrays of shape `[4,84,84]`.
- A 64-decision smoke configuration completed collection, replay sampling, optimizer
  steps, periodic target synchronization, metrics file creation and checkpoints.
- The final smoke checkpoint loaded successfully and completed one evaluation episode.
- The random-baseline path completed one evaluation episode with the same wrapper.
- Plot generation produced a PNG learning curve and CSV summary from logged values.

The one-episode smoke returns are deliberately excluded from project results: the run
is far too short and exists only to validate plumbing. Meaningful results require the
matched full configurations and multiple seeds described in the protocol.
