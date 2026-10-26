# Notebooks

The tested Python modules and command-line entry points are the source of truth.
Exploratory notebooks may be added here, but they must import project modules rather
than duplicate model, replay, or preprocessing logic.

`07_colab_gpu_breakout.ipynb` runs the existing training and evaluation entry points
on a Google Colab GPU runtime and places checkpoints and metrics on mounted Drive.
It runs this project's compact PyTorch experiment, not an exact reproduction of the
linked TensorFlow implementation. Its checkpoints are model checkpoints, not full
training-resume snapshots.
