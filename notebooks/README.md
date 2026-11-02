# Notebooks

The tested Python modules and command-line entry points are the source of truth.
Exploratory notebooks may be added here, but they must import project modules rather
than duplicate model, replay, or preprocessing logic.

`07_colab_gpu_breakout.ipynb` runs the training and evaluation entry points on a Google
Colab GPU runtime, writes checkpoints and metrics to mounted Drive, and can export a
sample using the linked Atari data-preparation pipeline. The run keeps this project's
compact learner settings, so it is not an exact reproduction of the linked TensorFlow
implementation. Its checkpoints are model checkpoints, not full training-resume
snapshots. See `docs/upstream_data_pipeline.md` for the sample schema and limits.
