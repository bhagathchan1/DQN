# Notebooks

The tested Python modules and command-line entry points are the source of truth.
Exploratory notebooks may be added here, but they must import project modules rather
than duplicate model, replay, or preprocessing logic.

`07_colab_gpu_breakout.ipynb` runs the upstream-compatible training profile on a Google
Colab GPU runtime, writes checkpoints, evaluation GIFs, metrics, and TensorBoard logs
to mounted Drive, and can export a sample of online transitions. The model and
environment interfaces are ported to PyTorch/Gymnasium, so results will not be
bit-for-bit identical to the linked TensorFlow/Gym implementation. Periodic checkpoints save model and optimizer state, but omit replay and environment state,
so they cannot resume an interrupted run exactly. See `docs/upstream_data_pipeline.md`.
