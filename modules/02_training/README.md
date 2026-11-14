# Module 2: Training pipeline

**Owner: Member 2**

## Purpose and responsibilities

Turn Module 1's online experience into trained DQN checkpoints and reproducible runs.

- Maintain the Q-network, target network, Bellman update, optimizer and exploration.
- Integrate collection, replay sampling, priority feedback and target synchronization.
- Maintain configuration handling, device selection, checkpointing and metric logs.
- Own orchestration of train, checkpoint evaluation and random-baseline stages.
- Run uniform/prioritized experiments with matched budgets and seeds.
- Supply inference/checkpoint integration support to the UI owner.

## Implementation ownership

- [model.py](../../src/dqn_project/model.py), [learner.py](../../src/dqn_project/learner.py),
  [optimizer.py](../../src/dqn_project/optimizer.py): network and learning.
- [agent.py](../../src/dqn_project/agent.py): action selection and exploration.
- [train.py](../../src/dqn_project/train.py), [pipeline.py](../../src/dqn_project/pipeline.py): execution and orchestration.
- [config.py](../../src/dqn_project/config.py), [logging_utils.py](../../src/dqn_project/logging_utils.py)
  and [configs](../../configs): configuration and training logs.

Member 3 owns evaluation correctness even when evaluation is launched by the pipeline.

## Inputs and handoff

Inputs: environment/replay interfaces, training configuration, seeds and compute
budget. Outputs: `.pt` checkpoints, resolved configs, CSV training logs,
pipeline manifest, evaluation summaries and captured dependency versions.

For example, the pipeline below writes a final checkpoint under
`experiments/team-smoke/uniform_seed7/checkpoints/uniform_seed7_final.pt`.
Hand over the matching `config.yaml`, game/action metadata, training budget and
logs with each checkpoint. Keep output directories unique; the pipeline rejects
an existing output directory.

## Run and validate

```powershell
python -m dqn_project.pipeline --config configs/default.yaml --output experiments/team-smoke --seeds 7 --smoke --device cpu
python -m dqn_project.pipeline --config configs/default.yaml --output experiments/team-uniform --seeds 7 17 29
python -m dqn_project.pipeline --config configs/prioritized.yaml --output experiments/team-prioritized --seeds 7 17 29
```

Start with the smoke command; full runs require an agreed compute budget. See the
[pipeline guide](../../docs/pipeline.md) for run layout and operational details.

```powershell
python -m unittest discover -s tests -p test_model.py -v
python -m unittest discover -s tests -p test_learning_update.py -v
python -m unittest discover -s tests -p test_pipeline.py -v
```

Completion requires a finite learning update that changes online parameters,
correct target synchronization, checkpoint save/load, and a successful smoke
pipeline. Deliver actual full-run artifacts separately; a working pipeline alone
does not establish a strong trained agent.
