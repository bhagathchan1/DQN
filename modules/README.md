# Team modules and responsibilities

## Allocation

Three members own three core modules. A fourth module provides the shared
human-versus-agent UI, with Member 3 coordinating integration.

| Module | Accountable owner | Main deliverable |
|---|---|---|
| [1. Dataset and experience generation](01_dataset/README.md) | Member 1 | Reproducible environment, processed observations and replay batches |
| [2. Training pipeline](02_training/README.md) | Member 2 | Configured training runs, checkpoints and training metrics |
| [3. Testing and evaluation](03_testing/README.md) | Member 3 | Verified behavior, held-out evaluation and comparison report |
| [4. Competitive UI](04_ui/README.md) | Member 3, supported by Members 1 and 2 | Playable human/AI rounds and saved score comparisons |

These are logical ownership boundaries over `src/dqn_project`, not separate
installable packages. Keep the existing imports and command-line entry points.

## How the modules connect

1. Member 1 supplies environment observations and replay batches to Member 2.
   In DQN, collection and training alternate: the current agent generates new
   transitions, which become its evolving training dataset.
2. Member 2 supplies checkpoints, resolved configurations and logs to Member 3.
3. Member 3 evaluates frozen checkpoints on held-out seeds and publishes results.
4. The UI uses Member 1's environment, Member 2's checkpoint and Member 3's
   comparison rules to let a human compete with the trained agent's score.

There is no required pre-generated static dataset or offline dataset loader in
the current training workflow. Replay storage is the training data interface.
An exported trajectory dataset would be a future extension, requiring an agreed
format and a loader before it could replace online collection.

## Shared contracts and handoffs

The [project specification](../docs/SPEC.md) defines the binding tensor and replay
interfaces; the [experiment protocol](../docs/experiment_protocol.md) defines the
scientific comparison rules.

| Handoff | Required information | Reviewer |
|---|---|---|
| Dataset to training | Environment ID, action count, preprocessing, seeds; `uint8 [4,84,84]` states and `ReplayBatch` | Member 2 |
| Training to testing | Checkpoint, matching config, game, replay type, training seed/budget and logs | Member 3 |
| Training to UI | Checkpoint with environment/action metadata and inference settings | Member 3 |
| Testing to team | Commands, actual results, baseline comparison, limitations and pass/fail evidence | Members 1 and 2 |

Member 1 owns environment/replay behavior; Member 2 owns learning and training
orchestration; Member 3 owns evaluation, analysis and UI integration. Changes to
shared config, checkpoint formats or preprocessing require review from the affected
owner. Member 2 maintains the combined train/evaluate pipeline; Member 3 maintains
the evaluation logic it calls.

## Delivery sequence

1. Agree on one game, state/action contract, seeds, budget and evaluation protocol.
2. Validate collection and replay with synthetic observations and an environment smoke test.
3. Complete a short end-to-end training/checkpoint/evaluation smoke run.
4. Run matched uniform/prioritized experiments and evaluate held-out episodes.
5. Integrate the checkpoint into the UI and demonstrate paired human/AI rounds.

All commands in the module READMEs run from `DQN/`, after installing the package
using the [main setup instructions](../README.md). Smoke runs validate plumbing;
they are not evidence of learned performance. Each member contributes their module's
report section, test evidence and demo explanation.
