# Module 3: Testing and evaluation

**Owner: Member 3**

## Purpose and responsibilities

Verify component correctness and measure trained-agent performance independently
of training. Coordinate the shared UI module and its final demonstration.

- Maintain regression/integration coverage and coordinate fixes with code owners.
- Evaluate frozen checkpoints on held-out seeds using a fixed episode protocol.
- Compare uniform replay, prioritized replay and the random-action baseline.
- Report episode returns, mean/spread, seed variation and experiment limitations.
- Produce plots and tables from recorded outputs and verify their provenance.
- Validate UI score handling, checkpoint compatibility and completed/aborted rounds.

## Implementation ownership

- [evaluate.py](../../src/dqn_project/evaluate.py): checkpoint and baseline evaluation.
- [plot_results.py](../../src/dqn_project/plot_results.py) and
  [analyze_results.py](../../scripts/analyze_results.py): plots and analysis.
- [tests](../../tests): suite coordination; Members 1 and 2 maintain tests for their components.
- [experiment protocol](../../docs/experiment_protocol.md): agreed comparison procedure.
- [UI module](../04_ui/README.md): integration and acceptance.

## Inputs and outputs

Inputs: frozen checkpoints, matching configs, manifests/logs and held-out seeds.
Outputs: test results, evaluation JSON, comparison summaries, plots and a report
that states the actual training budget and observed outcomes.

Use the same environment, evaluation seeds, episode limits and evaluation epsilon
for compared agents. Keep raw episode scores for reporting; do not substitute
clipped learning rewards. Do not tune on final evaluation seeds or infer a benefit
from a single favorable run.

## Run and validate

```powershell
python -m unittest discover -s tests -v
python -m dqn_project.evaluate --config experiments/team-smoke/uniform_seed7/config.yaml --checkpoint experiments/team-smoke/uniform_seed7/checkpoints/uniform_seed7_final.pt --output experiments/team-smoke/uniform_seed7/recheck_dqn.json
python -m dqn_project.evaluate --config experiments/team-smoke/uniform_seed7/config.yaml --random-baseline --output experiments/team-smoke/uniform_seed7/recheck_random.json
```

The last two commands depend on Module 2's smoke run and demonstrate evaluation
mechanics only. Use completed full-run checkpoints and their matching configs for
the final report. The pipeline already writes DQN/random evaluation files and a
summary, so standalone rechecks are optional.

Completion requires passing regression checks, matched evaluations across the
agreed seeds, traceable plots/tables and a UI demo validated against the
[UI acceptance criteria](../04_ui/README.md#acceptance-criteria).
