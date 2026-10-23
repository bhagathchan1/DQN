import React from "react";
import { EvidenceChart, RichNarrative, useDataApp } from "../../data-app-public.jsx";

function EvaluationChart() {
  const { reviewedRows } = useDataApp();
  const rows = reviewedRows("seed_summary", ["policy", "seed"]);
  return <EvidenceChart id="evaluation-by-seed" queryId="seed_summary"
    title="Held-out reward varies by training seed"
    spec={{ type: "bar", x: "seed_label", y: "mean", series: "policy_label", yTickCount: 5, valueDecimals: 1 }}
    rows={rows} sourceRows={rows} height={330} />;
}

function TrainingChart() {
  const { reviewedRows } = useDataApp();
  const rows = reviewedRows("training", ["replay_type", "seed", "environment_steps"]);
  return <EvidenceChart id="training-reward-trends" queryId="training"
    title="Training reward is noisy; 50-episode rolling mean smooths the trace"
    spec={{ type: "line", x: "environment_steps", y: "rolling_reward_50", series: "series_label", yTickCount: 5, valueDecimals: 1 }}
    rows={rows} sourceRows={rows} height={380} />;
}

export function ReportContent() {
  return <article className="report-content report-page">
    <header className="report-hero">
      <p className="report-deck">A matched-seed, compute-limited reproduction of the Atari DQN paper’s replay comparison.</p>
      <RichNarrative id="dqn-results-lede" value={`# Breakout DQN: what the runs show\n\nSix trained agents, three training seeds, and 60 held-out evaluation episodes. At this budget, uniform replay beats the random baseline consistently; prioritized replay is less consistent against uniform.`} />
    </header>

    <section>
      <RichNarrative id="dqn-results-topline" value={`## The result, in one read\n\nUniform replay averaged **3.5 points per episode** across the three seed means, versus **1.0** for random. Prioritized replay averaged **2.3**. The paired prioritized-minus-uniform differences were +2.1, −1.1, and −4.7 points for seeds 7, 17, and 29; the mean paired difference was −1.2. With only three seeds and short training, this is descriptive evidence—not a reliable ranking of replay algorithms.`} />
      <EvaluationChart />
    </section>

    <section>
      <RichNarrative id="dqn-training-interpretation" value={`## Learning curves: substantial run-to-run variation\n\nEach curve is one 250,000-action training run. The plotted measure is the 50-episode rolling mean of the raw, unclipped game score during epsilon-greedy training—not the held-out score. Curves are noisy and do not establish convergence.`} />
      <TrainingChart />
    </section>

    <section>
      <RichNarrative id="dqn-method-and-caveats" value={`## What was reproduced—and what was not\n\nThe implementation follows the paper’s core DQN setup and compares uniform with prioritized replay under matched budgets and fixed seeds. Evaluation uses epsilon 0.05 and up to 4,500 selected actions per episode (four emulator frames per action), with ten episodes per policy and seed.\n\nThis is a **compute-limited reproduction**, not a scale-faithful rerun of the Nature paper: each agent trained for 250,000 selected actions rather than the paper’s 50 million emulator frames. The smaller budget, three seeds, and short observed scores limit conclusions. Full setup and deviations are documented in docs/paper_alignment.md; raw run and episode-level files are included with the project.\n\n**Execution record:** macOS 26.5.1 (arm64), CPU-only; Python 3.11.15, PyTorch 2.14.1, Gymnasium 1.4.0 and ALE-py 0.12.1.`} />
    </section>
  </article>;
}
