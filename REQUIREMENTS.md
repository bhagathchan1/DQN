# Pong demo requirements for approval

**Draft v0.2. Implemented and smoke-tested; formal approval not yet recorded and
requirements are not yet frozen. The acceptance scenarios below have not all been run.**

This supersedes the v0.1 multi-game dashboard proposal. See
[ARCHITECTURE.md](ARCHITECTURE.md) for design and [README.md](README.md) for overview.
This file defines behavior and acceptance, not package installation dependencies.

## Review decisions

| Area | Proposed decision |
| --- | --- |
| Game | Atari Pong with simultaneous human and machine players |
| Game boundary | Environment termination or 120 seconds simulation time; a point is not a game |
| Main loop | Play -> result/compare -> automatic learning -> summary -> Play again |
| Training | Vanilla DQN, machine transitions, cumulative uniform replay, updates after game only |
| Learning budget | 450 updates, batch 32, minimum 256 sampleable transitions |
| State retained | Online/target weights, optimizer, replay, RNG and schedules across games |
| UI | Local FastAPI + React/TypeScript/Vite application |
| Initial policy | From scratch, or explicit compatible model import with origin displayed |
| Comparison | Previous valid match and session history; model versions/protocol shown |
| Approval record | Pending; no implementation authorized by this draft itself |

On approval, record the accepted version/date and any changed defaults as v1.0.
Further changes name the affected requirement IDs and revise acceptance criteria.

## Mandatory v1 requirements

| ID | Requirement | Acceptance evidence |
| --- | --- | --- |
| GAME-01 | Run a real two-player Atari Pong match with human and DQN controlling opposing paddles. | Both controls are exercised in the same live environment; side/action mapping is verified and saved. |
| GAME-02 | Freeze machine weights and epsilon for the duration of each match. | No optimizer steps or active-version changes occur between match start and end. |
| GAME-03 | End on environment termination or the declared simulation-time cap. | A rally/point does not launch learning; a valid time-capped match proceeds to learning with truncation recorded. |
| GAME-04 | Always show per-player raw environment return; show distinct goal/serve-penalty counters only when verified. | Goal and serve-timeout cases preserve the actual per-player rewards. If event counts are unavailable, mark them unavailable and show a return leader/draw, not a verified game winner; never assume human return equals negative machine return. |
| GAME-05 | Target 60 emulator frames/s with one four-frame repeat per decision and show actual cadence. | A test detects doubled frame skip; sustained throughput below 90% for over five seconds invalidates/aborts the match. |
| GAME-06 | Release keys on focus loss; disconnect or 500 ms stale input aborts the match. | Test confirms no stuck control and no automatic learning from that aborted match. |
| DATA-01 | Automatically collect machine state, action, reward, successor and terminal/truncation information during play. | Records match executed machine actions; no human annotation or downloaded dataset is required. |
| DATA-02 | Stage match experience and commit it once only for valid completed or time-capped matches. | Duplicate completion/reconnect cannot append twice; aborted/invalid records stay out of replay. |
| DATA-03 | Keep a bounded replay across games, preserving match/reset boundaries and collection model versions. | Three games contribute to the same replay; stack reconstruction cannot cross a match reset or overwritten history. |
| DATA-04 | Retain raw reporting rewards and sign-clipped training rewards, without hidden shaping. | Sample transitions agree with environment output and the declared transform. |
| LEARN-01 | Automatically start a bounded learning phase after each eligible game when replay is sufficient. | Completed match creates one phase; updates never run during gameplay. |
| LEARN-02 | Use the existing 4x84x84 CNN, vanilla DQN Bellman target, Huber loss, uniform replay and frozen target network. | Terminal/nonterminal target tests and a finite weight-changing update pass. |
| LEARN-03 | Train only from machine actions/rewards; human controls are opponent context. | A unit/integration test proves human action indices are not substituted as the machine's training actions. |
| LEARN-04 | Keep learner/target/optimizer, replay, RNG and cumulative schedules across games. | Game 2 starts from Game 1's committed learner state rather than freshly initialized networks or optimizer. |
| LEARN-05 | Use proposed defaults: replay 20,000, minimum 256 sampleable entries, batch 32, 450 updates, gamma 0.99, Nature RMSProp lr 0.00025. | Actual settings and update counts are saved and shown; overrides are validated and identify a new comparison protocol. |
| LEARN-06 | Copy targets every 1,000 cumulative optimizer updates; use match-frozen epsilon from a 1.0-to-0.05, 10,000-committed-decision schedule. | Counters cross game boundaries without reset; target copies and epsilon values match the configured schedule. |
| LEARN-07 | If replay is insufficient, show progress toward 256 entries and perform zero updates. | No false trained status/model version is produced; Play again stays available. |
| LEARN-08 | Promote candidate weights/optimizer/target state only after a successful learning phase. | Failure/cancel leaves active model intact; explicit retry starts from the correct pre-phase state. |
| VIEW-01 | Show live learning progress, loss, absolute TD error, mean Q, replay use, elapsed time and target-copy events. | Charts come from persisted real optimizer events, including a no-update empty state. |
| VIEW-02 | Show input/output model versions and fixed-probe before/after Q/TD diagnostics. | Identical probe samples and pre-phase target network are used for both diagnostic calculations; display labels them as training diagnostics. |
| VIEW-03 | Provide optional processed-frame and action-Q inspectors. | Four ordered machine input frames and chosen action/Q-values agree with the model input used during play. |
| COMP-01 | Compare the current result with the preceding valid match in the same session/protocol. | First game has a clear empty comparison; aborted/invalid attempts are skipped, never overwritten. |
| COMP-02 | Display both model versions, outcomes, scores/returns, duration and current-minus-previous deltas. | Game 2 using v1 compares with Game 1 using v0; the later v2 update is not credited for Game 2. |
| COMP-03 | Show session history and trends without promising improvement after every match. | Changing exploration, human behavior and protocol are visible; new protocol starts a new series. |
| UI-01 | Provide one Play -> Result -> Learning -> Summary -> Play again journey. | User completes three cycles without editing files or restarting the model. |
| UI-02 | Disable gameplay during learning and keep status/cancel controls responsive. | No parallel match can launch against partially updated weights; worker does not block the API. |
| UI-03 | Show ready/playing/learning/insufficient-data/completed/failed/cancelled/interrupted states. | Each failure/recovery path has truthful status, accessible logs and an explicit next action. |
| SYS-01 | Persist versioned session, match, replay-commit, training and model lineage records. | Each result maps to the exact policy used and each promoted checkpoint maps to one successful phase. |
| SYS-02 | Restore the last committed between-game state; refresh cannot duplicate a match/learning job. Bound retained transition payloads to replay capacity plus reconstruction history. | Learner/replay restore consistently; unfinished phases are uncommitted and unfinished games aborted. Compaction retains scalar history and never removes the only recoverable snapshot. |
| SYS-03 | Enforce checkpoint/replay compatibility across environment version, action map, player side and preprocessing. | An incompatible single-agent Pong checkpoint is rejected even if output dimension matches. |
| SYS-04 | Preserve existing single-agent CLI and desktop tools while adding the new adapter/service. | Existing regression checks and a new two-player smoke workflow pass. |

## Frozen contracts proposed for approval

- Machine observations: uint8 `[4,84,84]`, oldest processed frame first, byte input
  normalized by 255; RGB frames are used for display.
- Machine transition fields: state, action, raw reward, training reward, successor,
  terminated, truncated, match ID, policy version, step index and opponent ID.
  Human actions may be logged separately; they are not correct-action labels.
- Bootstrap mask: real game termination disables the target; a demo time cap
  truncates while retaining the real successor. A point alone does not terminate.
- Environment settings, actual action map and side mapping are verified during
  adapter integration and persisted. Apply frame repeat exactly once.
- The replay and learning clock advance from committed valid experience and
  completed optimizer updates; all frame/decision/update units stay distinct.
- New sessions start from random weights unless a compatible checkpoint is
  selected explicitly. Initial weak/random play and skipped learning are labelled.
- Model lineage: match N uses vK; post-match learning may publish vK+1 for the next
  match. A skipped/failed phase retains vK. Match history preserves this mapping.
- Existing per-player environment rewards define training. A verified scoring
  adapter exposes goals/serve penalties separately; if unavailable, the UI clearly
  labels cumulative reward and its return leader/draw, with event counts unavailable.

## Acceptance scenarios

1. Start from scratch, play one complete match, see experience committed, observe
   real updates or an explicit insufficient-data state, and save the phase outcome.
2. Play two more games using the retained learner/replay; verify chronology and
   last-valid-game comparisons in both UI and exported records.
3. Exercise a scored point, serve penalty, environment terminal and time cap;
   verify rewards and DQN bootstrap behavior. Check event counters if verified,
   otherwise the explicit return-only fallback and outcome label.
4. Abort/disconnect a match and confirm no automatic learning/replay commit.
5. Fail/cancel a learning phase, keep the active model intact, retry explicitly
   and confirm only a successful candidate is promoted.
6. Refresh/restart between games and restore the committed learner/replay/counters
   without duplicated data or updates. Reject an incompatible imported checkpoint.

Use mathematical unit tests, worker/integration tests, real two-player Atari
smoke tests and browser keyboard/reconnect tests. Learning diagnostics demonstrate
actual parameter updates; gameplay improvement requires measured outcomes over
many matches and, for causal claims, a separate fixed-opponent evaluation.

## Outside the revised demo scope

Breakout/other games; Double DQN or new RL algorithms; external dataset import;
the earlier standalone data/training/benchmark dashboards; self-play; public hosting;
remote GPU scheduling; exact recovery within a match or partly executed training
phase; and formal reproduction of the paper's single-agent scores. Existing CLI
features remain separate. No promise is made that every training phase improves
the next score or that a few human games produce a strong policy.
