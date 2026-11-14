# Speaking script: Deep Q-Networks, 15 minutes

Three presenters: **A** (learning problem, opening, close), **B** (architecture, replay and target network, paper evidence, paper-vs-implementation), **C** (our Pong lab and results, limits). Replace `[Presenter A/B/C]` and `[Date]` on slide 1. Deck: `DQN_Pong_15min.pptx` (speaker notes hold the same timing and sources).

| Slide | Time | Presenter | Title |
|---|---|---|---|
| 1 | 0:00–0:20 | A | Title |
| 2 | 0:20–0:45 | A | Agenda and roles |
| 3 | 0:45–2:00 | A | Agent and environment |
| 4 | 2:00–3:00 | A | Bellman target and DQN loss |
| 5 | 3:00–4:30 | B | Network architecture |
| 6 | 4:30–6:30 | B | Replay and target network |
| 7 | 6:30–8:00 | B | Paper: 49 games |
| 8 | 8:00–9:00 | B | Paper: ablation and Breakout strategy |
| 9 | 9:00–10:30 | C | Our Pong Learning Lab |
| 10 | 10:30–11:45 | C | Pong results |
| 11 | 11:45–12:45 | C | Breakout reproduction |
| 12 | 12:45–13:45 | B | Paper vs implementation |
| 13 | 13:45–14:30 | C | Limits and next steps |
| 14 | 14:30–15:00 | A | Summary and questions |

---

## Presenter A

**Slide 1 (20 s).** Good [morning/afternoon]. Today we cover how a Deep Q-Network learns to play from pixels, what the 2015 Nature paper showed, and what we built and measured in this repository. I'm [Presenter A]; [Presenter B] and [Presenter C] are next.

**Slide 2 (25 s).** Here is the plan. I'll cover the learning problem; [B] covers the network, the two stabilizers and the paper's evidence; [C] covers our Pong lab and results; then [B] compares the paper with our implementation, and [C] states the limits.

**Slide 3 (75 s).** Reinforcement learning is a loop. At each step the agent observes a state, picks an action, and the environment returns a reward and a new state. The agent's job is to maximize the total discounted reward. In Atari the state is raw pixels, so the agent gets no hand-written game features. In the paper, the state is four stacked 84 by 84 frames, and the agent acts once every four emulator frames. Our Pong configs use the same six actions.

**Slide 4 (60 s).** The key quantity is the action value, Q of s and a: the expected return after taking a in s and then acting well. The Bellman optimality equation says the best value equals the immediate reward plus the discounted best value of the next state. DQN turns that into a regression target. The network predicts Q of s and a, and the target uses a separate, older copy of the network, which [B] will explain. The loss is the squared difference between the two. For a terminal transition the target is just the reward.
*Handoff:* "[B] will show the network that computes these values."

---

## Presenter B

**Slide 5 (90 s).** The input is four stacked grayscale frames, 84 by 84. Three convolutional layers reduce it to a 7 by 7 by 64 feature map, which feeds a 512-unit layer. The output has one value per action, so one forward pass gives the score of every action. Activations are ReLU. The shapes follow from the filter sizes: 84 minus 8 over 4, plus 1, gives 20; then 9; then 7.

**Slide 6 (120 s).** Two components make this stable. The first is experience replay. We store each transition in a buffer of up to one million frames and train on random minibatches. Random sampling breaks the correlation between consecutive frames. The second is the target network. The target depends on the weights we are updating, so it would move with every step. We copy the online weights into a frozen target network every 10,000 updates and compute the target with that copy. Our Nature config uses the same numbers: a 1,000,000-frame buffer, batch size 32, and a target update every 10,000 updates.
*Handoff:* "[B] continues with the paper's results."

**Slide 7 (90 s).** The paper uses one algorithm, one architecture and one set of hyperparameters across 49 Atari games. The agent receives only pixels and the game score. The paper reports that DQN achieves more than 75% of the professional tester's score on more than half of the games, 29 games. The normalized score is shown on the slide: 100 times DQN minus random, over human minus random. Be clear that this is the paper's protocol, with 50 million frames per game and 30 evaluation episodes, not our budget.

**Slide 8 (60 s).** The paper also tests whether the components matter. It disables replay and the target network, alone and together, and reports a detrimental effect on performance. The full table is Extended Data Table 3 with 10 million frames per agent. We are quoting the claim here, not the numbers. The paper also describes a learned Breakout strategy: digging a tunnel around the side of the wall so the ball can reach the back and clear many bricks.
*Handoff:* "[C] will show what we built on top of this."

---

## Presenter C

**Slide 9 (90 s).** We built the Pong Learning Lab. You play one paddle against the DQN's paddle in a two-player Atari Pong match. The machine's weights are frozen during the game. After each completed game it learns from its replay for 450 updates, then the new version plays the next game. It decides 15 times per second, one action per four frames, as in the paper. This is a demo adaptation: the paper learns online, while the lab learns between games.

**Slide 10 (75 s).** The single-player Nature agent reached a validation mean of +19.1 after 10 million steps; the Dueling agent reached +17.2. The two-player models were fine-tuned for 3 million steps from those weights and reached +21.0 and +19.0 in 8-game evaluations. Pong is scored to 21, so these are strong results, but the protocols differ, so do not read the bars as a ranking.

**Slide 11 (60 s).** This is a small Breakout reproduction: three seeds per replay method, about one million frames each, and 10 held-out episodes per policy. Uniform replay beat random on all three seeds: 2.4 against 1.0, 1.9 against 0.8, and 6.2 against 1.2. Prioritized replay did not beat uniform consistently; the paired differences were +2.1, minus 1.1 and minus 4.7. With three seeds we report the replay ranking as unsettled.
*Handoff:* "[B] puts the paper and our implementation side by side."

---

## Presenter B

**Slide 12 (60 s).** The left column is the paper; the right is what we ran. The mechanism matches: a convolutional network on four frames, replay, a target network, one-step targets and epsilon-greedy exploration. What changes is the budget, the games and the schedule. The biggest change for the Pong lab is that learning happens after each game rather than online.
*Handoff:* "[C] covers the limits."

---

## Presenter C

**Slide 13 (45 s).** Three seeds are too few for a replay-method claim, and the 10-million-step Breakout runs report training averages with exploration on, so they are not greedy scores. In the lab, rewards include serve penalties, so we show returns and only verified goals. Next steps are greedy evaluation of the 10-million-step checkpoints, more seeds with confidence intervals, and verified goal counting in the lab.
*Handoff:* "[A] closes."

---

## Presenter A

**Slide 14 (30 s, then questions).** To summarize: DQN combines a convolutional network, replay and a target network to learn Q-values from pixels. The paper reports strong results across 49 games. Our Pong lab shows the same mechanism training an opponent a person can play against, and our Breakout reproduction shows learning at a small budget. We have not reproduced the full benchmark, and we have not settled the replay-method ranking. Thank you. We're happy to take questions.

If a question goes beyond the sources, say so and offer to check the paper or the repository.

---

## Sources

- Mnih et al., *Human-level control through deep reinforcement learning*, Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236
- Hugging Face Deep RL course, unit 3: https://huggingface.co/learn/deep-rl-course/en/unit3/deep-q-algorithm
- PettingZoo Atari Pong: https://pettingzoo.farama.org/environments/atari/pong/
- Prioritized replay (extension): https://arxiv.org/abs/1511.05952
- Dueling networks: https://arxiv.org/abs/1511.06581
- Repository: `docs/paper_alignment.md`, `trained_models/README.md`, `results/reviewed_experiment_snapshot.json`
