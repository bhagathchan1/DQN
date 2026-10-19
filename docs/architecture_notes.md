# DQN architecture and learner notes

DQN approximates the optimal action-value function directly from pixels. A state is
the four most recent preprocessed frames, which provides short-term motion information
that a single still image lacks. One network forward pass produces a Q-value for every
legal action, so selecting the greedy action is an `argmax` over the output vector.

The visual model matches the paper's architecture: 32 filters of size 8 x 8 at stride
4, 64 filters of size 4 x 4 at stride 2, 64 filters of size 3 x 3 at stride 1, then a
512-unit fully connected hidden layer. ReLU follows each hidden layer; the action
outputs are linear because Q-values are not probabilities.

Training uses transitions sampled from replay instead of only the latest consecutive
experience. This improves data reuse and weakens the temporal correlations in each
minibatch. For selected action `a`, the online estimate is `Q_online(s, a)`. The target
is `r` at a true terminal transition and otherwise
`r + gamma * max_a' Q_target(s', a')`. Gradients update only the online network.

The target network is a delayed copy of the online network. Holding target parameters
fixed for many optimizer updates reduces the moving-target feedback that can destabilize
nonlinear Q-learning. The implementation makes synchronization explicit and tests it.
Huber loss implements the paper's error-clipping idea smoothly: it is quadratic near
zero and linear for large errors. Rewards are sign-clipped by default, gamma is 0.99,
and RMSProp is used for paper alignment.

Uniform replay samples all populated slots equally. The extension uses proportional
prioritized replay: transition `i` has probability proportional to
`(|TD error_i| + epsilon)^alpha`. New transitions receive the current maximum priority.
Importance weights `(N * P(i))^-beta`, normalized within the minibatch, reduce sampling
bias; beta is annealed toward one. This is a later, documented extension inspired by
the 2015 paper's observation that uniform replay cannot emphasize informative samples,
not an algorithm fully specified by that paper.
