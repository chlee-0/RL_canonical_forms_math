# Phase 6: DQN in the Pentagon Puzzle

## Objective

- Extend Phase 4 (Q-learning) to a **DQN**.


---

## Environment (recap)

- State: $s = (s_0,\dots,s_4) \in \mathbb{Z}^5$ (labels on pentagon vertices).
- Action $i \in \{0,\dots,4\}$ update the triple $(i-1, i, i+1)$ (with indices mod 5) as:
$$s_{i-1}' = s_{i-1} + s_i,\quad s_i'     = -s_i,\quad s_{i+1}' = s_{i+1} + s_i,$$
- Terminal when all entries are $\ge 0$.
- Same start as before: $s_\text{init} = (20, 18, -1, -13, -17)$.
- Rewards: step = $-1$; optional goal bonus (e.g., $+20$) on first reach of a terminal state.
- discount rate $\gamma = 1$.

---

## What is new vs Phase 5 (Gridworld DQN)?

- **Input size** is 5 (the vector itself), **output size** is 5 actions.
- **State scale.** Use a simple normalization (e.g., divide by a constant like 20–50 or clip to a range) so the net sees moderate numbers.
- Episodes are cut off after at most **50 steps** (if we have taken 50 moves without reaching success, we just stop that episode), matching Phase 4.
- Everything else (replay, target net, $\epsilon$-greedy, TD target) is unchanged.

---

## DQN Parts and Training (Quick)

- **Q-network.** Input 5-dim vector (normalized) → 128 ReLU → 128 ReLU → linear head of size 5.
- **Target network.** Copy of the online net; only it appears in the target. $Q_\theta$ is online, $Q_{\theta^-}$ is the frozen copy.
- **Replay buffer.** Store $(s, a, r, s', \text{done})$; sample random mini-batches to break correlation.
- **TD target.** For each sample,
  $$y = r + \gamma (1 - \text{done}) \cdot \max_{a'} Q_{\theta^-}(s', a').$$
  Treat $Q_{\theta^-}$ as frozen numbers; update $Q_\theta$ to match $y$.
- **Loss + step.** Minimize MSE between $Q_\theta(s, a)$ and $y$ with Adam (plain MLP regression on bootstrapped targets).
- **Exploration.** $\epsilon$-greedy, decaying over episodes.
- **Why stable?** Replay shuffles data; the target net keeps targets from moving too fast.

---

## Training Loop (sketch)

1. Make replay buffer and two nets (online + target). Copy online weights into target.
2. For each episode (cap steps at ~50):
   - Start at $s_\text{init}$.
   - Choose action with $\epsilon$-greedy on $Q_\theta$.
   - Apply the triple update, log $(s,a,r,s',\text{done})$ to replay.
   - If replay is warm, sample a batch, build $y$ with $Q_{\theta^-}$, and take an optimizer step on $Q_\theta$.
   - Every few steps (e.g., 100 updates), copy online weights into $Q_{\theta^-}$.
   - Stop the episode if terminal.
3. After training, roll out the greedy policy to see a path and total reward.

---

## How to read the result

- Larger $Q(s,a)$ means that move is closer (in expected steps) to success, given the reward design.
- The greedy rollout path should look similar to the Phase 4 path; if it diverges, increase episodes or tweak scaling/bonus.
- Printing the approximate $Q$-table at $s_\text{init}$ is a quick sanity check on which vertex move the net prefers.

---

## Exercises (what to look for)

1. **State scaling.** Try dividing states by 10, 50, or clipping to [-50, 50]. Goal: see how input scale affects stability.
2. **Goal bonus.** Turn the terminal bonus on/off or change its size. Goal: check if a bonus helps the net find success faster.
3. **Target updates.** Make target syncs frequent vs. sparse (e.g., 20 vs. 500 steps). Goal: trade off stability vs. freshness.
4. **Compare to Phase 4.** After training, compare the greedy path to the Phase 4 Q-learning path. Goal: see if function approximation changes the route or step count.
