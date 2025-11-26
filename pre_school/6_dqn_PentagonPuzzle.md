# Phase 6: DQN in the Pentagon Puzzle

## Objective

- Take the Pentagon Puzzle from Phase 4 (tabular Q-learning) and upgrade it to a **DQN**, using the same ideas introduced for Gridworld in Phase 5.


---

## Environment (recap)

- State: $s = (s_0,\dots,s_4) \in \mathbb{Z}^5$ (labels on pentagon vertices).
- Action $i \in \{0,\dots,4\}$ update the triple $(i-1, i, i+1)$ (with indices mod 5) as:
$$s_{i-1}' = s_{i-1} + s_i,\quad s_i'     = -s_i,\quad s_{i+1}' = s_{i+1} + s_i,$$
- Terminal when all entries are $\ge 0$.
- Same start as before: $s_\text{init} = (20, 18, -1, -13, -17)$.
- Rewards: step = $-1$; optional goal bonus (e.g., $+20$) on first reach of a terminal state.
- discount factor $\gamma = 1$.


---

## What is new vs Phase 5 (Gridworld DQN)?

- **Input size** is 5 (the vector itself), **output size** is 5 actions.
- **State scale.** Use a simple normalization (e.g., divide by a constant like 20–50 or clip to a range) so the net sees moderate numbers. In the code below we simply divide by 20.0, which roughly matches the magnitude of $s_\text{init}$.
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

## Key code pieces (Python)

The full runnable script lives in  `pre_school/6_dqn_PentagonPuzzle.py`.  
Here we highlight just the core pieces and how they relate to the ideas above.

### State representation and normalization

We reuse the Pentagon dynamics from Phase 4 and only change how we feed states into a network:

```python
DIM = 5
STATE_SCALE = 20.0  # simple input scaling

def state_to_tensor(state: tuple[int, ...]) -> torch.Tensor:
    # state is a 5-tuple of ints, e.g. (20, 18, -1, -13, -17)
    arr = torch.tensor(state, dtype=torch.float32, device=DEVICE) / STATE_SCALE
    return arr.unsqueeze(0)  # shape (1, 5)
```

Dividing by `STATE_SCALE` keeps the inputs in a moderate range so the MLP trains more stably.

The Q-network itself directly maps these 5‑dimensional inputs to 5 action values:

```python
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS),  # NUM_ACTIONS = 5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

### Replay buffer and TD update

We store transitions in a replay buffer and sample random mini‑batches:

```python
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
```

The TD loss matches the DQN target described above:

```python
def compute_td_loss(q_net, target_net, optimizer, batch) -> float:
    states, actions, rewards, next_states, dones = batch

    state_tensor = states_to_tensor(states)       # shape (B, 5)
    next_state_tensor = states_to_tensor(next_states)

    action_tensor = torch.tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    done_tensor = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

    q_values = q_net(state_tensor).gather(1, action_tensor).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_state_tensor).max(1)[0]
        target = reward_tensor + GAMMA * (1.0 - done_tensor) * next_q

    loss = nn.functional.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())
```

The only differences from Phase 5 (Gridworld DQN) are the input dimension (5 instead of 2) and the environment‑specific reward/termination.

### Training loop (sketch)

The training loop follows the same pattern as Phase 5:

1. Make replay buffer and two nets (online + target). Copy online weights into target.
2. For each episode (cap steps at ~50):
   - Start at $s_\text{init}$.
   - Choose action with $\epsilon$-greedy on $Q_\theta$.
   - Apply the triple update, log $(s,a,r,s',\text{done})$ to replay.
   - If the replay buffer has enough samples (at least one batch), sample a batch, build the targets $y$ with $Q_{\theta^-}$, and take an optimizer step on $Q_\theta$.
   - Every few steps (e.g., 100 updates), copy online weights into $Q_{\theta^-}$.
   - Stop the episode if terminal.
3. After training, roll out the greedy policy to see a path and total reward.

In the actual script we also:

- run a gradient update only every few environment steps (a small `UPDATE_EVERY`),
- log the cumulative fraction of successful episodes every 100 episodes,
- and print a final greedy rollout from $s_\text{init}$ with the corresponding $Q(s,\cdot)$ values at each visited state.

---

## How to read the result

- Larger $Q(s,a)$ generally means that move is better under the "$-1$ per step + optional goal bonus" reward design—typically corresponding to fewer expected steps to success, plus any contribution from the terminal bonus.
- The greedy rollout path should look similar to the Phase 4 path; if it diverges, increase episodes or tweak scaling/bonus.
- Printing the approximate $Q$-table at $s_\text{init}$ is a quick sanity check on which vertex move the net prefers.

---

## Exercise: checking generalization

After training the DQN on the Pentagon Puzzle from the fixed initial state $s_\text{init}$, test whether the network has learned something that generalizes beyond that single start:

1. Sample many new integer states $s = (s_0,\dots,s_4)$ with $\sum_i s_i > 0$. 
2. For each such $s$, run a greedy rollout using the learned $Q_\theta$ (always taking $\arg\max_a Q_\theta(s, a)$) with the same step cap (e.g. 50 steps) and record:
   - whether the rollout reaches a success state (all coordinates $\ge 0$),
   - how many steps it took if successful (or that it failed within the cap).
3. Estimate the success rate. Does the network seem to have learned a genuinely useful strategy on a region of the state space, or only a narrow route from the original start?
