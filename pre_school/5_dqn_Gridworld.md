# Phase 5: Deep Q-Network (DQN) in Gridworld

## Objective

- Pick up from Phase 3 and **approximate $Q(s, a)$ with a tiny neural net**.
- Learn two stabilizers: **experience replay** (shuffle training data) and a **target network** (slow-moving target).
- Use the same 4×4 Gridworld and compare with the tabular result.
- Keep the wording light; focus on intuition and the connection to earlier phases.

---

## Environment: 4×4 Gridworld (Same as Phases 1 & 3)

- Start $S = (0,0)$; goal $G = (3,3)$.
- Actions: up / down / left / right; move outside the grid → stay put.
- Reward: $-1$ per step; $\gamma = 1.0$.

```text
S . . .
. . . .
. . . .
. . . G
```

---

## Why Move Beyond Tabular Q?

Tabular Q is fine on a tiny grid, but tables do not scale when the state space grows. A neural net lets nearby states share value estimates. DQN is the standard bridge from tabular Q to function approximation.

---

## What is DQN?

Deep Q-Network (DQN) is Q-learning where the Q-table is replaced by a neural network. It keeps the same TD target as tabular Q-learning, but adds two stabilizers:

- a **replay buffer** to shuffle experience,
- a **target network** to make the bootstrap target change slowly.

That’s essentially it—the same objective, but with a function approximator and two stabilizing mechanisms.

---

## DQN Parts and Training (Quick)

- **Q-network.** Input $(r, c)$ normalized to $[0,1]$ → 64 ReLU → 64 ReLU → linear head of size 4 giving $Q(s, \cdot)$.
- **Target network.** A slow-copy of the Q-net; only it appears in the target. Notation: $Q_\theta$ is the online (trainable) net, $Q_{\theta^-}$ is the copied target net.
- **Replay buffer.** Store recent transitions $(s, a, r, s', \text{done})$ in a queue; train on random mini-batches to break the “recent steps are all similar” correlation. This makes gradient updates closer to i.i.d. samples and stabilizes training.
- **TD target.** For each sample,
  $$y = r + \gamma (1 - \text{done}) \cdot \max_{a'} Q_{\theta^-}(s', a').$$
  Think of $Q_{\theta^-}$ as frozen numbers; only the online net $Q_\theta$ gets updated to match $y$.
- **Loss + step.** Minimize MSE between $Q_\theta(s, a)$ and $y$ with Adam (just MLP regression with bootstrapped targets).
- **Exploration.** $\epsilon$-greedy, decayed over episodes.
- **Why stable?** Replay shuffles data; the target net slows the moving target.

---

## Algorithm Sketch (Gridworld)

1. Initialize replay buffer and two networks: online $Q_\theta$ and target $Q_{\theta^-}$ (copied from $Q_\theta$).
2. For each episode:
   - Start at $S$ and repeat up to a fixed step limit:
     - Pick $a$ with $\epsilon$-greedy on $Q_\theta$.
     - Move in the grid, get $(r, s', \text{done})$; push it into the buffer.
     - If the buffer has enough samples, grab a random mini-batch, build targets with $Q_{\theta^-}$, and train $Q_\theta$.
     - Every `TARGET_UPDATE` steps, copy $\theta$ into $\theta^-$.
     - If `done`, stop the episode.
3. After training, derive a greedy policy $\pi(s) = \arg\max_a Q_\theta(s, a)$.

On this tiny grid, DQN should recover the same shortest-path behavior as tabular Q, but the wiring is now ready for larger problems.

---

## Code Example (python + PyTorch)

The script below lives at `pre_school/5_dqn_Gridworld.py`; run it with `python pre_school/5_dqn_Gridworld.py`. Adjust hyperparameters near the top.

```python
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Gridworld setup
GRID_SIZE = 4
NUM_ACTIONS = 4  # up, down, left, right
START_STATE = (0, 0)
GOAL_STATE = (GRID_SIZE - 1, GRID_SIZE - 1)

# Hyperparameters
GAMMA = 1.0
ALPHA = 1e-3  # learning rate
EPSILON_START = 0.2
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
BUFFER_CAPACITY = 50_000
NUM_EPISODES = 1500
MAX_STEPS_PER_EPISODE = 50
TARGET_UPDATE = 100  # steps between target syncs
LOG_INTERVAL = 200

STEP_REWARD = -1.0
DEVICE = torch.device("cpu")


def is_terminal(state: tuple[int, int]) -> bool:
    return state == GOAL_STATE


def step(state: tuple[int, int], action: int):
    r, c = state
    if action == 0:  # up
        dr, dc = -1, 0
    elif action == 1:  # down
        dr, dc = 1, 0
    elif action == 2:  # left
        dr, dc = 0, -1
    else:  # right
        dr, dc = 0, 1

    nr, nc = r + dr, c + dc
    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
        nr, nc = r, c  # bounce off walls

    next_state = (nr, nc)
    done = is_terminal(next_state)
    reward = STEP_REWARD
    return next_state, reward, done


def state_to_tensor(state: tuple[int, int]) -> torch.Tensor:
    # Normalize coordinates to [0,1] for the network
    r, c = state
    return torch.tensor([[r / (GRID_SIZE - 1), c / (GRID_SIZE - 1)]], dtype=torch.float32, device=DEVICE)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, d = zip(*batch)
        return s, a, r, s_next, d

    def __len__(self):
        return len(self.buf)


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


def select_action(q_net: QNetwork, state, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(NUM_ACTIONS)
    with torch.no_grad():
        q_vals = q_net(state_to_tensor(state))
    return int(torch.argmax(q_vals).item())


def compute_td_loss(q_net, target_net, optimizer, batch):
    states, actions, rewards, next_states, dones = batch

    state_tensor = torch.cat([state_to_tensor(s) for s in states], dim=0)
    next_state_tensor = torch.cat([state_to_tensor(s) for s in next_states], dim=0)

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


def train():
    q_net = QNetwork().to(DEVICE)
    target_net = QNetwork().to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=ALPHA)
    buffer = ReplayBuffer(BUFFER_CAPACITY)

    epsilon = EPSILON_START
    step_count = 0
    returns = []
    lengths = []

    for episode in range(1, NUM_EPISODES + 1):
        state = START_STATE
        ep_return = 0.0
        ep_len = 0
        for _ in range(MAX_STEPS_PER_EPISODE):
            action = select_action(q_net, state, epsilon)
            next_state, reward, done = step(state, action)
            buffer.push(state, action, reward, next_state, done)

            state = next_state
            step_count += 1
            ep_return += reward
            ep_len += 1

            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                compute_td_loss(q_net, target_net, optimizer, batch)

            if step_count % TARGET_UPDATE == 0 and step_count > 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        # Decay epsilon after each episode
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        returns.append(ep_return)
        lengths.append(ep_len)

        if episode % LOG_INTERVAL == 0:
            recent_returns = returns[-LOG_INTERVAL:]
            recent_lengths = lengths[-LOG_INTERVAL:]
            avg_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
            avg_len = float(np.mean(recent_lengths)) if recent_lengths else 0.0
            print(
                f"Episode {episode}/{NUM_EPISODES}, epsilon={epsilon:.3f}, "
                f"buffer={len(buffer)}, avg_return(last{LOG_INTERVAL})={avg_ret:.2f}, "
                f"avg_len={avg_len:.1f}"
            )

    return q_net


def greedy_policy(q_net: QNetwork):
    policy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            with torch.no_grad():
                q_vals = q_net(state_to_tensor((r, c)))
            policy[r, c] = int(torch.argmax(q_vals).item())
    return policy


def policy_to_arrows(policy: np.ndarray) -> list[str]:
    """Convert numeric policy (0 up, 1 down, 2 left, 3 right) to arrow strings."""
    arrow_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    lines = []
    for r in range(GRID_SIZE):
        arrows = " ".join(arrow_map[int(policy[r, c])] for c in range(GRID_SIZE))
        lines.append(arrows)
    return lines


def print_q_table(q_net: QNetwork):
    """Print approximate Q-values for all grid cells, similar to the tabular Phase 3 log."""
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            with torch.no_grad():
                q_vals = q_net(state_to_tensor((r, c))).squeeze(0).cpu().numpy()
            q_str = np.array2string(q_vals, precision=2, separator=" ", suppress_small=False)
            print(f"s=({r}, {c}) Q=[up,down,left,right] {q_str}")


def evaluate_greedy(q_net: QNetwork, max_steps: int = MAX_STEPS_PER_EPISODE):
    state = START_STATE
    path = [state]
    total_reward = 0.0

    for _ in range(max_steps):
        action = select_action(q_net, state, epsilon=0.0)
        next_state, reward, done = step(state, action)
        total_reward += reward
        path.append(next_state)
        state = next_state
        if done:
            break

    steps_taken = len(path) - 1
    return path, total_reward, steps_taken


if __name__ == "__main__":
    trained_q = train()
    print("Learned Q-table (approx):")
    print_q_table(trained_q)
    policy = greedy_policy(trained_q)
    path, total_reward, steps_taken = evaluate_greedy(trained_q)
    print("Greedy policy (arrows):")
    for line in policy_to_arrows(policy):
        print(line)
    print(f"Greedy rollout: steps={steps_taken}, total_reward={total_reward:.1f}, path={path}")
```

---

## How to Interpret the Result

- The greedy policy should reproduce the shortest-path pattern from Phase 3. If it wobbles, increase episodes or buffer size.
- Because we use function approximation, nearby states can share information: even if some cells are rarely visited, the network can infer reasonable $Q$-values from similar states.
- On larger environments, the same wiring (replay + target net + neural $Q_\theta$) scales where tabular Q fails.

---

## Exercises

1. **Match tabular Q.**  
   Run DQN and see if the greedy policy matches Phase 3. Goal: confirm the deep version can recover the tabular answer and spot if exploration or training time is the limiter.
2. **Remove replay buffer.**  
   Train on just the latest transition (no replay). Goal: observe how correlated data makes learning unstable or slower.
3. **Target updates.**  
   Change `TARGET_UPDATE` to something small or large (e.g., 10 or 500). Goal: see how different target-update frequencies affect stability and the final policy.
