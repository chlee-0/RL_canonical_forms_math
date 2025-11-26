import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Pentagon puzzle setup (same dynamics as Phase 4)
DIM = 5
NUM_ACTIONS = 5  # vertices 0..4

GAMMA = 1.0
STEP_REWARD = -1.0
GOAL_REWARD = 20.0

# DQN hyperparameters (rough starting point; feel free to tweak)
ALPHA = 1e-3  # learning rate
EPSILON_START = 0.2
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_CAPACITY = 100_000
NUM_EPISODES = 20_000
MAX_STEPS_PER_EPISODE = 50
TARGET_UPDATE = 200  # gradient steps between target syncs
LOG_INTERVAL = 1_000
UPDATE_EVERY = 4  # how often to run a gradient step

# Simple input scaling so the network sees moderate numbers
STATE_SCALE = 20.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

s_init = np.array([20, 18, -1, -13, -17], dtype=int)
INIT_STATE = tuple(int(x) for x in s_init)


def is_success(s: np.ndarray) -> bool:
    return np.all(s >= 0)


def apply_action(s: np.ndarray, i: int) -> np.ndarray:
    """Apply the triple update at vertex i (indices mod 5)."""
    i = i % DIM
    left = (i - 1) % DIM
    right = (i + 1) % DIM

    s = s.copy().astype(int)
    si = s[i]
    s[left] += si
    s[i] = -si
    s[right] += si
    return s


def state_to_tensor(state: tuple[int, ...]) -> torch.Tensor:
    """Convert an integer 5-tuple state to a normalized 1×5 tensor."""
    arr = torch.tensor(state, dtype=torch.float32, device=DEVICE) / STATE_SCALE
    return arr.unsqueeze(0)


def states_to_tensor(states) -> torch.Tensor:
    """Batch version: list/tuple of 5-tuples -> (B,5) tensor on DEVICE."""
    arr = torch.tensor(states, dtype=torch.float32, device=DEVICE) / STATE_SCALE
    return arr


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> None:
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buf)


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_action(q_net: QNetwork, state: tuple[int, ...], epsilon: float) -> int:
    """Epsilon-greedy action selection on Q-network."""
    if random.random() < epsilon:
        return random.randrange(NUM_ACTIONS)
    with torch.no_grad():
        q_vals = q_net(state_to_tensor(state))
    return int(torch.argmax(q_vals).item())


def compute_td_loss(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    batch,
) -> float:
    states, actions, rewards, next_states, dones = batch

    state_tensor = states_to_tensor(states)
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


def train():
    print(f"Using device: {DEVICE}")
    q_net = QNetwork().to(DEVICE)
    target_net = QNetwork().to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=ALPHA)
    buffer = ReplayBuffer(BUFFER_CAPACITY)

    epsilon = EPSILON_START
    step_count = 0
    returns: list[float] = []
    lengths: list[int] = []
    success_episodes = 0

    for episode in range(1, NUM_EPISODES + 1):
        state = INIT_STATE
        ep_return = 0.0
        ep_len = 0
        reached_terminal = False

        for _ in range(MAX_STEPS_PER_EPISODE):
            s_arr = np.array(state, dtype=int)
            if is_success(s_arr):
                reached_terminal = True
                break

            action = select_action(q_net, state, epsilon)
            next_arr = apply_action(s_arr, action)
            next_state = tuple(int(x) for x in next_arr)

            done = is_success(next_arr)
            reward = GOAL_REWARD if done else STEP_REWARD

            buffer.push(state, action, reward, next_state, done)

            state = next_state
            step_count += 1
            ep_return += reward
            ep_len += 1

            if len(buffer) >= BATCH_SIZE and step_count % UPDATE_EVERY == 0:
                batch = buffer.sample(BATCH_SIZE)
                compute_td_loss(q_net, target_net, optimizer, batch)

            if step_count % TARGET_UPDATE == 0 and step_count > 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                reached_terminal = True
                break

        if reached_terminal:
            success_episodes += 1

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        returns.append(ep_return)
        lengths.append(ep_len)

        if episode in (1, 10, 100) or episode % LOG_INTERVAL == 0:
            frac_success = success_episodes / float(episode)
            print(
                f"Episode {episode}/{NUM_EPISODES} completed | "
                f"cumulative terminal fraction = {frac_success:.4f}"
            , flush=True)

    return q_net, success_episodes


def greedy_rollout(
    q_net: QNetwork,
    max_steps: int = MAX_STEPS_PER_EPISODE,
) -> list[tuple[tuple[int, ...], int | None]]:
    """Follow the greedy policy from s_init using the learned Q-network.

    Returns a list of (state, action) pairs; the action is None at the
    terminal state where rollout stops. This mirrors Phase 4's style.
    """
    state = INIT_STATE
    path: list[tuple[tuple[int, ...], int | None]] = []

    for _ in range(max_steps):
        s_arr = np.array(state, dtype=int)
        if is_success(s_arr):
            path.append((state, None))
            break

        action = select_action(q_net, state, epsilon=0.0)
        next_arr = apply_action(s_arr, action)
        next_state = tuple(int(x) for x in next_arr)

        done = is_success(next_arr)
        path.append((state, action))
        state = next_state

        if done:
            path.append((state, None))
            break

    return path


if __name__ == "__main__":
    trained_q, success_episodes = train()

    print(
        f"\nReached terminal in {success_episodes} / {NUM_EPISODES} episodes "
        f"({success_episodes / max(1, NUM_EPISODES):.4f} fraction)."
    )

    # Value at a reference state, similar to Phase 4.
    ref_state = (2, 1, 0, 3, 1)
    print("\nValue at reference state (2, 1, 0, 3, 1):")
    with torch.no_grad():
        q_vals_ref = trained_q(state_to_tensor(ref_state)).squeeze(0).cpu().numpy()
    v_ref = float(np.max(q_vals_ref))
    q_row_ref = np.round(q_vals_ref, 3)
    print(f"s = {ref_state}, V(s) ≈ {v_ref:.3f}, Q(s,·) = {q_row_ref}")

    rollout = greedy_rollout(trained_q)
    print("\nGreedy rollout from s_init:")
    print(f"(s_init = {INIT_STATE})")
    for idx, (s, a) in enumerate(rollout):
        if a is None:
            print(f"{idx:02d}: s = {s} (terminal/unknown)")
        else:
            if idx + 1 < len(rollout):
                s_next = rollout[idx + 1][0]
                print(f"{idx:02d}: s = {s} --a={a}--> {s_next}")
            else:
                print(f"{idx:02d}: s = {s}, a = {a}")
