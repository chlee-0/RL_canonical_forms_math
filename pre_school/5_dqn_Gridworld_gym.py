"""
Phase 5: Deep Q-Network (DQN) in Gridworld, rewritten using the Gymnasium API.

The original script `5_dqn_Gridworld.py` implements a small DQN loop with
its own step() function. Here we wrap the 4×4 Gridworld as a Gymnasium
environment and keep the DQN parts essentially the same.

To run this script you need gymnasium installed, e.g.:
    pip install gymnasium
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
from gymnasium import spaces


# Gridworld setup (same as in 5_dqn_Gridworld.py)
GRID_SIZE = 4
NUM_ACTIONS = 4  # up, down, left, right
START_STATE = (0, 0)
GOAL_STATE = (GRID_SIZE - 1, GRID_SIZE - 1)

STEP_REWARD = -1.0
GAMMA = 1.0
MAX_STEPS_PER_EPISODE = 50


class GridworldEnv(gym.Env):
    """Simple 4x4 Gridworld using the Gymnasium API."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = MAX_STEPS_PER_EPISODE):
        super().__init__()
        self.grid_size = GRID_SIZE
        self.max_steps = max_steps

        # Observations: (r, c) coordinates as float32
        low = np.array([0, 0], dtype=np.float32)
        high = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.state: tuple[int, int] | None = None
        self.steps = 0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.state = START_STATE
        self.steps = 0
        r, c = self.state
        obs = np.array([r, c], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action: int):
        assert self.state is not None, "Call reset() before step()."
        self.steps += 1

        r, c = self.state
        if action == 0:  # up
            dr, dc = -1, 0
        elif action == 1:  # down
            dr, dc = 1, 0
        elif action == 2:  # left
            dr, dc = 0, -1
        else:  # action == 3, right
            dr, dc = 0, 1

        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            nr, nc = r, c  # bounce off walls

        self.state = (nr, nc)

        terminated = self.state == GOAL_STATE
        truncated = self.steps >= self.max_steps and not terminated
        reward = STEP_REWARD

        obs = np.array([nr, nc], dtype=np.float32)
        info = {}
        return obs, reward, terminated, truncated, info


# DQN hyperparameters (mirroring 5_dqn_Gridworld.py)
ALPHA = 1e-3  # learning rate
EPSILON_START = 0.2
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
BUFFER_CAPACITY = 50_000
NUM_EPISODES = 1_500
TARGET_UPDATE = 100  # steps between target syncs
LOG_INTERVAL = 200
UPDATE_EVERY = 1  # how often to run a gradient step

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    """Convert (r,c) observation to normalized 1×2 tensor on DEVICE."""
    r, c = obs
    x = np.array(
        [r / (GRID_SIZE - 1), c / (GRID_SIZE - 1)],
        dtype=np.float32,
    )
    return torch.tensor(x[None, :], dtype=torch.float32, device=DEVICE)


def batch_obs_to_tensor(observations) -> torch.Tensor:
    """Batch version: list/array of (r,c) -> (B,2) tensor."""
    obs_np = np.asarray(observations, dtype=np.float32)
    obs_np[:, 0] /= (GRID_SIZE - 1)
    obs_np[:, 1] /= (GRID_SIZE - 1)
    return torch.from_numpy(obs_np).to(DEVICE)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_action(q_net: QNetwork, obs: np.ndarray, epsilon: float) -> int:
    """Epsilon-greedy action selection given the current observation."""
    if random.random() < epsilon:
        return random.randrange(NUM_ACTIONS)
    with torch.no_grad():
        q_vals = q_net(obs_to_tensor(obs))
    return int(torch.argmax(q_vals).item())


def compute_td_loss(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    batch,
) -> float:
    obs, actions, rewards, next_obs, dones = batch

    obs_tensor = batch_obs_to_tensor(obs)
    next_obs_tensor = batch_obs_to_tensor(next_obs)

    action_tensor = torch.tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    done_tensor = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

    q_values = q_net(obs_tensor).gather(1, action_tensor).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_obs_tensor).max(1)[0]
        target = reward_tensor + GAMMA * (1.0 - done_tensor) * next_q

    loss = nn.functional.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def train():
    print(f"Using device: {DEVICE}")

    env = GridworldEnv(max_steps=MAX_STEPS_PER_EPISODE)

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
        obs, _ = env.reset()
        terminated = False
        truncated = False
        ep_return = 0.0
        ep_len = 0

        while not (terminated or truncated):
            action = select_action(q_net, obs, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            done_flag = bool(terminated or truncated)
            buffer.push(obs, action, reward, next_obs, done_flag)

            obs = next_obs
            step_count += 1
            ep_return += reward
            ep_len += 1

            if len(buffer) >= BATCH_SIZE and step_count % UPDATE_EVERY == 0:
                batch = buffer.sample(BATCH_SIZE)
                compute_td_loss(q_net, target_net, optimizer, batch)

            if step_count % TARGET_UPDATE == 0 and step_count > 0:
                target_net.load_state_dict(q_net.state_dict())

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
                f"avg_len={avg_len:.1f}",
                flush=True,
            )

    return q_net, env


def greedy_rollout(
    q_net: QNetwork,
    env: gym.Env,
    max_steps: int = MAX_STEPS_PER_EPISODE,
):
    """Roll out the greedy policy from START_STATE and print Q-values."""
    obs, _ = env.reset()
    path: list[tuple[tuple[int, int], int | None, np.ndarray]] = []

    for _ in range(max_steps):
        r, c = int(obs[0]), int(obs[1])
        state_tuple = (r, c)
        with torch.no_grad():
            q_vals = q_net(obs_to_tensor(obs)).squeeze(0).cpu().numpy()
        q_row = np.round(q_vals, 3)

        # If already at goal, stop.
        if state_tuple == GOAL_STATE:
            path.append((state_tuple, None, q_row))
            break

        action = select_action(q_net, obs, epsilon=0.0)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        path.append((state_tuple, action, q_row))
        obs = next_obs

        if terminated or truncated:
            r_f, c_f = int(obs[0]), int(obs[1])
            final_state = (r_f, c_f)
            with torch.no_grad():
                final_q_vals = q_net(obs_to_tensor(obs)).squeeze(0).cpu().numpy()
            final_q_row = np.round(final_q_vals, 3)
            path.append((final_state, None, final_q_row))
            break

    print("\nGreedy rollout from START_STATE (Gymnasium env):")
    for idx, (s, a, q_row) in enumerate(path):
        if a is None:
            print(f"{idx:02d}: s = {s}, Q(s,·) = {q_row} (terminal/unknown)")
        else:
            print(f"{idx:02d}: s = {s}, Q(s,·) = {q_row}, a = {a}")


if __name__ == "__main__":
    trained_q, env = train()
    greedy_rollout(trained_q, env)

