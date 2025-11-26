"""
Phase 6 (Pentagon Puzzle DQN) rewritten using the Gymnasium API.

This makes the environment look like a standard RL env:
- env.reset() -> obs, info
- env.step(action) -> obs, reward, terminated, truncated, info

The DQN pieces (network, replay buffer, TD target, epsilon-greedy)
are the same as in 6_dqn_PentagonPuzzle.py; only the environment
interaction is switched to Gymnasium style.

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


# Environment / puzzle parameters (same as Phase 4 / 6)
DIM = 5
STEP_REWARD = -1.0
GOAL_REWARD = 20.0
GAMMA = 1.0

MAX_STEPS_PER_EPISODE = 50

s_init = np.array([20, 18, -1, -13, -17], dtype=int)


def is_success(state: np.ndarray) -> bool:
    """Success if all coordinates are >= 0."""
    return np.all(state >= 0)


class PentagonPuzzleEnv(gym.Env):
    """Gymnasium-style environment for the Pentagon Puzzle."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = MAX_STEPS_PER_EPISODE):
        super().__init__()
        self.dim = DIM
        self.max_steps = max_steps

        # Observations are 5D integer vectors, but we expose them as float32.
        # In principle the entries can grow, so we use a wide Box.
        high = np.full((self.dim,), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32,
        )

        # Actions are vertex indices 0..4
        self.action_space = spaces.Discrete(self.dim)

        self.state: np.ndarray | None = None
        self.steps = 0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.state = s_init.copy()
        self.steps = 0
        obs = self.state.astype(np.float32)
        info = {}
        return obs, info

    def step(self, action: int):
        assert self.state is not None, "Call reset() before step()."
        self.steps += 1

        action = int(action) % self.dim
        s = self.state

        # Triple update at vertices (i-1, i, i+1) modulo 5
        left = (action - 1) % self.dim
        right = (action + 1) % self.dim

        s = s.copy().astype(int)
        si = s[action]
        s[left] += si
        s[action] = -si
        s[right] += si

        self.state = s

        terminated = is_success(self.state)
        truncated = self.steps >= self.max_steps and not terminated
        reward = GOAL_REWARD if terminated else STEP_REWARD

        obs = self.state.astype(np.float32)
        info = {}
        return obs, reward, terminated, truncated, info


# DQN hyperparameters (kept simple for comparison)
NUM_ACTIONS = DIM
STATE_SCALE = 20.0  # simple normalization, as in 6_dqn_PentagonPuzzle.py

ALPHA = 1e-3  # learning rate
EPSILON_START = 0.2
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

BATCH_SIZE = 64
BUFFER_CAPACITY = 100_000
NUM_EPISODES = 1_000
UPDATE_EVERY = 4  # run a gradient step every N environment steps
TARGET_UPDATE = 200  # gradient steps between target syncs
LOG_INTERVAL = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    """Convert a raw 5D observation to a normalized 1×5 tensor on DEVICE."""
    arr = torch.tensor(obs, dtype=torch.float32, device=DEVICE) / STATE_SCALE
    return arr.unsqueeze(0)


def batch_obs_to_tensor(observations) -> torch.Tensor:
    """Batch version: list/array of observations -> (B, 5) tensor."""
    # observations is typically a tuple/list of np.ndarray; stack first for speed.
    obs_np = np.asarray(observations, dtype=np.float32)
    return torch.from_numpy(obs_np).to(DEVICE) / STATE_SCALE


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
            nn.Linear(DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS),
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

    env = PentagonPuzzleEnv(max_steps=MAX_STEPS_PER_EPISODE)

    q_net = QNetwork().to(DEVICE)
    target_net = QNetwork().to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=ALPHA)
    buffer = ReplayBuffer(BUFFER_CAPACITY)

    epsilon = EPSILON_START
    step_count = 0
    success_episodes = 0

    for episode in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        reached_terminal = False

        while not (terminated or truncated):
            action = select_action(q_net, obs, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            done_flag = bool(terminated or truncated)
            buffer.push(obs, action, reward, next_obs, done_flag)

            obs = next_obs
            step_count += 1

            if len(buffer) >= BATCH_SIZE and step_count % UPDATE_EVERY == 0:
                batch = buffer.sample(BATCH_SIZE)
                compute_td_loss(q_net, target_net, optimizer, batch)

            if step_count % TARGET_UPDATE == 0 and step_count > 0:
                target_net.load_state_dict(q_net.state_dict())

        if terminated:
            reached_terminal = True

        if reached_terminal:
            success_episodes += 1

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % LOG_INTERVAL == 0:
            frac_success = success_episodes / float(episode)
            print(
                f"Episode {episode}/{NUM_EPISODES} completed | "
                f"cumulative terminal fraction = {frac_success:.4f}",
                flush=True,
            )

    return q_net, env, success_episodes


def greedy_rollout(
    q_net: QNetwork,
    env: gym.Env,
    max_steps: int = MAX_STEPS_PER_EPISODE,
):
    """Roll out the greedy policy from the default initial state."""
    obs, _ = env.reset()
    path: list[tuple[np.ndarray, int | None]] = []

    for _ in range(max_steps):
        # Compute Q(s,·) for logging
        with torch.no_grad():
            q_vals = q_net(obs_to_tensor(obs)).squeeze(0).cpu().numpy()
        q_row = np.round(q_vals, 3)

        # If already terminal, stop.
        if is_success(obs.astype(int)):
            path.append((obs.copy(), None))
            break

        action = select_action(q_net, obs, epsilon=0.0)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        path.append((obs.copy(), action))
        obs = next_obs

        if terminated or truncated:
            # Log final state as terminal/unknown
            with torch.no_grad():
                final_q_vals = q_net(obs_to_tensor(obs)).squeeze(0).cpu().numpy()
            final_q_row = np.round(final_q_vals, 3)
            path.append((obs.copy(), None))
            break

    return path


if __name__ == "__main__":
    trained_q, env, success_episodes = train()

    print(
        f"\nReached terminal in {success_episodes} / {NUM_EPISODES} episodes "
        f"({success_episodes / max(1, NUM_EPISODES):.4f} fraction)."
    )

    rollout = greedy_rollout(trained_q, env)
    print("\nGreedy rollout from s_init (Gymnasium env):")
    for idx, (obs, a) in enumerate(rollout):
        with torch.no_grad():
            q_vals = trained_q(obs_to_tensor(obs)).squeeze(0).cpu().numpy()
        q_row = np.round(q_vals, 3)
        state_tuple = tuple(int(x) for x in obs)
        if a is None:
            print(f"{idx:02d}: s = {state_tuple}, Q(s,·) = {q_row} (terminal/unknown)")
        else:
            print(f"{idx:02d}: s = {state_tuple}, Q(s,·) = {q_row}, a = {a}")
