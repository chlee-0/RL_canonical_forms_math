import math
import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


"""
DQN experiment for the action of SL(2, Z) on P^1(Q).

States are projective rational points [p:q] with integers p, q, represented
in a normalized form. The generators S, T, T^{-1} act via
2x2 matrices:

    S = [[0, -1],
         [1,  0]]

    T = [[1, 1],
         [0, 1]]

    T^{-1} = [[1, -1],
              [0,  1]]

The goal is to map a fixed starting rational point to [0:1] using these
generators; we treat this as a shortest-path problem and train a small
DQN agent.
"""


Action = int
State = Tuple[int, int]  # (p, q) representing [p:q]


# Environment setup
INIT_STATE: State = (3, 4)  # starting rational: 7/5
TARGET_STATE: State = (0, 1)  # [0:1]

ACTION_S: Action = 0
ACTION_T: Action = 1
ACTION_T_INV: Action = 2
NUM_ACTIONS: int = 3

MAX_STEPS_PER_EPISODE = 32
MAX_ABS_COORD = 100
STEP_REWARD = -1.0
SUCCESS_REWARD = 10.0
OVERFLOW_PENALTY = -10.0
# Reward shaping: bonus added when denominator decreases (q_next < q)
DENOM_SHAPING_BONUS = 0.0


# DQN hyperparameters (rough starting point; feel free to tweak)
GAMMA = 0.99
ALPHA = 1e-3  # learning rate
EPSILON_START = 0.2
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_CAPACITY = 10_000
NUM_EPISODES = 50_000
TARGET_UPDATE = 100  # gradient steps between target syncs
LOG_INTERVAL = 1000
UPDATE_EVERY = 1  # how often to run a gradient step

# Simple input scaling so the network sees moderate numbers
STATE_SCALE = 20.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_projective(p: int, q: int) -> State:
    """Return a canonical representative of [p:q] in P^1(Q).

    We divide by gcd and enforce q >= 0 except for the point at infinity.
    Infinity is represented by (1, 0) or (-1, 0), which we normalize to (1, 0).
    """
    if q == 0:
        # Represent infinity as (1, 0)
        return (1, 0)

    if p == 0:
        # [0:q] = [0:1]
        return (0, 1)

    g = math.gcd(p, q)
    p //= g
    q //= g

    if q < 0:
        p = -p
        q = -q

    return (p, q)


def apply_generator(state: State, action: Action) -> State:
    """Apply S, T, or T^{-1} to a projective point [p:q]."""
    p, q = state

    if action == ACTION_S:
        a, b, c, d = 0, -1, 1, 0
    elif action == ACTION_T:
        a, b, c, d = 1, 1, 0, 1
    elif action == ACTION_T_INV:
        a, b, c, d = 1, -1, 0, 1
    else:
        raise ValueError(f"Unknown action {action}")

    p2 = a * p + b * q
    q2 = c * p + d * q
    return normalize_projective(p2, q2)


def is_target(state: State) -> bool:
    return state == TARGET_STATE


def is_overflow(state: State) -> bool:
    p, q = state
    return abs(p) > MAX_ABS_COORD or abs(q) > MAX_ABS_COORD


def state_to_tensor(state: State) -> torch.Tensor:
    """Convert an integer (p, q) state to a normalized 1×2 tensor."""
    arr = torch.tensor(state, dtype=torch.float32, device=DEVICE) / STATE_SCALE
    return arr.unsqueeze(0)


def states_to_tensor(states) -> torch.Tensor:
    """Batch version: list/tuple of (p, q) -> (B,2) tensor on DEVICE."""
    arr = torch.tensor(states, dtype=torch.float32, device=DEVICE) / STATE_SCALE
    return arr


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Tuple[State, int, float, State, bool]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
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
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_action(
    q_net: QNetwork,
    state: State,
    epsilon: float,
) -> int:
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

    action_tensor = torch.tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(
        1
    )
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


def train() -> Tuple[QNetwork, int]:
    print(f"Using device: {DEVICE}")
    q_net = QNetwork().to(DEVICE)
    target_net = QNetwork().to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=ALPHA)
    buffer = ReplayBuffer(BUFFER_CAPACITY)

    epsilon = EPSILON_START
    step_count = 0
    returns: List[float] = []
    success_episodes = 0
    success_window: Deque[int] = deque(maxlen=LOG_INTERVAL)
    return_window: Deque[float] = deque(maxlen=LOG_INTERVAL)

    for episode in range(1, NUM_EPISODES + 1):
        state = INIT_STATE
        ep_return = 0.0
        reached_target = False

        for _ in range(MAX_STEPS_PER_EPISODE):
            if is_target(state):
                reached_target = True
                break

            action = select_action(q_net, state, epsilon)
            next_state = apply_generator(state, action)

            # Reward shaping: give +1 if the denominator strictly decreases
            # (ignoring cases where the denominator is 0, i.e., infinity).
            _, q = state
            _, q_next = next_state
            shaping_bonus = 0.0
            if q > 0 and q_next > 0 and q_next < q:
                shaping_bonus = DENOM_SHAPING_BONUS

            if is_target(next_state):
                done = True
                reward = SUCCESS_REWARD
            elif is_overflow(next_state):
                done = True
                reward = OVERFLOW_PENALTY
            else:
                done = False
                reward = STEP_REWARD

            reward += shaping_bonus

            buffer.push(state, action, reward, next_state, done)

            state = next_state
            step_count += 1
            ep_return += reward

            if len(buffer) >= BATCH_SIZE and step_count % UPDATE_EVERY == 0:
                batch = buffer.sample(BATCH_SIZE)
                compute_td_loss(q_net, target_net, optimizer, batch)

            if step_count % TARGET_UPDATE == 0 and step_count > 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                reached_target = is_target(next_state)
                break

        if reached_target:
            success_episodes += 1
            success_window.append(1)
        else:
            success_window.append(0)

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        returns.append(ep_return)
        return_window.append(ep_return)

        if episode % LOG_INTERVAL == 0:
            frac_success = success_episodes / float(episode)
            window_success = sum(success_window) / float(len(success_window))
            avg_return = sum(return_window) / float(len(return_window)) if return_window else 0.0
            print(
                f"Episode {episode}/{NUM_EPISODES} completed | "
                f"epsilon = {epsilon:.4f} | "
                f"cumulative terminal fraction = {frac_success:.4f} | "
                f"success_rate_last_{LOG_INTERVAL} = {window_success:.4f} | "
                f"avg_return_last_{LOG_INTERVAL} = {avg_return:.2f}",
                flush=True,
            )

    return q_net, success_episodes


def action_to_symbol(action: Action) -> str:
    if action == ACTION_S:
        return "S"
    if action == ACTION_T:
        return "T"
    if action == ACTION_T_INV:
        return "T^{-1}"
    return "?"


def greedy_rollout(
    q_net: QNetwork,
    max_steps: int = MAX_STEPS_PER_EPISODE,
) -> List[Tuple[State, Action | None]]:
    """Follow the greedy policy from INIT_STATE using the learned Q-network."""
    state = INIT_STATE
    path: List[Tuple[State, Action | None]] = []

    for _ in range(max_steps):
        if is_target(state) or is_overflow(state):
            path.append((state, None))
            break

        action = select_action(q_net, state, epsilon=0.0)
        next_state = apply_generator(state, action)

        path.append((state, action))
        state = next_state

        if is_target(state) or is_overflow(state):
            path.append((state, None))
            break

    return path


def main() -> None:
    print("Training DQN on the modular projective environment...")
    trained_q, success_episodes = train()

    print(
        f"\nReached target in {success_episodes} / {NUM_EPISODES} episodes "
        f"({success_episodes / max(1, NUM_EPISODES):.4f} fraction)."
    )

    rollout = greedy_rollout(trained_q)
    print("\nGreedy rollout from INIT_STATE:")
    print(f"(INIT_STATE = {INIT_STATE})")

    for idx, (s, a) in enumerate(rollout):
        p, q = s
        if q == 0:
            z_repr = "∞"
        else:
            z_repr = f"{p}/{q}"

        with torch.no_grad():
            q_vals = trained_q(state_to_tensor(s)).squeeze(0).cpu().numpy()
        q_row = np.round(q_vals, 3)

        if a is None:
            print(
                f"{idx:02d}: [p:q] = [{p}:{q}], z = {z_repr}, "
                f"Q(s,·) = {q_row} (terminal/overflow/stop)"
            )
        else:
            sym = action_to_symbol(a)
            if idx + 1 < len(rollout):
                s_next = rollout[idx + 1][0]
                p2, q2 = s_next
                if q2 == 0:
                    z2_repr = "∞"
                else:
                    z2_repr = f"{p2}/{q2}"
                print(
                    f"{idx:02d}: [p:q] = [{p}:{q}], z = {z_repr}, "
                    f"Q(s,·) = {q_row} --{sym}--> [{p2}:{q2}] (z = {z2_repr})"
                )
            else:
                print(
                    f"{idx:02d}: [p:q] = [{p}:{q}], z = {z_repr}, "
                    f"Q(s,·) = {q_row}, action = {sym}"
                )


if __name__ == "__main__":
    main()
