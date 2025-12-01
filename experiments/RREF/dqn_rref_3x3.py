import argparse
import os
import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# For this small network and environment, CPU is typically faster
# than incurring GPU transfer overhead.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardConfig:
    """Reward configuration (mirrors q_learning.py)."""

    STEP = -1.0
    INVALID = -1.0
    NOOP = -1.0
    GOAL = 100.0
    PIVOT_GAIN = 5.0
    PIVOT_LOSS = -5.0
    ZERO_GAIN = 3.0
    ZERO_LOSS = -3.0


class RREFEnv3x3Int:
    """RREF environment for random 3x3 integer matrices."""

    def __init__(self, max_abs_entry: int = 5):
        self.max_abs_entry = int(max_abs_entry)
        self.matrix = None

        n_rows, n_cols = 3, 3

        # Build discrete action space:
        #  - Swap(i,j) for 0 <= i < j < n_rows
        #  - Normalize(i,j) for all 0 <= i < n_rows, 0 <= j < n_cols
        #  - Eliminate(i,p) for i != p (rows)
        self.actions: List[Tuple[str, int, int]] = []
        # Swap actions
        for i in range(n_rows):
            for j in range(i + 1, n_rows):
                self.actions.append(("swap", i, j))
        # Normalize actions
        for i in range(n_rows):
            for j in range(n_cols):
                self.actions.append(("normalize", i, j))
        # Eliminate actions
        for i in range(n_rows):
            for p in range(n_rows):
                if i != p:
                    self.actions.append(("eliminate", i, p))

        self.n_actions = len(self.actions)

    def _sample_matrix(self) -> np.ndarray:
        """Sample a random 3x3 integer matrix with entries in [-max_abs_entry, max_abs_entry]."""
        low, high = -self.max_abs_entry, self.max_abs_entry
        A = np.random.randint(low, high + 1, size=(3, 3)).astype(float)
        return A

    def reset(self, matrix: np.ndarray | None = None) -> np.ndarray:
        """Reset environment to a given 3x3 integer matrix or a random one."""
        if matrix is None:
            self.matrix = self._sample_matrix()
        else:
            # Always copy to avoid in-place mutation of external arrays
            mat = np.array(matrix, dtype=float, copy=True)
            if mat.shape != (3, 3):
                raise ValueError(f"Expected 3x3 matrix, got shape {mat.shape}")
            self.matrix = mat
        return self.matrix.copy()

    @staticmethod
    def _is_rref(matrix: np.ndarray) -> bool:
        """Check if the matrix is in RREF."""
        A = np.round(matrix, decimals=8)
        m, n = A.shape
        pivot_col = -1
        seen_zero_row = False

        for i in range(m):
            row = A[i]
            nonzero_indices = [j for j, val in enumerate(row) if abs(val) > 1e-8]

            if not nonzero_indices:
                # Row is all zero. Once a zero row appears, all later rows
                # must also be zero in RREF/REF.
                seen_zero_row = True
                continue
            else:
                # If we already saw a zero row, any subsequent non-zero row
                # violates the "all zero rows at the bottom" rule.
                if seen_zero_row:
                    return False

                j = nonzero_indices[0]
                # Pivot must move strictly to the right
                if j <= pivot_col:
                    return False
                pivot_col = j

                # Pivot must be 1
                if abs(A[i, j] - 1.0) > 1e-8:
                    return False

                # Pivot column must have zeros elsewhere
                for k in range(m):
                    if k == i:
                        continue
                    if abs(A[k, j]) > 1e-8:
                        return False

        return True

    @staticmethod
    def _pivot_count(matrix: np.ndarray) -> int:
        """Count how many leading rows satisfy REF pivot structure."""
        A = np.round(matrix, decimals=8)
        m, n = A.shape
        prev_col = -1
        count = 0

        for i in range(m):
            row = A[i]
            lead_col = None
            for j, val in enumerate(row):
                if abs(val) > 1e-8:
                    lead_col = j
                    break

            if lead_col is None:
                break

            if lead_col > prev_col:
                count += 1
                prev_col = lead_col
            else:
                break

        return count

    @staticmethod
    def _zero_count(matrix: np.ndarray) -> int:
        """Count entries that are (approximately) zero in the matrix."""
        A = np.round(matrix, decimals=8)
        return int(np.sum(np.abs(A) <= 1e-8))

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Apply action and return (next_state_matrix, reward, done, info)."""
        op_type, a, b = self.actions[action_idx]
        A = self.matrix
        # Clip entries to keep numerical values in a safe range
        np.clip(A, -1e6, 1e6, out=A)

        reward = RewardConfig.STEP
        done = False

        before = A.copy()
        pivot_before = self._pivot_count(before)
        zero_before = self._zero_count(before)

        if op_type == "swap":
            i, j = a, b
            A[[i, j], :] = A[[j, i], :]

        elif op_type == "normalize":
            i, j = a, b
            pivot_val = A[i, j]
            if abs(pivot_val) < 1e-8:
                reward = RewardConfig.INVALID
            else:
                A[i, :] = A[i, :] / pivot_val

        elif op_type == "eliminate":
            i, p = a, b
            row_p = A[p]
            nonzero_indices = [col for col, val in enumerate(row_p) if abs(val) > 1e-8]
            if not nonzero_indices:
                reward = RewardConfig.INVALID
            else:
                j = nonzero_indices[0]
                pivot_val = A[p, j]
                if abs(pivot_val) < 1e-8:
                    reward = RewardConfig.INVALID
                else:
                    factor = A[i, j]
                    A[i, :] = A[i, :] - factor * A[p, :]

        # No-op check
        if np.allclose(before, A, atol=1e-8):
            reward = RewardConfig.NOOP

        # Terminal check
        if self._is_rref(A):
            reward = RewardConfig.GOAL
            done = True

        # Shaping: pivot count
        pivot_after = self._pivot_count(A)
        if pivot_after > pivot_before:
            reward += RewardConfig.PIVOT_GAIN
        elif pivot_after < pivot_before:
            reward += RewardConfig.PIVOT_LOSS

        # Shaping: zero count
        zero_after = self._zero_count(A)
        delta_zero = zero_after - zero_before
        if delta_zero > 0:
            reward += RewardConfig.ZERO_GAIN * delta_zero
        elif delta_zero < 0:
            reward += RewardConfig.ZERO_LOSS * (-delta_zero)

        # Final safety clip to avoid runaway growth in entries
        np.clip(A, -1e6, 1e6, out=A)
        self.matrix = A
        return self.matrix.copy(), reward, done, {}


def state_to_tensor(matrix: np.ndarray) -> torch.Tensor:
    """Map a 3x3 matrix to a 9D input vector, with simple normalization.

    Uses safe casting + clipping to avoid overflow/NaNs, and constructs
    tensors from a single numpy array (no list-of-ndarrays warning).
    """
    # Work in float64 first to avoid overflow on cast,
    # then clip, then cast down to float32.
    v64 = np.asarray(matrix, dtype=np.float64).reshape(-1)
    # Clip extremely large values to keep things numerically stable.
    v64 = np.clip(v64, -1e6, 1e6)

    max_abs = float(np.max(np.abs(v64))) if v64.size > 0 else 1.0
    if not np.isfinite(max_abs) or max_abs == 0.0:
        v_norm = v64
    else:
        v_norm = v64 / max_abs

    v_norm = v_norm.astype(np.float32, copy=False)
    return torch.from_numpy(v_norm[None, :]).to(DEVICE)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buf.append((state.copy(), action, float(reward), next_state.copy(), done))

    def sample(
        self, batch_size: int
    ) -> Tuple[List[np.ndarray], List[int], List[float], List[np.ndarray], List[bool]]:
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return list(states), list(actions), list(rewards), list(next_states), list(dones)

    def __len__(self) -> int:
        return len(self.buf)


class QNetwork(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_action(q_net: QNetwork, state: np.ndarray, epsilon: float, n_actions: int) -> int:
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        q_vals = q_net(state_to_tensor(state))
    return int(torch.argmax(q_vals).item())


def compute_td_loss(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    batch: Tuple[
        List[np.ndarray],
        List[int],
        List[float],
        List[np.ndarray],
        List[bool],
    ],
    gamma: float,
) -> float:
    states, actions, rewards, next_states, dones = batch

    state_tensor = torch.cat([state_to_tensor(s) for s in states], dim=0)
    next_state_tensor = torch.cat([state_to_tensor(s) for s in next_states], dim=0)

    action_tensor = torch.tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    done_tensor = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

    q_values = q_net(state_tensor).gather(1, action_tensor).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_state_tensor).max(1)[0]
        target = reward_tensor + gamma * (1.0 - done_tensor) * next_q

    loss = nn.functional.mse_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def train_dqn_rref(
    episodes: int,
    max_steps: int,
    sample_bound: int,
    buffer_capacity: int = 100_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    epsilon_start: float = 0.2,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    target_update: int = 1_000,
    log_interval: int = 500,
    pretrain_episodes: int = 0,
    pretrain_pool_size: int = 0,
    q_net: "QNetwork | None" = None,
) -> Tuple[RREFEnv3x3Int, QNetwork]:
    env = RREFEnv3x3Int(max_abs_entry=sample_bound)
    if q_net is None:
        q_net = QNetwork(env.n_actions).to(DEVICE)
    target_net = QNetwork(env.n_actions).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    # Optional pretraining pool: fixed matrices reused for the first
    # `pretrain_episodes` episodes before switching to fully random starts.
    sample_pool: List[np.ndarray] = []
    if pretrain_episodes > 0 and pretrain_pool_size > 0:
        for _ in range(pretrain_pool_size):
            sample_pool.append(env._sample_matrix())
        print(
            f"[Pretrain] Using a pool of {len(sample_pool)} fixed 3x3 matrices "
            f"for the first {pretrain_episodes} episodes."
        )

    epsilon = epsilon_start
    step_count = 0

    returns: List[float] = []
    lengths: List[int] = []
    successes = 0

    total_episodes = pretrain_episodes + episodes

    for episode in range(1, total_episodes + 1):
        in_pretrain = episode <= pretrain_episodes
        phase = "Pretrain" if in_pretrain else "Train"

        if episode == 1 or episode % log_interval == 0:
            print(
                f"\n[{phase}] Starting episode {episode}/{total_episodes} "
                f"(epsilon={epsilon:.3f})"
            )

        if in_pretrain and sample_pool:
            start_matrix = random.choice(sample_pool)
            state = env.reset(start_matrix)
        else:
            state = env.reset()
        ep_return = 0.0
        ep_len = 0
        reached_goal = False

        for _ in range(max_steps):
            action = select_action(q_net, state, epsilon, env.n_actions)
            next_state, reward, done, _ = env.step(action)

            buffer.push(state, action, reward, next_state, done)

            state = next_state
            ep_return += reward
            ep_len += 1
            step_count += 1

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                compute_td_loss(q_net, target_net, optimizer, batch, gamma)

            if step_count % target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                reached_goal = True
                break

        if reached_goal:
            successes += 1

        returns.append(ep_return)
        lengths.append(ep_len)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % log_interval == 0:
            recent_returns = returns[-log_interval:]
            recent_lengths = lengths[-log_interval:]
            avg_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
            avg_len = float(np.mean(recent_lengths)) if recent_lengths else 0.0
            success_frac = successes / float(episode)
            progress = 100.0 * float(episode) / float(total_episodes)
            print(
                f"[{phase}] Episode {episode}/{total_episodes} "
                f"({progress:.1f}% done), epsilon={epsilon:.3f}, "
                f"env_steps={step_count}, buffer={len(buffer)}, "
                f"avg_return(last{log_interval})={avg_ret:.2f}, "
                f"avg_len={avg_len:.1f}, success_frac={success_frac:.3f}"
            )

    return env, q_net


def evaluate_random_matrix(
    q_net: QNetwork,
    sample_bound: int,
    max_steps: int,
) -> None:
    env = RREFEnv3x3Int(max_abs_entry=sample_bound)

    # Sample a random integer 3x3 matrix and fix it for this evaluation.
    start_matrix = env._sample_matrix()
    env.reset(start_matrix)

    print("\n=== Greedy evaluation on a random 3x3 integer matrix ===")
    print("Start matrix:")
    print(start_matrix)
    print("\nAction space (index -> (type, i, j)):")
    for idx, act in enumerate(env.actions):
        print(f"  {idx:2d}: {act}")

    state = start_matrix.copy()
    for t in range(max_steps):
        with torch.no_grad():
            q_vals = q_net(state_to_tensor(state)).squeeze(0).cpu().numpy()
        best_action = int(np.argmax(q_vals))
        op_type, a, b = env.actions[best_action]

        print(f"\nStep {t+1}:")
        print("Current matrix:")
        print(state)
        print("Q-values (per action index):")
        for idx, q in enumerate(q_vals):
            print(f"  a={idx:2d} {env.actions[idx]} -> {q:.4f}")
        print(f"Greedy action = {op_type}, args = ({a}, {b})")

        next_state, reward, done, _ = env.step(best_action)
        print("Next matrix:")
        print(next_state)
        print(f"Reward: {reward:.2f}, Done: {done}")

        state = next_state
        if done:
            print("\nReached RREF (according to environment check).")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DQN training for RREF on random 3x3 integer matrices, "
            "using the current reward shaping."
        )
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20_000,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--sample-bound",
        type=int,
        default=5,
        help="Random entries sampled from [-sample-bound, sample-bound].",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum steps per episode and evaluation.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiments/RREF/dqn_rref_3x3.pt",
        help="Path to save trained DQN weights.",
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default="",
        help="Optional path to load initial DQN weights.",
    )
    parser.add_argument(
        "--pretrain-episodes",
        type=int,
        default=0,
        help=(
            "Number of pretraining episodes that use a fixed pool "
            "of matrices before switching to fully random starts."
        ),
    )
    parser.add_argument(
        "--pretrain-pool-size",
        type=int,
        default=10,
        help="Number of fixed 3x3 matrices in the pretraining pool.",
    )
    args = parser.parse_args()

    print(
        f"Training DQN on 3x3 integer matrices with "
        f"episodes={args.episodes}, sample_bound={args.sample_bound}, "
        f"max_steps={args.max_steps}, pretrain_episodes={args.pretrain_episodes}, "
        f"pretrain_pool_size={args.pretrain_pool_size}"
    )
    print(f"Using device: {DEVICE}")

    env = RREFEnv3x3Int(max_abs_entry=args.sample_bound)
    q_net = QNetwork(env.n_actions).to(DEVICE)

    if args.load_path:
        if os.path.exists(args.load_path):
            state_dict = torch.load(args.load_path, map_location=DEVICE)
            q_net.load_state_dict(state_dict)
            print(f"Loaded initial weights from {args.load_path}")
        else:
            print(f"Warning: load-path {args.load_path} not found; training from scratch.")

    _, trained_q_net = train_dqn_rref(
        episodes=args.episodes,
        max_steps=args.max_steps,
        sample_bound=args.sample_bound,
        pretrain_episodes=args.pretrain_episodes,
        pretrain_pool_size=args.pretrain_pool_size,
        q_net=q_net,
    )

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(trained_q_net.state_dict(), args.save_path)
        print(f"\nSaved trained weights to {args.save_path}")

    evaluate_random_matrix(trained_q_net, sample_bound=args.sample_bound, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
