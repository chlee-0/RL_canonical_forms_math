import numpy as np
from collections import defaultdict
import random
import os
import sys
from datetime import datetime, timezone, timedelta
from contextlib import redirect_stdout

# ==============================
# Reward configuration
# ==============================

class RewardConfig:
    """Central place to configure environment rewards."""
    # Penalty for any step (time pressure)
    STEP = -1
    # Extra penalty for invalid operations (e.g. divide by zero, no pivot)
    INVALID = -1.0
    # Penalty for actions that do not change the state (no-op)
    NOOP = -1.0
    # Reward for reaching RREF
    GOAL = 100.0
    # Shaping reward: improvement in pivot-count (REF conformity)
    PIVOT_GAIN = 5.0
    # Shaping penalty: worsening in pivot-count
    PIVOT_LOSS = -5.0
    # Shaping reward/penalty for change in number of (approximate) zeros
    ZERO_GAIN = 3   # reward per new zero
    ZERO_LOSS = -3  # penalty per lost zero

# ==============================
# Environment: RREF puzzle for a fixed 3x5 matrix
# ==============================

class RREFEnv3x3:
    def __init__(self):
        # Initial 3x5 matrix (float for simplicity)
        self.initial_matrix = np.array([
            [2.0, -1, 3, 0.0, 5],
            [4, 2, -1, 1, 1],
            [-2, 3, 0, 2, -4],
        ], dtype=float)
        # self.initial_matrix = np.array([
        #     [2.0, 4.0, -2.0],
        #     [1.0, 1.0,  1.0],
        #     [3.0, 9.0, -3.0],
        # ], dtype=float)        
        # self.initial_matrix = np.array([
        #     [4, -7, 6],
        #     [-3,5,-4],
        #     [3,-5,3],
        # ], dtype=float)  
        self.initial_matrix = np.array([
            [ 2, -1,  3,  0,  5, -2],
            [ 4,  2, -1,  1,  1,  0],
            [-2,  3,  0,  2, -4,  1],
            [ 3, -2,  4, -1,  0,  3]
        ], dtype=float)


        n_rows, n_cols = self.initial_matrix.shape

        # Build discrete action space:
        #  - Swap(i,j) for 0 <= i < j < n_rows
        #  - Normalize(i,j) for all 0 <= i < n_rows, 0 <= j < n_cols
        #  - Eliminate(i,p) for i != p (rows)
        self.actions = []
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
        self.reset()

    def reset(self):
        """Reset environment to the initial matrix."""
        self.matrix = self.initial_matrix.copy()
        return self._state_key()

    def _state_key(self):
        """Serialize matrix to a hashable state representation.
        We round to avoid tiny floating errors exploding the state space.
        """
        rounded = np.round(self.matrix, decimals=3)  # adjust decimals if needed
        return tuple(map(tuple, rounded.tolist()))

    def _is_rref(self):
        """Check if the current matrix is in RREF."""
        A = np.round(self.matrix, decimals=8)  # numerical stability
        m, n = A.shape
        pivot_col = -1
        seen_nonzero_row = False

        for i in range(m):
            # Find first non-zero in row i
            row = A[i]
            nonzero_indices = [j for j, val in enumerate(row) if abs(val) > 1e-8]

            if not nonzero_indices:
                # Row is all zero
                if seen_nonzero_row:
                    # OK: zero rows are allowed at bottom
                    continue
                else:
                    # Could still be all-zero matrix; just continue
                    continue
            else:
                seen_nonzero_row = True
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

    def _pivot_count(self, matrix):
        """Count how many leading rows satisfy REF pivot structure.

        This implements the pivot-count idea:
        - For each row, find its leading non-zero column (if any).
        - Maintain prev_col; each time lead_col > prev_col, increase count.
        - Stop when a zero row appears or the REF rule is violated.
        """
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
                # Zero row: in strict REF, all following rows must also be zero.
                break

            if lead_col > prev_col:
                count += 1
                prev_col = lead_col
            else:
                # REF rule violated (pivot not strictly to the right); stop here.
                break

        return count

    def _zero_count(self, matrix):
        """Count entries that are (approximately) zero in the matrix."""
        A = np.round(matrix, decimals=8)
        return int(np.sum(np.abs(A) <= 1e-8))

    def step(self, action_idx):
        """Apply action and return (next_state, reward, done, info)."""
        action = self.actions[action_idx]
        op_type, a, b = action
        A = self.matrix

        reward = RewardConfig.STEP  # base step penalty
        done = False

        # Keep a copy to detect no-op
        before = A.copy()
        pivot_before = self._pivot_count(before)
        zero_before = self._zero_count(before)

        if op_type == "swap":
            i, j = a, b
            # Swap rows i and j
            A[[i, j], :] = A[[j, i], :]

        elif op_type == "normalize":
            i, j = a, b
            pivot_val = A[i, j]
            if abs(pivot_val) < 1e-8:
                # Invalid: cannot normalize by zero
                reward = RewardConfig.INVALID
            else:
                A[i, :] = A[i, :] / pivot_val

        elif op_type == "eliminate":
            i, p = a, b
            # Use row p as a pivot row.
            # Find first non-zero entry in row p (pivot column)
            row_p = A[p]
            nonzero_indices = [col for col, val in enumerate(row_p) if abs(val) > 1e-8]
            if not nonzero_indices:
                # No pivot in row p -> invalid elimination
                reward = RewardConfig.INVALID
            else:
                j = nonzero_indices[0]  # pivot column
                pivot_val = A[p, j]
                if abs(pivot_val) < 1e-8:
                    reward = RewardConfig.INVALID
                else:
                    factor = A[i, j]
                    # R_i <- R_i - factor * R_p
                    A[i, :] = A[i, :] - factor * A[p, :]

        # Check if matrix actually changed
        if np.allclose(before, A, atol=1e-8):
            # No change -> treat as bad move
            reward = RewardConfig.NOOP

        # Check for terminal condition
        if self._is_rref():
            reward = RewardConfig.GOAL
            done = True

        # Pivot-count based shaping reward
        pivot_after = self._pivot_count(A)
        if pivot_after > pivot_before:
            reward += RewardConfig.PIVOT_GAIN  # moved closer to valid REF
        elif pivot_after < pivot_before:
            reward += RewardConfig.PIVOT_LOSS  # moved further from valid REF

        # Zero-count based shaping reward
        zero_after = self._zero_count(A)
        delta_zero = zero_after - zero_before
        if delta_zero > 0:
            reward += RewardConfig.ZERO_GAIN * delta_zero
        elif delta_zero < 0:
            reward += RewardConfig.ZERO_LOSS * (-delta_zero)

        self.matrix = A
        next_state = self._state_key()
        return next_state, reward, done, {}

# ==============================
# Q-learning implementation
# ==============================

class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.n_actions = n_actions
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        # Q-table: dict mapping state -> np.array of action values
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))

    def select_action(self, state_key):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            q_values = self.Q[state_key]
            # argmax with random tie-breaking
            max_q = np.max(q_values)
            candidates = np.where(q_values == max_q)[0]
            return int(np.random.choice(candidates))

    def update(self, state_key, action, reward, next_state_key, done):
        """Standard Q-learning update."""
        q_values = self.Q[state_key]
        q_next = self.Q[next_state_key]

        best_next = np.max(q_next) if not done else 0.0
        td_target = reward + self.gamma * best_next
        td_error = td_target - q_values[action]
        q_values[action] += self.alpha * td_error

# ==============================
# Training loop
# ==============================

def train(max_episodes=5000,
          max_steps=30,
          verbose_every=500,
          target_success_rate=None):
    """Train Q-learning agent.

    Training stops either when max_episodes is reached, or (if
    target_success_rate is not None) when the goal_rate over the
    last verbose_every episodes exceeds target_success_rate.
    """
    env = RREFEnv3x3()
    agent = QLearningAgent(n_actions=env.n_actions,
                           alpha=0.1,
                           gamma=0.95,
                           epsilon=0.2)
    # Counters over the current logging window
    success_in_window = 0   # number of episodes that reached goal
    reward_in_window = 0.0  # sum of total_reward over episodes
    steps_in_window = 0     # sum of steps taken over episodes

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        reached_goal = False

        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                reached_goal = True
                break

        episode_steps = t + 1
        if reached_goal:
            success_in_window += 1
        reward_in_window += total_reward
        steps_in_window += episode_steps

        # Simple logging
        if verbose_every is not None and episode % verbose_every == 0:
            success_rate = success_in_window / verbose_every
            # Averages over the window
            avg_reward = reward_in_window / verbose_every
            avg_steps = steps_in_window / verbose_every
            print(
                f"Episode {episode}, "
                f"avg_total_reward(last {verbose_every}) = {avg_reward:.2f}, "
                f"avg_steps(last {verbose_every}) = {avg_steps:.2f}, "
                f"goal_rate(last {verbose_every}) = {success_rate:.3f}, "
                f"epsilon = {agent.epsilon:.3f}"
            )
            if (target_success_rate is not None and
                    success_rate >= target_success_rate):
                print(
                    f"Stopping training: target_success_rate="
                    f"{target_success_rate:.3f} reached."
                )
                break
            success_in_window = 0
            reward_in_window = 0.0
            steps_in_window = 0

    return env, agent

# ==============================
# Inspect learned behavior
# ==============================

class Tee:
    """File-like object that duplicates writes to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

def print_action_space(env):
    """Print mapping from action index to (op_type, i, j)."""
    print("\n=== Action space (index -> action) ===")
    for idx, action in enumerate(env.actions):
        print(f"{idx}: {action}")

def run_greedy_episode(env, agent, max_steps=20):
    """Run one episode with greedy policy and show Q-values at each step."""
    state = env.reset()
    print("Initial state:")
    print(np.array(state))

    for t in range(max_steps):
        q_values = agent.Q[state]
        action = int(np.argmax(q_values))
        op_type, a, b = env.actions[action]

        print(f"\nStep {t+1}:")
        print("Q-values at this state:")
        for idx, q in enumerate(q_values):
            print(f"  a={idx:2d} {env.actions[idx]} -> {q:.4f}")
        print(f"Greedy action = {op_type}, args = ({a}, {b})")

        next_state, reward, done, _ = env.step(action)
        print("Next matrix:")
        print(np.array(next_state))
        print(f"Reward: {reward:.2f}, Done: {done}")

        state = next_state
        if done:
            break

def _find_action_index(env, op_type, a, b):
    """Find the index of a specific (op_type, a, b) action in env.actions.

    For \"swap\" actions we treat (i, j) and (j, i) as the same,
    since the action list only stores i < j.
    """
    if op_type == "swap" and a > b:
        a, b = b, a
    for idx, act in enumerate(env.actions):
        if act == (op_type, a, b):
            return idx
    raise ValueError(f"Action {(op_type, a, b)} not found in env.actions")

def run_known_rref_procedure():
    """Demonstrate a generic Gauss-Jordan RREF procedure using env actions."""
    env = RREFEnv3x3()
    state = env.reset()
    print("Initial state (known algorithm demo):")
    print(np.array(state))

    A = env.matrix
    m, n = A.shape
    row = 0
    step_id = 0
    eps = 1e-8

    # Standard Gauss-Jordan elimination, implemented via env actions.
    for col in range(n):
        if row >= m:
            break

        # Find a pivot row at or below current row with non-zero in this column.
        pivot_row = None
        for r in range(row, m):
            if abs(A[r, col]) > eps:
                pivot_row = r
                break

        if pivot_row is None:
            # No pivot in this column; move to next column.
            continue

        # 1) Swap pivot row into current 'row' position if needed.
        if pivot_row != row:
            action_idx = _find_action_index(env, "swap", pivot_row, row)
            step_id += 1
            print(f"\n[Known algorithm] Step {step_id}: action = swap, args = ({pivot_row}, {row})")
            state, reward, done, _ = env.step(action_idx)
            print("Matrix:")
            print(np.array(state))
            print(f"Reward: {reward:.2f}, Done: {done}")

        # 2) Normalize pivot row so that pivot at (row, col) becomes 1.
        action_idx = _find_action_index(env, "normalize", row, col)
        step_id += 1
        print(f"\n[Known algorithm] Step {step_id}: action = normalize, args = ({row}, {col})")
        state, reward, done, _ = env.step(action_idx)
        print("Matrix:")
        print(np.array(state))
        print(f"Reward: {reward:.2f}, Done: {done}")

        # 3) Eliminate this pivot column from all other rows.
        for r in range(m):
            if r == row:
                continue
            if abs(A[r, col]) <= eps:
                continue  # already zero; skip to avoid no-op
            action_idx = _find_action_index(env, "eliminate", r, row)
            step_id += 1
            print(f"\n[Known algorithm] Step {step_id}: action = eliminate, args = ({r}, {row})")
            state, reward, done, _ = env.step(action_idx)
            print("Matrix:")
            print(np.array(state))
            print(f"Reward: {reward:.2f}, Done: {done}")

        # Move to next row for the next pivot.
        row += 1

if __name__ == "__main__":
    # Build log file path based on current KST in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kst = timezone(timedelta(hours=9))
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(script_dir, f"q_learning_log_{timestamp}_kst.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee):
            print(f"# q_learning.py log created at (KST) {datetime.now(kst).isoformat()}")

            # Train the agent (stop early if target_success_rate is reached)
            env, agent = train(
                max_episodes=500000,
                max_steps=50,
                verbose_every=1000,
                target_success_rate=0.95,
            )

            # Show the standard hand-crafted RREF procedure first
            print("\n=== Known RREF algorithm (hand-crafted) ===")
            run_known_rref_procedure()

            # Show the action ordering
            print_action_space(env)

            # Test the greedy policy and show Q-values at each step
            print("\n=== Greedy episode after training ===")
            run_greedy_episode(env, agent, max_steps=20)
