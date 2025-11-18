import numpy as np


GRID_SIZE = 4
NUM_ACTIONS = 4  # up, down, left, right

START_STATE = (0, 0)
GOAL_STATE = (GRID_SIZE - 1, GRID_SIZE - 1)

GAMMA = 1.0
ALPHA = 0.1
EPSILON = 0.1
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 100
STEP_REWARD = -1.0


def is_terminal(state: tuple[int, int]) -> bool:
    return state == GOAL_STATE


def step(state: tuple[int, int], action: int) -> tuple[tuple[int, int], float, bool]:
    """Take one step in the 4x4 Gridworld."""
    r, c = state
    if action == 0:  # up
        dr, dc = -1, 0
    elif action == 1:  # down
        dr, dc = 1, 0
    elif action == 2:  # left
        dr, dc = 0, -1
    else:  # action == 3, right
        dr, dc = 0, 1

    nr, nc = r + dr, c + dc
    # If moving would leave the grid, stay in place
    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
        nr, nc = r, c

    next_state = (nr, nc)
    done = is_terminal(next_state)
    reward = STEP_REWARD  # -1 on every step (including the last move)
    return next_state, reward, done


def epsilon_greedy(Q: np.ndarray, state: tuple[int, int], epsilon: float) -> int:
    """Choose an action using epsilon-greedy with respect to Q."""
    if np.random.rand() < epsilon:
        return int(np.random.randint(NUM_ACTIONS))
    r, c = state
    return int(np.argmax(Q[r, c]))


def q_learning_gridworld(
    num_episodes: int = NUM_EPISODES,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Run Q-learning on the 4x4 Gridworld and return the learned Q-table."""
    # Q has shape (GRID_SIZE, GRID_SIZE, NUM_ACTIONS)
    Q = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS), dtype=float)

    for episode in range(num_episodes):
        state = START_STATE

        for _ in range(MAX_STEPS_PER_EPISODE):
            if is_terminal(state):
                break

            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = step(state, action)

            r, c = state
            nr, nc = next_state

            # TD target using max over next-state actions (off-policy)
            best_next = float(np.max(Q[nr, nc]))
            td_target = reward + gamma * best_next
            td_error = td_target - Q[r, c, action]

            # Q-learning update
            Q[r, c, action] += alpha * td_error

            state = next_state
            if done:
                break

        # Optional: simple progress print
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed")

    return Q


def greedy_policy_from_Q(Q: np.ndarray) -> np.ndarray:
    """Return a 2D array of greedy actions from Q."""
    policy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            policy[r, c] = int(np.argmax(Q[r, c]))
    return policy


def state_values_from_Q(Q: np.ndarray) -> np.ndarray:
    """Approximate state values V(s) ≈ max_a Q(s, a)."""
    return np.max(Q, axis=2)


def print_value_table(V: np.ndarray) -> None:
    """Pretty-print the state value table."""
    print("State values V(s) ≈ max_a Q(s, a):")
    print(np.round(V, 2))


def print_policy(policy: np.ndarray) -> None:
    """Print greedy policy both as action indices and arrows."""
    print("Greedy policy (0=up,1=down,2=left,3=right):")
    print(policy)

    symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    print("Greedy policy as arrows:")
    for r in range(GRID_SIZE):
        row = [symbols[int(policy[r, c])] for c in range(GRID_SIZE)]
        print(" ".join(row))


if __name__ == "__main__":
    Q = q_learning_gridworld()

    # Show basic information about the learned Q-table
    print("Q-table shape:", Q.shape)
    print(f"Q at START_STATE {START_STATE}:")
    print(Q[START_STATE[0], START_STATE[1]])

    # Derive and print state value table V(s) ≈ max_a Q(s, a)
    V = state_values_from_Q(Q)
    print_value_table(V)

    # Derive and print greedy policy
    policy = greedy_policy_from_Q(Q)
    print_policy(policy)
