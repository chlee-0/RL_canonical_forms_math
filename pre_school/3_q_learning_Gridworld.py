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
    reward = STEP_REWARD  # -1 for every step (including the step that reaches the goal)
    return next_state, reward, done


def epsilon_greedy(Q: np.ndarray, state: tuple[int, int], epsilon: float) -> int:
    """Choose an action using epsilon-greedy with respect to Q."""
    if np.random.rand() < epsilon:
        return np.random.randint(NUM_ACTIONS)
    r, c = state
    return int(np.argmax(Q[r, c]))


def q_learning_gridworld(
    num_episodes: int = NUM_EPISODES,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    epsilon: float = EPSILON,
) -> np.ndarray:
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
            best_next = np.max(Q[nr, nc])
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


def print_policy_arrows(policy: np.ndarray) -> None:
    """Pretty-print the greedy policy using arrow symbols."""
    mapping = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    for r in range(GRID_SIZE):
        row = " ".join(mapping[int(a)] for a in policy[r])
        print(row)


def print_q_table(Q: np.ndarray) -> None:
    """Pretty-print Q-values for each state."""
    np.set_printoptions(precision=2, suppress=True)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            print(f"s={(r, c)} Q=[up,down,left,right] {Q[r, c]}")


if __name__ == "__main__":
    Q = q_learning_gridworld(num_episodes=10000)
    policy = greedy_policy_from_Q(Q)
    print("\nLearned Q-table:")
    print_q_table(Q)
    print("\nGreedy policy (arrows):")
    print_policy_arrows(policy)
