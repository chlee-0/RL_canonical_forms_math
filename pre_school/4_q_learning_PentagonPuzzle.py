import numpy as np


DIM = 5
GAMMA = 1.0
STEP_REWARD = -1.0
GOAL_REWARD = 20

ALPHA = 0.1
EPSILON = 0.1
NUM_EPISODES = 200000
MAX_STEPS_PER_EPISODE = 50

#s_init = np.array([-1, 1, -3, 6, 4], dtype=int)
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


def get_Q(Q: dict[tuple[int, ...], np.ndarray], state: tuple[int, ...]) -> np.ndarray:
    """Return Q(state, ·) as a length-5 array, creating it if needed."""
    if state not in Q:
        Q[state] = np.zeros(DIM, dtype=float)
    return Q[state]


def epsilon_greedy(
    Q: dict[tuple[int, ...], np.ndarray],
    state: tuple[int, ...],
    epsilon: float,
) -> int:
    """Choose an action using epsilon-greedy with respect to Q."""
    if np.random.rand() < epsilon:
        return int(np.random.randint(DIM))
    q_values = get_Q(Q, state)
    return int(np.argmax(q_values))


def q_learning_pentagon(
    num_episodes: int = NUM_EPISODES,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    epsilon: float = EPSILON,
) -> tuple[dict[tuple[int, ...], np.ndarray], int]:
    """Run Q-learning on the Pentagon Puzzle starting from s_init.

    Returns:
        Q: learned Q-table.
        success_episodes: number of episodes that reached a success state.
    """
    Q: dict[tuple[int, ...], np.ndarray] = {}
    success_episodes = 0

    for episode in range(num_episodes):
        state = INIT_STATE
        reached_terminal = False

        for _ in range(MAX_STEPS_PER_EPISODE):
            s_arr = np.array(state, dtype=int)
            if is_success(s_arr):
                break

            action = epsilon_greedy(Q, state, epsilon)
            next_arr = apply_action(s_arr, action)
            next_state = tuple(int(x) for x in next_arr)

            done = is_success(next_arr)
            reward = GOAL_REWARD if done else STEP_REWARD

            q_sa = get_Q(Q, state)

            if done:
                best_next = 0.0
            else:
                q_next = get_Q(Q, next_state)
                best_next = float(np.max(q_next))

            td_target = reward + gamma * best_next
            td_error = td_target - q_sa[action]
            q_sa[action] += alpha * td_error

            state = next_state
            if done:
                reached_terminal = True
                break

        if reached_terminal:
            success_episodes += 1

        if (episode + 1) % 1000 == 0:
            frac = success_episodes / float(episode + 1)
            print(
                f"Episode {episode + 1}/{num_episodes} completed | "
                f"cumulative terminal fraction = {frac:.4f}"
            )

    print(
        f"\nReached terminal in {success_episodes} / {num_episodes} episodes "
        f"({success_episodes / max(1, num_episodes):.4f} fraction)."
    )
    return Q, success_episodes


def state_values_from_Q(
    Q: dict[tuple[int, ...], np.ndarray]
) -> dict[tuple[int, ...], float]:
    """Approximate state values V(s) ≈ max_a Q(s, a)."""
    return {s: float(np.max(q_values)) for s, q_values in Q.items()}


def show_top_states(
    Q: dict[tuple[int, ...], np.ndarray],
    values: dict[tuple[int, ...], float],
    k: int = 20,
    min_abs_value: float = 1.0,
) -> None:
    """Print top-k states by value together with their Q-rows."""
    filtered = [(s, v) for s, v in values.items() if abs(v) > min_abs_value]
    if not filtered:
        print(
            f"\nNo states with |V(s)| > {min_abs_value:.1e} "
            "(most values are still ~0)."
        )
        return

    items = sorted(filtered, key=lambda x: x[1], reverse=True)
    print(f"\nTop {min(k, len(items))} states by value (|V| > {min_abs_value:.1e}):")
    for s, v in items[:k]:
        q_row = Q.get(s)
        if q_row is None:
            q_str = "[0, 0, 0, 0, 0]"
        else:
            q_str = np.round(q_row, 3)
        print(f"s = {s}, V(s) ≈ {v:.3f}, Q(s,·) = {q_str}")


def greedy_rollout(
    Q: dict[tuple[int, ...], np.ndarray],
    max_steps: int = 50,
) -> list[tuple[tuple[int, ...], int | None]]:
    """Follow the greedy policy from s_init using the learned Q-table.

    Returns a list of (state, action) pairs; the action is None at the
    terminal/unknown state where rollout stops.
    """
    path: list[tuple[tuple[int, ...], int | None]] = []
    state = INIT_STATE

    for _ in range(max_steps):
        s_arr = np.array(state, dtype=int)

        if is_success(s_arr):
            path.append((state, None))
            break

        if state not in Q:
            # We have no information about this state.
            path.append((state, None))
            break

        q_values = Q[state]
        action = int(np.argmax(q_values))
        path.append((state, action))
        next_arr = apply_action(s_arr, action)
        state = tuple(int(x) for x in next_arr)

        if is_success(next_arr):
            path.append((state, None))
            break

    return path


if __name__ == "__main__":
    Q, success_episodes = q_learning_pentagon()

    values = state_values_from_Q(Q)

    # Always show value at a specific reference state first, if available.
    ref_state = (2, 1, 0, 3, 1)
    print("\nValue at reference state (2, 1, 0, 3, 1):")
    if ref_state in values:
        print(f"s = {ref_state}, V(s) ≈ {values[ref_state]:.3f}")
    else:
        print("state not visited; implicit V(s) ≈ 0.000")

    show_top_states(Q, values, k=20)

    rollout = greedy_rollout(Q)
    print("\nGreedy rollout from s_init:")
    print(f"(s_init = {INIT_STATE})")
    for idx, (s, a) in enumerate(rollout):
        q_row = Q.get(s)
        if q_row is None:
            q_str = "[unvisited]"
        else:
            q_str = np.round(q_row, 3)
        if a is None:
            print(f"{idx:02d}: s = {s}, Q(s,·) = {q_str} (terminal/unknown)")
        else:
            if idx + 1 < len(rollout):
                s_next = rollout[idx + 1][0]
                print(f"{idx:02d}: s = {s}, Q(s,·) = {q_str} --a={a}--> {s_next}")
            else:
                print(f"{idx:02d}: s = {s}, Q(s,·) = {q_str}, a = {a}")
