import math
from typing import Dict, List, Tuple

import numpy as np


"""
Tabular Q-learning experiment for the action of SL(2, Z) on P^1(Q).

State space: projective rational points [p:q] with integers p, q, normalized.
Actions: S, T, T^{-1} as 2x2 matrices.
Goal: map a fixed starting rational point to [0:1].
"""


Action = int
State = Tuple[int, int]  # (p, q) representing [p:q]


# Environment setup (kept in sync conceptually with the DQN script)
INIT_STATE: State = (7, 5)  # starting rational: 7/5
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
DENOM_SHAPING_BONUS = 1.0

# Q-learning hyperparameters
GAMMA = 0.99
ALPHA = 0.1
EPSILON = 0.2
NUM_EPISODES = 200_000


def normalize_projective(p: int, q: int) -> State:
    """Return a canonical representative of [p:q] in P^1(Q)."""
    if q == 0:
        return (1, 0)  # infinity

    if p == 0:
        return (0, 1)  # [0:1]

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


def get_Q(Q: Dict[State, np.ndarray], state: State) -> np.ndarray:
    """Return Q(state, ·) as a length-NUM_ACTIONS array, creating it if needed."""
    if state not in Q:
        Q[state] = np.zeros(NUM_ACTIONS, dtype=float)
    return Q[state]


def epsilon_greedy(Q: Dict[State, np.ndarray], state: State, epsilon: float) -> int:
    """Choose an action using epsilon-greedy with respect to Q."""
    if np.random.rand() < epsilon:
        return int(np.random.randint(NUM_ACTIONS))
    q_values = get_Q(Q, state)
    return int(np.argmax(q_values))


def q_learning_modular(
    num_episodes: int = NUM_EPISODES,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    epsilon: float = EPSILON,
) -> Tuple[Dict[State, np.ndarray], int]:
    """Run Q-learning on the modular projective problem starting from INIT_STATE.

    Returns:
        Q: learned Q-table.
        success_episodes: number of episodes that reached TARGET_STATE.
    """
    Q: Dict[State, np.ndarray] = {}
    success_episodes = 0

    for episode in range(num_episodes):
        state = INIT_STATE
        reached_terminal = False

        for _ in range(MAX_STEPS_PER_EPISODE):
            if is_target(state):
                break

            action = epsilon_greedy(Q, state, epsilon)
            next_state = apply_generator(state, action)

            # Reward shaping: bonus when denominator decreases (excluding infinity)
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

            q_sa = get_Q(Q, state)

            if done:
                best_next = 0.0
            else:
                q_next_arr = get_Q(Q, next_state)
                best_next = float(np.max(q_next_arr))

            td_target = reward + gamma * best_next
            td_error = td_target - q_sa[action]
            q_sa[action] += alpha * td_error

            state = next_state
            if done:
                reached_terminal = is_target(next_state)
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
        f"\nReached target in {success_episodes} / {num_episodes} episodes "
        f"({success_episodes / max(1, num_episodes):.4f} fraction)."
    )
    return Q, success_episodes


def state_values_from_Q(Q: Dict[State, np.ndarray]) -> Dict[State, float]:
    """Approximate state values V(s) ≈ max_a Q(s, a)."""
    return {s: float(np.max(q_values)) for s, q_values in Q.items()}


def show_top_states(
    Q: Dict[State, np.ndarray],
    values: Dict[State, float],
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
            q_str = "[0, 0, 0]"
        else:
            q_str = np.round(q_row, 3)
        p, q = s
        if q == 0:
            z_repr = "∞"
        else:
            z_repr = f"{p}/{q}"
        print(f"s = [{p}:{q}] (z={z_repr}), V(s) ≈ {v:.3f}, Q(s,·) = {q_str}")


def greedy_rollout(
    Q: Dict[State, np.ndarray],
    max_steps: int = MAX_STEPS_PER_EPISODE,
) -> List[Tuple[State, Action | None]]:
    """Follow the greedy policy from INIT_STATE using the learned Q-table."""
    path: List[Tuple[State, Action | None]] = []
    state = INIT_STATE

    for _ in range(max_steps):
        if is_target(state) or is_overflow(state):
            path.append((state, None))
            break

        if state not in Q:
            path.append((state, None))
            break

        q_values = Q[state]
        action = int(np.argmax(q_values))
        path.append((state, action))
        next_state = apply_generator(state, action)
        state = next_state

        if is_target(state) or is_overflow(state):
            path.append((state, None))
            break

    return path


def action_to_symbol(action: Action | None) -> str:
    if action is None:
        return "-"
    if action == ACTION_S:
        return "S"
    if action == ACTION_T:
        return "T"
    if action == ACTION_T_INV:
        return "T^{-1}"
    return "?"


if __name__ == "__main__":
    Q, success_episodes = q_learning_modular()

    values = state_values_from_Q(Q)

    # Always show value at INIT_STATE first, if available.
    print("\nValue at INIT_STATE:")
    if INIT_STATE in values:
        print(f"INIT_STATE = {INIT_STATE}, V(s) ≈ {values[INIT_STATE]:.3f}")
    else:
        print("INIT_STATE not visited; implicit V(s) ≈ 0.000")

    show_top_states(Q, values, k=20)

    rollout = greedy_rollout(Q)
    print("\nGreedy rollout from INIT_STATE:")
    print(f"(INIT_STATE = {INIT_STATE})")
    for idx, (s, a) in enumerate(rollout):
        p, q = s
        if q == 0:
            z_repr = "∞"
        else:
            z_repr = f"{p}/{q}"
        q_row = Q.get(s)
        if q_row is None:
            q_str = "[unvisited]"
        else:
            q_str = np.round(q_row, 3)
        sym = action_to_symbol(a)
        if a is None:
            print(
                f"{idx:02d}: [p:q] = [{p}:{q}], z = {z_repr}, "
                f"Q(s,·) = {q_str} (terminal/overflow/unknown)"
            )
        else:
            print(
                f"{idx:02d}: [p:q] = [{p}:{q}], z = {z_repr}, "
                f"Q(s,·) = {q_str}, a = {sym}"
            )

