# Phase 4: Q-Learning in the Pentagon Puzzle

## Objective

- Transfer Q-learning from a small Gridworld (Phase 3) to the **Pentagon Puzzle**.
- Learn how to run **model-free control** in a state space that is (in principle) infinite.
- Represent a **Q-table on a graph of integer vectors** using Python dictionaries.
- See how $Q(s, a)$ again approximates “minus the expected steps-to-success”.

**Disclaimer.** As in previous phases, we keep the math informal and intuition‑driven. The goal is to build a solid gut feeling for Q-learning in a more abstract environment, not to be fully rigorous.

---

## Environment Recap: Pentagon Puzzle

We reuse the environment from Phase 2.

**States.**  
Each state is a 5‑dimensional integer vector
$$s = (s_0, s_1, s_2, s_3, s_4) \in \mathbb{Z}^5,$$
thought of as labels on the vertices of a regular pentagon.

**Actions.**  
An action chooses a vertex $i \in \{0,1,2,3,4\}$.  
Given a state $s$ and an action $i$, we update the triple $(i-1, i, i+1)$ (indices mod 5):
$$
s_{i-1}' = s_{i-1} + s_i,\quad
s_i'     = -s_i,\quad
s_{i+1}' = s_{i+1} + s_i,
$$
and all other coordinates stay unchanged. The transition is **deterministic**.

**Termination (Goal).**  
We terminate successfully when all components are non‑negative:
$$\text{success}(s) \iff \forall i,\; s_i \ge 0.$$

In Phase 2 we focused on the initial state
$$s_{\text{init}} = (20, 18, -1, -13, -17).$$

We will **reuse the same initial state** $s_{\text{init}}$ for Q-learning as well, so that we can compare value-iteration and Q-learning behavior on exactly the same puzzle. (Of course, in code you can easily change `s_init` to experiment with other starting points.)

In Phase 2 we assumed (and one can prove) that some finite sequence of actions takes $s_{\text{init}}$ to a success state. However, the overall state space is infinite: repeated moves can generate arbitrarily large positive and negative entries.

---

## Reward Design and Interpretation

To parallel earlier phases, we keep a simple reward design but add a **terminal bonus**:

- Step reward: $R(s, a) = -1$ for every non‑terminal step.
- Terminal reward: on the step that first enters a success state, we give an extra reward $+R_{\text{goal}}$ (in the code example we use $R_{\text{goal}} = 20$).
- Success states (all components $\ge 0$) are terminal and have no outgoing actions.
- Discount factor: $\gamma = 1.0$.

Under this design, for an episode that reaches success in $T$ steps, the return looks like
$$
G_0 \approx -T + R_{\text{goal}},
$$
so among successful paths **짧은 경로일수록** 여전히 유리합니다.  
또한 성공을 전혀 못 하는 경로는 $-T$만 누적하고 끝나므로, 터미널 보너스 덕분에 “성공으로 이어지는 경로”가 더 빠르게 식별되고 Q 값에 강하게 반영됩니다.

---

## Why Q-Learning Here?

In Phase 2, we:

1. Fixed $s_{\text{init}}$.
2. Ran a bounded BFS to some `MAX_DEPTH` to collect a finite set of reachable states.
3. Ran **value iteration** on this finite graph to approximate $V_*(s)$ near $s_{\text{init}}$.

This required us to:

- Build an explicit graph of transitions.
- Use dynamic programming sweeps over all collected states.

In Phase 4 (Q-learning), we instead:

- Treat the environment as a **black box** providing samples $(s, a, r, s')$.
- Do **not** pre‑build a full graph of reachable states.
- Learn $Q(s, a)$ **online** from episodes starting at $s_{\text{init}}$.

This is closer to the typical RL scenario where we may not know the dynamics in advance, or the reachable state set may be very large.

---

## Action-Value Function for the Pentagon Puzzle

As in Phase 3, we use the **action-value function**:

$$
Q_\pi(s, a)
  = \mathbb{E}_\pi\Big[\sum_{t=0}^{T} \gamma^t R_{t+1}
     \,\Big|\, S_0 = s, A_0 = a\Big],
$$

and its optimal version
$$
Q_*(s, a) = \max_\pi Q_\pi(s, a).
$$

An optimal policy is obtained greedily:
$$
\pi_*(s) = \arg\max_{a} Q_*(s, a).
$$

With our step reward $-1$ and $\gamma = 1$, we can again think of $Q_*(s, a)$ as:

- “minus the expected number of moves to reach a success state,  
   if we take action $a$ now at $s$ and then act optimally.”

---

## Q-Learning Algorithm (Conceptually)

We run episodes that start from $s_{\text{init}}$ and end when we either:

- Reach a success state, or
- Hit a maximum step limit (to avoid infinite wandering).

Within each episode:

1. Initialize $s \leftarrow s_{\text{init}}$.
2. At each step:
   - Choose an action $a \in \{0,1,2,3,4\}$ using an $\epsilon$-greedy strategy.
   - Apply the triple update rule to get $s'$.
   - Observe reward $r = -1$.
   - Update $Q(s, a)$ via the Q-learning rule.
   - Set $s \leftarrow s'$ and continue.

The **Q-learning update** is exactly as in Phase 3:

$$
Q_{t+1}(s_t, a_t)
=
Q_t(s_t, a_t)
+
\alpha \Bigl[
  r_{t+1}
  +
  \gamma \max_{a'} Q_t(s_{t+1}, a')
  -
  Q_t(s_t, a_t)
\Bigr],
$$

with learning rate $\alpha \in (0, 1]$.

Because the puzzle is deterministic, the only randomness comes from our exploration policy (the $\epsilon$-greedy choices).

---

## Representing the Q-Table

In 4×4 Gridworld, the state space was small and fixed, so we could store $Q$ in a 3D array `Q[4,4,4]`.

In the Pentagon Puzzle:

- The state space (orbit of $s_{\text{init}}$) is potentially infinite.
- We do not know in advance which states we will visit.

Instead, we use a **dictionary-based** Q-table:

- Keys are states as 5‑tuples: `state = (s0, s1, s2, s3, s4)`.
- Values are NumPy arrays of length 5: one Q-value per action.

In Python:

```python
Q: dict[tuple[int, ...], np.ndarray] = {}

def get_Q(Q, state):
    key = tuple(int(x) for x in state)
    if key not in Q:
        Q[key] = np.zeros(DIM, dtype=float)
    return Q[key]
```

This way:

- We only allocate Q-values for states we actually visit.
- The Q-table naturally grows around the region of state space explored from $s_{\text{init}}$.

---

## Code Example (python)

Below is a self-contained Q-learning script for the Pentagon Puzzle.  
You can save this as `pre_school/4_q_learning_PentagonPuzzle.py` and run it with Python.  
It mirrors the value-iteration setup in `2_value_iteration_PentagonPuzzle.py` but learns from episodes instead of building the full graph first.

```python
import numpy as np

DIM = 5
GAMMA = 1.0
STEP_REWARD = -1.0
GOAL_REWARD = 20.0

ALPHA = 0.1
EPSILON = 0.1
NUM_EPISODES = 200000
MAX_STEPS_PER_EPISODE = 50

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


def epsilon_greedy(Q: dict[tuple[int, ...], np.ndarray],
                   state: tuple[int, ...],
                   epsilon: float) -> int:
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
            print(f"Episode {episode + 1}/{num_episodes} completed")

    print(
        f"\nReached terminal in {success_episodes} / {num_episodes} episodes "
        f"({success_episodes / max(1, num_episodes):.4f} fraction)."
    )
    return Q, success_episodes


def state_values_from_Q(Q: dict[tuple[int, ...], np.ndarray]) -> dict[tuple[int, ...], float]:
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


def greedy_rollout(Q: dict[tuple[int, ...], np.ndarray],
                   max_steps: int = 50) -> list[tuple[int, ...]]:
    """Follow the greedy policy from s_init using the learned Q-table."""
    path: list[tuple[int, ...]] = []
    state = INIT_STATE

    for _ in range(max_steps):
        path.append(state)
        s_arr = np.array(state, dtype=int)

        if is_success(s_arr):
            break

        if state not in Q:
            # We have no information about this state.
            break

        q_values = Q[state]
        action = int(np.argmax(q_values))
        next_arr = apply_action(s_arr, action)
        state = tuple(int(x) for x in next_arr)

        if is_success(next_arr):
            path.append(state)
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
    for s in rollout:
        print(s)
```

---

## How to Interpret the Result

- The learned $Q(s, a)$ approximates $Q_*(s, a)$ for states in the region explored from $s_{\text{init}}$.
- The derived approximate value function
  $$V(s) \approx \max_a Q(s, a)$$
  plays the same role as in Phase 2: more negative means farther from success, closer to 0 means nearer.
- The `greedy_rollout` produced after learning is an approximate solution path from $s_{\text{init}}$ to a success state, analogous to following a greedy policy on the value table.

Because we learn from sampled episodes rather than exhaustive graph construction, different runs (with different random seeds) may produce slightly different Q-tables and paths, but they should share the same qualitative behavior.

---

## Practice Exercises

1. **Compare with value iteration.**  
   Run both `2_value_iteration_PentagonPuzzle.py` and the Q-learning script. How do the approximate values and solution paths compare?

2. **Vary hyperparameters.**  
   Experiment with different learning rates $\alpha$, exploration rates $\epsilon$, and numbers of episodes. How do they affect convergence speed and path quality?

3. **Inspect Q for the initial state.**  
   Print `Q[INIT_STATE]` after training. Which vertex actions look best? Can you interpret this in terms of the puzzle moves?

4. **Try different initial states.**  
   Change `s_init` to other vectors and see whether Q-learning can still discover success paths within a reasonable number of episodes, and how the learned values look.
