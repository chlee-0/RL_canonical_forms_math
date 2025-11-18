# Phase 2: Value Iteration in the Pentagon Puzzle

## Objective

- Transfer the intuition from Gridworld value iteration to a simple mathematical problem.

**Disclaimer.** Throughout this note we keep the math intentionally informal and intuitive; the goal is to build a strong gut feeling for value functions and policies, not to be fully rigorous.

---

## Environment: Pentagon Puzzle

We place labels on the vertices of a regular pentagon:

- Vertices are labelled $0, 1, 2, 3, 4$ in a cycle (indices are always taken **mod 5**).

**State.**  
A state is a length‑5 integer vector
$$s = (s_0, s_1, s_2, s_3, s_4) \in \mathbb{Z}^5.$$
We will think of $s_i$ as the integer written at vertex $i$.


**Action.**  
An action chooses one vertex $i \in \{0,1,2,3,4\}$.  
Given a state $s$ and an action $i$, we update the triple $(i-1, i, i+1)$ (with indices mod 5) as:
$$s_{i-1}' = s_{i-1} + s_i,\quad s_i'     = -s_i,\quad s_{i+1}' = s_{i+1} + s_i,$$
and all other components stay the same. The transition is **deterministic**.


**Termination (Goal).**  
We terminate successfully when all components are non‑negative:
$$\text{success}(s) \iff \forall i,\; s_i \ge 0.$$
In this note we focus on a concrete starting state
$$s_{\text{init}} = (20, 18, -1, -13, -17),$$
and we are interested in reaching some success state from $s_{\text{init}}$ in as few steps as possible.

For this initial state one can show that such a success state is indeed reachable by some finite sequence of moves; in this note we simply assume this fact, but you are encouraged to either accept it for now or try to prove it yourself as a separate exercise.

Even if we start from a single initial point $s_{\text{init}}$, repeatedly applying the transition rule can (in principle) generate infinitely many distinct integer vectors, so the underlying state space is effectively infinite. This is a key difference from the 4×4 Gridworld in Phase 1, where the state space was a fixed finite set of 16 grid cells and we could run value iteration over all states directly.

**Exercise.**  
Starting from the initial state
$$s_{\text{init}} = (20, 18, -1, -13, -17),$$
compute the next state $s'$ explicitly for each action $i = 0,1,2,3,4$ by applying the triple update rule above. (This is the Pentagon analogue of “move up/down/left/right” in Gridworld.) You can also experiment with this update rule interactively by downloading [`pre_school/2_value_iteration_PentagonPuzzle_interactive.html`](../pre_school/2_value_iteration_PentagonPuzzle_interactive.html) from GitHub and then opening the downloaded file in your browser, clicking vertices, or trying different initial states.

### Mathematical viewpoint (optional)

Each action at vertex $i$ is a fixed linear map on $\mathbb{Z}^5$ (it sends $s$ to a new vector obtained by the triple update at $(i-1, i, i+1)$). If we denote these linear maps by $g_0, \dots, g_4$, they generate a subgroup
$$G \subset \mathrm{GL}_5(\mathbb{Z})$$
that acts on $\mathbb{Z}^5$ by $g \cdot s$ (in fact $G$ is an affine Coxeter group of type $\widetilde{A}_4$).

From this point of view, starting from $s_{\text{init}}$ and applying a sequence of actions corresponds to applying a product of generators from $\{g_0,\dots,g_4\}$, and the states we see along an episode are points in the **orbit**
$$G \cdot s_{\text{init}} = \{ g \cdot s_{\text{init}} : g \in G \}.$$

More abstractly, this fits into the general picture of a group action
$$G \times X \to X,\quad (g, x) \mapsto g \cdot x,$$
where here $X = \mathbb{Z}^5$, $G \subset \mathrm{GL}_5(\mathbb{Z})$, and $x = s_{\text{init}}$. The puzzle is to find a product of generators (an element $g \in G$) such that the transformed point $g \cdot x$ lies in a special subset of $X$ (for example, the set of states with all components $\ge 0$).


## Reward Design and Interpretation

To keep things parallel to the Gridworld example, we use:

- Step reward: $R(s, a) = -1$ for every non‑terminal step.
- Success states (all components $\ge 0$) have value $V_*(s) = 0$ and no outgoing actions.
- Discount factor: $\gamma = 1.0$.

Under this design, the optimal value interprets as
$$V_*(s) = - \text{(minimal steps‑to‑success from } s),$$
as long as success is reachable with some sequence of actions.

- States closer (in graph distance) to a success state will have values closer to 0.
- States farther away will have more negative values.

This is directly analogous to the Gridworld interpretation where $V_*(s)$ was (minus) the minimal number of steps to reach the goal.

---

## From Infinite States to a Finite Graph

In Gridworld, the state set $S$ was finite (4×4 grid), so we could run value iteration over all states directly.

In the Pentagon Puzzle:

- The ambient space is $\mathbb{Z}^5$, but starting from $s_{\text{init}}$ and applying our moves we only ever visit states in its orbit $G \cdot s_{\text{init}}$ (which is typically infinite).
- There is no obvious global bound on the entries of $s$.

So instead of trying to handle all of this infinite orbit at once, we:

1. Fix a particular initial state $s_{\text{init}}$.
2. Explore the states that are reachable from $s_{\text{init}}$ **up to some maximum depth** (number of steps).
3. Treat this reachable set as a finite graph and run value iteration on that graph.

Formally, we build a graph
$$G_{0} = (S_{\text{reachable}}, A, T),$$
where
- $S_{\text{reachable}}$ is the set of states reached within a depth cutoff,
- $A(s) = \{0,1,2,3,4\}$ for all non‑terminal states,
- $T(s,a) = s'$ is the deterministic transition defined by the triple update rule.

---

## Value Iteration on the graph of states

Once we have a finite graph of reachable states, the value iteration update is conceptually identical to the Gridworld case.

We start from some initial guess $V_0(s)$ (often all zeros) and update repeatedly:
$$V_{k+1}(s) =
\begin{cases}
0, & \text{if } s \text{ is a success state (all } s_i \ge 0), \\
\displaystyle
\max_{a \in A(s)} \left[ R(s,a) + \gamma \cdot V_k\big(T(s,a)\big) \right],
& \text{otherwise.}
\end{cases}$$

Here:
- $R(s,a) = -1$ for non‑terminal states,
- $\gamma = 1.0$,
- $T(s,a)$ applies the triple update at vertex $a$.

This update converges (on the truncated graph) to an approximation of the optimal value function $V_*(s)$ for states near $s_{\text{init}}$.


---

## Code Sketch (python)

Below is a sketch of how one might implement value iteration for the Pentagon Puzzle by building a finite graph from a fixed initial state. This matches the script `pre_school/2_value_iteration_PentagonPuzzle.py`.

```python
import numpy as np
from collections import deque

DIM = 5
GAMMA = 1.0
STEP_REWARD = -1.0
MAX_DEPTH = 30  # how far we explore from s_init

s_init = np.array([20, 18, -1, -13, -17], dtype=int)

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

def build_graph(max_depth: int = MAX_DEPTH):
    """BFS from s_init up to max_depth to build a deterministic transition graph."""
    init_key = tuple(int(x) for x in s_init)
    graph: dict[tuple[int, ...], dict[int, tuple[int, ...]]] = {}
    depth: dict[tuple[int, ...], int] = {init_key: 0}

    q = deque([init_key])

    while q:
        cur = q.popleft()
        d = depth[cur]

        # Ensure current state exists in graph
        if cur not in graph:
            graph[cur] = {}

        # Do not expand successors from success states
        if is_success(np.array(cur, dtype=int)):
            continue

        if d >= max_depth:
            continue

        # Expand all actions 0..4
        for a in range(DIM):
            s_arr = np.array(cur, dtype=int)
            nxt = apply_action(s_arr, a)
            nxt_key = tuple(int(x) for x in nxt)
            graph[cur][a] = nxt_key

            if nxt_key not in depth:
                depth[nxt_key] = d + 1
                q.append(nxt_key)

    print(f"Built graph with {len(graph)} states (depth ≤ {max_depth}).")
    return graph, init_key


def value_iteration(
    graph, init_key, gamma: float = GAMMA, theta: float = 1e-6, max_iters: int = 10000
):
    """
    Run value iteration on the finite graph with step cost.

    We restrict attention to states that can actually reach a success state
    (all components >= 0) within the truncated graph, and ignore states that
    can never reach success under the depth cutoff.
    """
    # Build reverse graph to find states that can reach success.
    reverse: dict[tuple[int, ...], set[tuple[int, ...]]] = {
        s: set() for s in graph.keys()
    }
    for s, actions in graph.items():
        for nxt_key in actions.values():
            if nxt_key in reverse:
                reverse[nxt_key].add(s)

    # Success states in this truncated graph.
    success_states = [
        s for s in graph.keys() if is_success(np.array(s, dtype=int))
    ]

    if not success_states:
        print("No success state reachable within depth limit; value iteration is trivial.")
        states = list(graph.keys())
        values = np.zeros(len(states), dtype=np.float64)
        init_value = 0.0
        return states, values, init_value

    # Backward reachability: states that can reach some success state.
    good = set(success_states)
    queue = list(success_states)
    while queue:
        cur = queue.pop()
        for pred in reverse[cur]:
            if pred not in good:
                good.add(pred)
                queue.append(pred)

    if init_key not in good:
        print(
            "Warning: s_init cannot reach any success state within depth limit; "
            "values are only defined on states that can reach success."
        )

    # Restrict value iteration to states that can reach success.
    states = [s for s in graph.keys() if s in good]
    index = {s: i for i, s in enumerate(states)}
    values = np.zeros(len(states), dtype=np.float64)

    print(f"Running value iteration on {len(states)} states (reachable to success)...")

    for k in range(max_iters):
        delta = 0.0
        new_values = values.copy()
        for s in states:
            i = index[s]
            s_arr = np.array(s, dtype=int)

            if is_success(s_arr):
                v_new = 0.0
            else:
                best = -np.inf
                for nxt_key in graph.get(s, {}).values():
                    # Only consider successors that are also in the good set.
                    if nxt_key not in index:
                        continue
                    j = index[nxt_key]
                    candidate = STEP_REWARD + gamma * values[j]
                    if candidate > best:
                        best = candidate
                # In principle, states in "good" should have at least one
                # successor in the good set, but we guard just in case.
                v_new = best if best > -np.inf else values[i]

            delta = max(delta, abs(values[i] - v_new))
            new_values[i] = v_new

        values = new_values

        # Simple progress report every few iterations
        if (k + 1) % 10 == 0 or k == 0:
            v_init = values[index[init_key]]
            print(f"Iter {k + 1:4d} | delta = {delta:.3e} | V(s_init) ≈ {v_init:.3f}")

        if delta < theta:
            print(f"Converged in {k + 1} iterations, delta = {delta:.2e}")
            break

    init_value = values[index[init_key]]
    return states, values, init_value


def show_top_states(states, values, k: int = 20):
    """Print top-k states sorted by value (descending)."""
    items = list(zip(states, values))
    items.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {min(k, len(items))} states by value:")
    for s, v in items[:k]:
        print(f"s = {s}, V(s) = {v:.3f}")


graph, init_key = build_graph()
states, values, init_value = value_iteration(graph, init_key)
print(f"\nApproximate value at s_init: V(s_init) ≈ {init_value:.3f}")
show_top_states(states, values, k=20)
```

---

## How to Interpret the Result

- With step reward $-1$ and $\gamma = 1$, $V_*(s)$ again approximates “minus the minimal steps‑to‑success from $s$”, but now on a **graph of integer vectors** rather than a 4×4 grid.
- States that are closer (in the pentagon move graph) to a success state will have values closer to 0.
- The initial state $s_{\text{init}} = (20, 18, -1, -13, -17)$ will have a negative value whose magnitude reflects how many moves are needed (within the explored depth) to reach a non‑negative state.

In this way, the Gridworld intuition about a smooth gradient of values pointing “toward the goal” carries over to the Pentagon Puzzle, even though the state space is no longer a simple finite grid.

---

## Exercises

1. **Finding a solution path using the value table.**  
   By extending the script, find an action sequence $(a_0,\dots,a_{T-1})$ that moves the initial state to a terminal state using the computed value function.

2. **Discovering an explicit “hand algorithm”.**
   Look at the greedy path you obtained in Exercise 1. Can you recognize a simple deterministic rule (for example, “at each step choose the vertex with property X”) that would produce the same sequence of moves from $s_{\text{init}}$ without running value iteration? Try to formulate such an algorithm and explain why it must eventually terminate at a success state (hint: identify some quantity that decreases at every step).
