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
