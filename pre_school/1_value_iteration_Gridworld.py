import numpy as np


def value_iteration_gridworld(
    grid_size: int = 4,
    gamma: float = 1.0,
    step_reward: float = -1.0,
    theta: float = 1e-4,
    max_iters: int = 1000,
) -> np.ndarray:
    """
    Simple value iteration on a deterministic 4x4 GridWorld.

    Matches the setup in pre_school/1_value_iteration_Gridworld.md:
    - 4x4 grid
    - Start (S) at top-left, goal (G) at bottom-right
    - Reward -1 on every non-terminal step
    - γ = 1.0
    """

    # Start (S) at top-left, goal (G) at bottom-right:
    # S . . .
    # . . . .
    # . . . .
    # . . . G
    start_state = (0, 0)
    goal_state = (grid_size - 1, grid_size - 1)

    # Initialize value function V_0(s) = 0 for all states
    V = np.zeros((grid_size, grid_size))
    print("Initial values V_0:")
    print(V)

    def is_terminal(state: tuple[int, int]) -> bool:
        # Terminal state is the goal position G
        return state == goal_state

    def next_states(state: tuple[int, int]) -> list[tuple[int, int]]:
        # For each action (up, down, left, right), compute the next state
        r, c = state
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U, D, L, R
        result: list[tuple[int, int]] = []
        for dr, dc in actions:
            nr, nc = r + dr, c + dc
            # If moving would leave the grid, stay in place
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                result.append((nr, nc))
            else:
                result.append((r, c))
        return result

    # Perform value iteration updates
    for k in range(max_iters):
        delta = 0.0  # track maximum change in V across states
        new_V = V.copy()
        for r in range(grid_size):
            for c in range(grid_size):
                s = (r, c)
                if is_terminal(s):
                    # By convention, the terminal state's value stays fixed (here 0)
                    continue
                values: list[float] = []
                # Bellman backup: look at all possible next states for each action
                for s_next in next_states(s):
                    # One-step lookahead: R(s, a) + γ * V_k(s')
                    values.append(step_reward + gamma * V[s_next])
                # Take the maximum over actions
                new_V[s] = max(values)
                delta = max(delta, abs(V[s] - new_V[s]))
        V = new_V

        # Show the updated value function V_{k+1} at this iteration
        print(f"Iteration {k + 1} (V_{k+1}):")
        print(V)

        # Stop if the update is very small everywhere
        if delta < theta:
            print(f"Converged in {k + 1} iterations.")
            break

    return V


if __name__ == "__main__":
    V_star = value_iteration_gridworld()
    print("Optimal state values V_* for 4x4 GridWorld:")
    print(V_star)
