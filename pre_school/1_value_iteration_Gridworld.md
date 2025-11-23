# Phase 1: Value Iteration in Gridworld

## Objective

- Understand the idea of state value functions $V_{*}(s)$ as representing distance to the goal.
- Learn how to compute optimal values and policies using value iteration (a method based on recursive updates).
- Apply the algorithm to a simple gridworld example where all transitions are deterministic and safe.

**Disclaimer.** Throughout this note we keep the math intentionally informal and intuitive; the goal is to build a strong gut feeling for value functions and policies, not to be fully rigorous.

---

## Environment: 4x4 Gridworld

- Agent can move up, down, left, right.
- Every step gives a reward of -1.
- Episode ends when the agent reaches the goal.
- There are no pits or losing states.
- If the agent tries to move outside the grid, it stays in place and still receives the step reward of -1.
- $\gamma$ (discount factor) = 1.0 to reflect true step counts.

```text
S . . .
. . . .
. . . .
. . . G
```

- S = Start (0, 0)
- G = Goal (3, 3)

We define the finite set of all possible states as $S$, where each $s \in S$ represents a grid cell (e.g., $(r, c)$). Each state $s$ has an associated action set $A(s)$ (here: up, down, left, right). Actions deterministically move the agent. If it hits a wall, the agent stays in place with certainty and receives the step reward.




## What is a Policy?

A **policy** tells the agent how to choose actions in each state.

- A **deterministic policy** is a function $\pi : S \to A$ that picks a single action in each state (e.g., $\pi(s) = \text{right}$).
- A **stochastic policy** is a distribution over actions $\pi(a \mid s) = \mathbb{P}(A_t = a \mid S_t = s)$.

In general we write $\pi(a \mid s)$; a deterministic policy is the special case where all probability mass is on one action.



## What is the Value Function?

Given a policy $\pi$, we define the **value function** $V_\pi(s)$ as the expected return (cumulative reward) when starting from a state $s$ and following $\pi$ thereafter.

Formally,
$$V_{\pi}(s) = \mathbb{E}_{\pi} \left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t) \middle| s_0 = s \right]$$
- $s$ is a state in the finite set $S$.
- $a_t \in A(s_t)$ is an action chosen from the available actions at time $t$ (according to $\pi$).
- $R(s_t, a_t)$ is the reward for taking action $a_t$ in state $s_t$.
- $\gamma \in [0,1]$ is a discount factor.

This function tells us how desirable a state is, assuming we follow policy $\pi$.

We are especially interested in the **optimal value function**
$$V_{*}(s) = \max_\pi V_\pi(s),$$
which gives the best achievable value at each state.

Informally, you can think of this as
$$V_*(s) = \mathbb{E} \left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t) \middle| s_0 = s, \text{ choose actions in an optimal way} \right],$$
where "optimal" means we always pick actions to maximize this expected return.

Knowing $`V_*(s)`$ is powerful: we can recover an **optimal policy** by acting greedily with respect to these values. In this deterministic Gridworld, that means
$$\pi_*(s) = \arg\max_{a \in A(s)} \left[ R(s, a) + \gamma \cdot V_*(s') \right],$$
where $s'$ is the next state reached from $s$ by taking action $a$.



## What is Value Iteration?

To compute $V_*(s)$ in practice, we start from some initial guess $V_0(s)$ (often all zeros) and repeatedly update each state's value. At iteration $k+1$ we ask: "If I start in this state and take the best possible action, what total reward will I get?"

$$V_{k+1}(s) = \max_{a \in A(s)} \left[ R(s, a) + \gamma \cdot V_k(s') \right]$$

In words: "The new value of a state is the best expected return if I take the best possible action."

This gives us the **optimal value function** $V_*(s)$: the maximum return achievable from any policy.

Once values have converged to (an approximation of) $V_*(s)$, we can use the greedy optimal policy $\pi_*(s)$ defined in the previous section.

> Later, when we study **policy iteration**, we will start from a fixed policy $\pi$ and compute $V_\pi(s)$, the value function under that policy. Then, we improve $\pi$, recompute $V_\pi(s)$, and iterate. This process also converges to $V_*(s)$.

---

## Code Example (python)

You can run this example as a script from `pre_school/1_value_iteration_Gridworld.py`.

```python
import numpy as np

# 4x4 gridworld
grid_size = 4

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

# Discount factor γ and small threshold for convergence
gamma = 1.0
theta = 1e-4

# Reward for every non-terminal step
step_reward = -1.0

def is_terminal(state):
    # Terminal state is the goal position G
    return state == goal_state

def next_states(state):
    # For each action (up, down, left, right), compute the next state
    r, c = state
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U, D, L, R
    result = []
    for dr, dc in actions:
        nr, nc = r + dr, c + dc
        # If moving would leave the grid, stay in place
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            result.append((nr, nc))
        else:
            result.append((r, c))
    return result

# Perform value iteration updates
for k in range(1000):
    delta = 0.0  # track maximum change in V across states
    new_V = V.copy()
    for r in range(grid_size):
        for c in range(grid_size):
            s = (r, c)
            if is_terminal(s):
                # By convention, the terminal state's value stays fixed (here 0)
                continue
            values = []
            # Bellman backup: look at all possible next states for each action
            for s_next in next_states(s):
                # One-step lookahead: R(s, a) + γ * V_k(s')
                values.append(step_reward + gamma * V[s_next])
            # Take the maximum over actions
            new_V[s] = max(values)
            delta = max(delta, abs(V[s] - new_V[s]))
    V = new_V

    # Show the updated value function V_{k+1} at this iteration
    print(f"Iteration {k+1} (V_{k+1}):")
    print(V)

    # Stop if the update is very small everywhere
    if delta < theta:
        break

print(V)
```

---

## How to Interpret the Result

- Each value V(s) tells you how far (in steps) you are from the goal.
- States closer to the goal will have values closer to 0.
- These values form a smooth gradient pointing toward the goal.

---

## Exercises

1. Move the goal to (2,2) and re-run the algorithm. How does V change?
2. What happens if step reward is changed to -0.1?

---
