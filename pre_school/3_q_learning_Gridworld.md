# Phase 3: Q-Learning in Gridworld

## Objective


- In Phases 1 and 2, we could compute good behavior because we **knew exactly how the environment moves** and could run value iteration over all states.
  In this phase, we ask a different question: can we learn similar behavior **just from trial-and-error experience**, without using the transition rules?
- To do this, we introduce the **action-value function** $Q(s, a)$, which tells us how good it is to take action $a$ in state $s$ and then act well afterward.
- We implement the basic **Q-learning** update rule and combine it with simple $\epsilon$-greedy exploration.
- Finally, we apply Q-learning to the **same 4×4 Gridworld** from Phase 1 and compare the learned behavior to the value-iteration solution.

> (Aside)  
> In reinforcement learning, solving a task using a full model of the environment (as in Phase 1 value iteration)
> is often called **planning**.  
> Learning directly from experience without using the model, as we do here, is called **model‑free learning**.
> You don’t need to remember these terms yet; they are just names for ideas you are already seeing.



**Disclaimer.** As before, the math will be informal and intuition‑oriented. The goal is to build a strong gut feeling for how Q-learning works, not to prove theorems.

---

## Environment: 4×4 Gridworld (Same as Phase 1)

We use the same 4×4 Gridworld as in Phase 1 (start state $S=(0,0)$, goal state $G=(3,3)$, four moves up/down/left/right, reward $-1$ per step, $\gamma=1.0$; see Phase 1 for details).

```text
S . . .
. . . .
. . . .
. . . G
```

The key difference is how we use the environment:

- In Phase 1, we assumed full knowledge of transitions $P(s' \mid s, a)$ and could run **value iteration** over the entire state space.
- In Phase 3, we imagine the agent does not know the transition rules.
It only sees samples $(s_t, a_t, r_{t+1}, s_{t+1})$ by interacting with the environment.

We want to learn good behavior *directly from experience*, without building an explicit model of the environment.

---

## From State Values to Action Values

In Phase 1 we focused on the **state value function**:
$$V_\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t) \mid s_0 = s\right],$$
and especially on the **optimal** value function $V_*(s)$.

Now we introduce the **action-value function** (or **Q-function**):
$$Q_\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t) \mid s_0 = s, a_0 = a\right].$$

Intuitively:

- $Q_\pi(s, a)$ = “How good is it to *take action $a$ in state $s$* and then follow policy $\pi$ afterward?”
- $Q_*(s, a) = \max_\pi Q_\pi(s, a)$ is the best return achievable if we start from $(s, a)$ and act optimally.

We can recover an optimal policy greedily:
  $$\pi_*(s) = \arg\max_{a} Q_*(s, a).$$

---

## The Q-Learning Update Rule

At time $t$, suppose the agent is in state $s_t$, takes action $a_t$,
receives reward $r_{t+1} = R(s_t, a_t)$, and transitions to $s_{t+1}$.

Q-learning updates its estimate $Q(s_t, a_t)$ using the rule
$$Q_{t+1}(s_t, a_t)
=Q_t(s_t, a_t)
+\alpha \Bigl[
  r_{t+1}
  +
  \gamma \max_{a'} Q_t(s_{t+1}, a')
  -
  Q_t(s_t, a_t)
\Bigr].$$

Here:

- $\alpha \in (0, 1]$ is the **learning rate**.
- $\gamma$ is the discount factor (here $\gamma = 1.0$).
- $\max_{a'} Q_t(s_{t+1}, a')$ is the **bootstrap target**, assuming we act greedily from the next state.

The bracketed term is the **temporal-difference (TD) error)**:
  $$\delta_t = r_{t+1} + \gamma \max_{a'} Q_t(s_{t+1}, a') - Q_t(s_t, a_t).$$

Interpretation:

- If $\delta_t > 0$, we were too pessimistic about $(s_t, a_t)$ and increase $Q(s_t, a_t)$.
- If $\delta_t < 0$, we were too optimistic and decrease it.
- Over many episodes, these adjustments allow $Q(s, a)$ to approach the optimal values $Q_*(s, a)$.

Q-learning works **without** knowing the transition model $P(s' \mid s, a)$;
it only needs sample transitions $(s_t, a_t, r_{t+1}, s_{t+1})$.

---

## Exploration vs Exploitation: $\epsilon$-Greedy

If the agent always picks the action with the highest current value $Q(s, a)$,
it may get stuck repeating the same behavior and never discover better actions.

To encourage exploration, we use an **$\epsilon$-greedy** rule during training:

- With probability $1 - \epsilon$, choose the greedy action:
  $$a = \arg\max_{a'} Q(s, a').$$
- With probability $\epsilon$, choose a random action (explore).

Typically:
- Start with a relatively large $\epsilon$ (e.g., 0.1 or 0.2).
- Optionally **decay** $\epsilon$ over time as our estimates become more reliable.

This mechanism balances:
- **Exploration** — trying actions we are uncertain about,
- **Exploitation** — choosing the best-estimated action according to the current $Q$-values.

---

## Algorithm Sketch for Gridworld

We summarize a basic Q-learning loop for the 4×4 Gridworld:

1. Initialize $Q(s, a)$ to zeros.
2. For each episode:
   - Set $s \leftarrow S$ (start state).
   - Repeat until $s$ is terminal (we reach $G$):
     - Choose $a$ from $s$ using $\epsilon$-greedy on $Q(s, \cdot)$.
     - Take action $a$, observe reward $r$ and next state $s'$.
     - Update
       $$Q(s, a) \leftarrow Q(s, a) + \alpha \Bigl[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Bigr].$$
     - Set $s \leftarrow s'$.
3. After many episodes, derive an approximate optimal policy by taking
   $$\pi(s) = \arg\max_a Q(s, a).$$

In this simple Gridworld, the resulting greedy policy should resemble the Phase 1 value-iteration solution: a roughly shortest‑path route to the goal.

---

## Code Example (python)

Below is a minimal Q-learning implementation for the 4×4 Gridworld.  
You can copy this into a script (for example `pre_school/3_q_learning_Gridworld.py`) and run it with Python.

```python
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
    reward = STEP_REWARD  # -1 on every non-terminal step (including the last move)
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


if __name__ == "__main__":
    Q = q_learning_gridworld()
    policy = greedy_policy_from_Q(Q)
    print("Greedy policy (0=up,1=down,2=left,3=right):")
    print(policy)
```

---

## How to Interpret the Result

- The learned $Q(s, a)$ approximates the optimal action-value function $Q_*(s, a)$ for this Gridworld.
- The derived greedy policy should roughly correspond to a shortest-path strategy from $S$ to $G$.
- Because of randomness in exploration, individual runs may produce slightly different Q-tables and policies, but the qualitative behavior should be similar.

Comparing with Phase 1:

- Phase 1 computed $V_*(s)$ with full knowledge of the dynamics.
- Phase 3 learns behavior directly from sampled experience, without ever building a model $P(s' \mid s, a)$.

---

## Practice Exercises

1. **Change the reward structure.**  
   Try setting the step reward to $-0.1$ or giving a small positive reward at the goal (e.g., $+10$ on entering $G$). How does that affect the learned policy?

2. **Vary exploration.**  
   Experiment with different values of $\epsilon$ (e.g., 0.0, 0.05, 0.2) and see how it changes learning speed and final performance. What happens if $\epsilon = 0$ from the start?

3. **Decay $\epsilon$ over time.**  
   Implement a schedule where $\epsilon$ starts larger (e.g., 0.3) and slowly decays to a smaller value (e.g., 0.01). Does this help?

4. **Visualize the policy.**  
   Instead of printing the policy as numbers 0–3, print arrow symbols (↑, ↓, ←, →) in a 4×4 grid to visualize the path toward the goal.

