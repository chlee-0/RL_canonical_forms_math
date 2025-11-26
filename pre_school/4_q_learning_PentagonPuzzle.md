# Phase 4: Q-Learning in the Pentagon Puzzle

## Objective

- Transfer Q-learning from a small Gridworld (Phase 3) to the **Pentagon Puzzle**.
- Learn how to apply Q-learning in a system where the state space can grow indefinitely, without building the full transition graph in advance.
- Represent a **Q-table on a graph of integer vectors** using Python dictionaries.
- See how $Q(s, a)$ again reflects the expected steps-to-success (up to a constant shift).


---

## Environment Recap: Pentagon Puzzle

We reuse the environment from Phase 2 (see [Phase 2: Value Iteration in the Pentagon Puzzle](./2_value_iteration_PentagonPuzzle.md) for details).

**States.**  
Each state is a 5‑dimensional integer vector
$$s = (s_0, s_1, s_2, s_3, s_4) \in \mathbb{Z}^5,$$
thought of as labels on the vertices of a regular pentagon.

**Actions.**  
An action chooses a vertex $i \in \{0,1,2,3,4\}$.  
Given a state $s$ and an action $i$, we update the triple $(i-1, i, i+1)$ (indices mod 5):
$$s_{i-1}' = s_{i-1} + s_i,\quad s_i'     = -s_i,\quad s_{i+1}' = s_{i+1} + s_i,$$
and all other coordinates stay unchanged. The transition is **deterministic**.

**Termination (Goal).**  
We terminate successfully when all components are non‑negative:
$$\text{success}(s) \iff \forall i,\; s_i \ge 0.$$

In Phase 2 we focused on the initial state
$$s_{\text{init}} = (20, 18, -1, -13, -17).$$

We will **reuse the same initial state** $s_{\text{init}}$ for Q-learning as well, so that we can compare value-iteration and Q-learning behavior on exactly the same puzzle. (Of course, in code you can easily change `s_init` to experiment with other starting points.)


---

## Reward Design and Interpretation

To parallel earlier phases, we keep a simple reward design but add a **terminal bonus**:

- Step reward: $R(s, a) = -1$ for every non‑terminal step.
- Terminal reward: unlike Phases 1–3, which used only a step reward of $-1$,
  here we add a one-time bonus $R_{\text{goal}}$ when the agent first reaches a
  success state (we use $R_{\text{goal}} = 20$ in the code).
- Success states (all components $\ge 0$) are terminal and have no outgoing actions.
- Discount factor: $\gamma = 1.0$.

Under this design, for an episode that reaches a success state in $T$ steps, the return is
$$G_0 \approx -T + R_{\text{goal}}.$$

Thus, among successful trajectories, **shorter paths are still preferred**.
Episodes that never reach success simply accumulate $-T$ with no bonus, so the terminal reward makes successful paths stand out more clearly and helps Q-learning identify them more quickly.


---

## Why Q-Learning Here?

In Phase 2, we explicitly constructed a finite set of states by exploring all states
reachable from the initial vector up to a chosen depth, and then ran value iteration
over that collected graph.

In Phase 4, we avoid building such a graph altogether.  
Instead, Q-learning lets us treat the Pentagon Puzzle as a black-box environment:
we simply interact with it, obtain sampled transitions $(s, a, r, s')$,  
and update $Q(s,a)$ on the fly.

This approach is helpful here because the puzzle’s state space can grow very large,
and Q-learning naturally focuses computation only on the states that actually appear
in the episodes generated from the initial state.


## Q-Learning Algorithm (Conceptually)

The Q-learning procedure here is exactly the same as in Phase 3:
we run episodes starting from $s_{\text{init}}$, choose actions by
$\epsilon$-greedy exploration, and apply the same Q-update rule.

The only differences are:

- the state is now a 5-dimensional integer vector,
- an action applies the triple update rule from Phase 2,
- success states provide a terminal bonus.

Because transitions are deterministic, all randomness comes from the
$\epsilon$-greedy exploration.


---

## Implementation Differences from Phase 3

In Phase 3 (Gridworld), the state space was small and fixed, so the Q-table could be stored in a NumPy array of shape $(4,4,4)$.  
In the Pentagon Puzzle, we cannot pre-allocate such a table:

- the set of reachable states is not known in advance,  
- transitions can produce large integer values,  
- and episodes are capped at 50 steps (see `MAX_STEPS_PER_EPISODE` in the code), so learning only touches a small portion of the overall state space.

Therefore we store $Q$ in a **dictionary**, creating entries on demand:

- Keys are 5-tuples representing states.  
- Values are length-5 NumPy arrays (one per action).  

This keeps the Q-table small and focused on the states actually visited from $s_{\text{init}}$.


---

## Code Example (python)

Use the existing script `pre_school/4_q_learning_PentagonPuzzle.py` in this folder.
Run it with `python 4_q_learning_PentagonPuzzle.py`; it mirrors the value-iteration setup in `2_value_iteration_PentagonPuzzle.py` but learns from episodes instead of prebuilding the graph.
Tweak hyperparameters near the top of that file (e.g., `NUM_EPISODES`, `EPSILON`, `ALPHA`, `GOAL_REWARD`) to experiment.

---

## How to Interpret the Result

- The learned $Q(s, a)$ approximates $Q_*(s, a)$ for states in the region explored from $s_{\text{init}}$.
- The derived approximate value function
  $$V(s) \approx \max_a Q(s, a)$$
  plays the same role as in Phase 2: **larger values** mean states that are closer (in expectation) to success, while **smaller (more negative) values** correspond to states that are farther away.
- The `greedy_rollout` produced after learning is an approximate solution path from $s_{\text{init}}$ to a success state, analogous to following a greedy policy on the value table.

Because we learn from sampled episodes rather than exhaustive graph construction, different runs (with different random seeds) may produce slightly different Q-tables and paths, but they should share the same qualitative behavior.

---

## Exercises

1. **Inspect Q for the initial state.**  
   After training, look at the learned Q-values for the initial state: which vertex actions have the highest scores? Can you interpret this in terms of the puzzle moves?

2. **Compare with value iteration.**  
   In Phase 2, value iteration produced a shortest-path–like action sequence from $s_{\text{init}}$.  
   After running Q-learning, compare its greedy rollout with the Phase 2 path.  
   Do they match? If they diverge, at which step does it happen, and why?

3. **Vary hyperparameters.**  
   In the code, `NUM_EPISODES` is set to `200000`; try reducing it (e.g., 20k, 50k, 100k) or increasing it, and observe how this affects convergence speed, the stability of $Q(s,a)$, and the quality of the greedy rollout path.

4. **Try different initial states.**  
   Change `s_init` to other vectors and see whether Q-learning can still discover success paths within a reasonable number of episodes, and how the learned values look.
