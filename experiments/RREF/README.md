# RREF Environment Action Space

This experiment implements a reinforcement-learning environment whose states
are matrices and whose actions are **elementary row operations** geared
toward reaching **reduced row echelon form (RREF)**.

## Elementary Row Operations

In linear algebra, the standard elementary row operations are:

1. **Row swap**  
   \( R_i \leftrightarrow R_j \)

2. **Row scaling** by a nonzero scalar \(\lambda\)  
   \( R_i \leftarrow \lambda R_i \)

3. **Row replacement** (add a multiple of one row to another)  
   \( R_i \leftarrow R_i + \lambda R_j \)

These operations generate RREF, but the scalar \(\lambda\) can be any real
number, so the action space is infinite if we expose \(\lambda\) directly
to the agent.

## Making the Action Space Finite

In `q_learning.py` we turn this infinite family into a **finite, discrete**
action space by letting the agent choose only **row/column indices**, and
letting the environment derive the actual scalar values from the current
matrix.

The environment defines three action types:

- `("swap", i, j)`  
  Corresponds to the elementary operation  
  \( R_i \leftrightarrow R_j \).  
  Actions are enumerated for every pair of row indices
  \(0 \le i < j < n_{\text{rows}}\), so there are finitely many.

- `("normalize", i, j)`  
  Corresponds to scaling row \(R_i\) so that entry \((i, j)\) becomes 1:  
  \( R_i \leftarrow \frac{1}{A_{ij}} R_i \) when \(A_{ij} \neq 0\).  
  The agent chooses only the indices \((i, j)\); the scalar
  \(1 / A_{ij}\) is **implicitly determined** by the current matrix.  
  Actions are enumerated for all row–column index pairs
  \(0 \le i < n_{\text{rows}}, 0 \le j < n_{\text{cols}}\).

- `("eliminate", i, p)`  
  Corresponds to a row replacement using row \(p\) as the pivot row:  
  \( R_i \leftarrow R_i - \text{factor} \, R_p \).  
  The environment automatically:
  - finds the **first nonzero column** of row \(p\) (the pivot column \(j\)),
  - sets `factor = A[i, j]`, which drives the pivot-column entry in row \(i\)
    to zero after the update.  
  Thus the agent chooses only the pair of row indices \((i, p)\) with
  \(i \ne p\), and the scalar is again derived from the matrix.

Because `n_rows` and `n_cols` are finite, the total number of these tuples

- `("swap", i, j)`
- `("normalize", i, j)`
- `("eliminate", i, p)`

is finite. This yields a **finite discrete action space** that still
captures the standard elementary row operations needed to compute RREF,
without exposing continuous scalar choices to the agent.

