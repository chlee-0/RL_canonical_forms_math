"""
Standalone implementation of quiver mutation
for skew-symmetric integer matrices.
"""

from __future__ import annotations

import numpy as np


def matrix_mutation(matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Mutate a skew-symmetric integer matrix at vertex k.

    Parameters
    ----------
    matrix : np.ndarray
        n×n skew-symmetric integer matrix B.
    k : int
        Vertex index (0-based) to mutate at.

    Returns
    -------
    np.ndarray
        Mutated matrix B'.
    """
    B = np.array(matrix, dtype=int)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("matrix must be a square 2D array")

    n = B.shape[0]
    if not (0 <= k < n):
        raise IndexError(f"mutation index k={k} out of bounds for size {n}")

    Bp = B.copy()

    for i in range(n):
        for j in range(n):
            if i == k or j == k:
                # Flip signs in row/column k
                Bp[i, j] = -B[i, j]
            elif B[i, k] * B[k, j] > 0:
                # Fomin–Zelevinsky skew-symmetric mutation formula
                sign = 1 if B[i, k] > 0 else -1
                Bp[i, j] = int(B[i, j] + sign * B[i, k] * B[k, j])

    return Bp


__all__ = ["matrix_mutation"]

