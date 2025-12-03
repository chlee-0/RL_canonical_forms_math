"""
Example script showing how to use matrix_mutation
on a 4×4 skew-symmetric matrix.
"""

import numpy as np

from matrix_mutation import matrix_mutation


def main() -> None:
    # Example: A4-type 4×4 skew-symmetric matrix
    B = np.array(
        [
            [0, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, -1, 0, 1],
            [0, 0, -1, 0],
        ],
        dtype=int,
    )

    print("Original matrix B:")
    print(B)

    # Single mutation example
    k_single = 1  # mutate at vertex index 1 (0-based)
    B_single = matrix_mutation(B, k_single)

    print(f"\nSingle mutation at vertex {k_single}: μ_{k_single}(B)")
    print(B_single)

    # Sequence of mutations
    mutation_sequence = [1, 2, 1, 3]
    print(f"\nMutation sequence: {mutation_sequence}")

    B_seq = B.copy()
    for step, k in enumerate(mutation_sequence, start=1):
        B_seq = matrix_mutation(B_seq, k)
        print(f"\nStep {step}: mutate at vertex {k}")
        print(B_seq)

    print("\nFinal matrix after sequence:")
    print(B_seq)


if __name__ == "__main__":
    main()
