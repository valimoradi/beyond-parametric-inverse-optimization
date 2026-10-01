"""
inverse/constraints/monotonicity.py
=====================================
Constraint C6: lambda_k >= 0 for every observation, and lambda_k
non-decreasing in z (sorted slope ordering).
"""

import cvxpy as cp
import numpy as np

from config import N_FUNCTIONS


def build_c6_monotonicity(lambda_funcs: list, Z_binned=None) -> list:
    """
    Return the C6 constraints on lambda.

    lambda_k >= 0 (f_k non-decreasing) and lambda_k non-decreasing in z.
    """
    constraints = []
    for k in range(N_FUNCTIONS):
        constraints.append(lambda_funcs[k] >= 0)

        if Z_binned is not None and len(Z_binned[k]) >= 2:
            idx_sorted = np.argsort(Z_binned[k])
            constraints.append(
                lambda_funcs[k][idx_sorted[1:]] >= lambda_funcs[k][idx_sorted[:-1]]
            )

    return constraints
