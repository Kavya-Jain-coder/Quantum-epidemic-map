"""
Validation utilities for quantum and classical immunity pools.

Checks that a pool's empirical marginals and hub–neighbour correlations match the
target parameters *p* and *epsilon* within a specified tolerance.  Can be
called from tests or from the command line for quick sanity checks.

Usage
-----
>>> from qsim.shot_pool import ShotPool
>>> from qsim.validate import validate_pool
>>> pool = ShotPool(p=0.3, epsilon=0.15, pool_size=8192, seed=42)
>>> result = validate_pool(pool, p=0.3, epsilon=0.15)
>>> print(result)
"""

import numpy as np


def validate_pool(pool, p: float, epsilon: float,
                  atol: float = 0.03) -> dict:
    """
    Check that a pool's patterns match expected marginals and correlations.

    Parameters
    ----------
    pool : ShotPool or ClassicalPool
        Must expose a ``.patterns`` attribute of shape ``(N, 4)``, dtype bool.
    p : float
        Expected marginal immunity probability for every qubit.
    epsilon : float
        Expected hub–neighbour Pearson correlation.
    atol : float
        Absolute tolerance for pass/fail decisions.

    Returns
    -------
    dict
        ``marginals``           – list of 4 measured marginal probabilities
        ``hub_leaf_correlations``– list of 3 measured Pearson correlations
        ``marginals_pass``      – True if all marginals within *atol* of *p*
        ``correlations_pass``   – True if all hub–neighbour corrs within *atol*
                                  of *epsilon*
    """
    patterns = pool.patterns.astype(np.float64)  # (N, 4)

    # ---- marginals: mean of each column ----
    marginals = patterns.mean(axis=0).tolist()

    # ---- hub–neighbour Pearson correlations ----
    hub = patterns[:, 0]
    hub_leaf_corrs: list[float] = []
    for i in range(1, 4):
        leaf = patterns[:, i]
        # np.corrcoef returns a 2×2 matrix; off-diagonal is the correlation
        corr = float(np.corrcoef(hub, leaf)[0, 1])
        hub_leaf_corrs.append(corr)

    # ---- pass / fail ----
    marginals_pass = all(abs(m - p) < atol for m in marginals)
    correlations_pass = all(abs(c - epsilon) < atol for c in hub_leaf_corrs)

    return {
        "marginals": marginals,
        "hub_leaf_correlations": hub_leaf_corrs,
        "marginals_pass": marginals_pass,
        "correlations_pass": correlations_pass,
    }
