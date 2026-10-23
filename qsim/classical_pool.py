"""
Classical control sampler: same joint distribution as the quantum circuit,
generated with plain NumPy.

This exists to confirm that the 4-qubit circuit produces no quantum advantage —
the same correlations can be reproduced classically (as expected for a
classically-simulable 4-qubit system).

Sampling procedure
------------------
For each of *pool_size* patterns:

1. Sample hub ~ Bernoulli(p).
2. For each of the 3 leaves, sample independently:
       leaf ~ Bernoulli(a1)  if hub = 1
       leaf ~ Bernoulli(a0)  if hub = 0

   where  a1 = p + ε(1−p),  a0 = p − εp.

Leaves are conditionally independent given the hub, so:
  - Marginal of every variable = p
  - Hub–neighbour correlation = ε
  - Neighbour–neighbour correlation = ε²

Usage
-----
>>> pool = ClassicalPool(p=0.3, epsilon=0.15, pool_size=4096, seed=42)
>>> rng  = np.random.default_rng(123)
>>> pattern = pool.draw(rng)           # shape (4,), dtype bool
"""

import numpy as np


class ClassicalPool:
    """
    Pre-computed pool of correlated immunity patterns via classical sampling.

    Parameters
    ----------
    p : float
        Marginal immunity probability per cell.
    epsilon : float
        Hub–neighbour correlation.
    pool_size : int
        Number of patterns to generate.
    seed : int
        Seed for NumPy RNG (reproducibility).
    """

    def __init__(self, p: float, epsilon: float,
                 pool_size: int = 4096, seed: int = 42):
        self.p = p
        self.epsilon = epsilon
        self.pool_size = pool_size

        # Conditional probabilities (same as the quantum circuit)
        a1 = p + epsilon * (1.0 - p)   # P(leaf=1 | hub=1)
        a0 = p - epsilon * p           # P(leaf=1 | hub=0)

        rng = np.random.default_rng(seed)

        # --- sample hub ---
        hub = rng.random(pool_size) < p              # shape (pool_size,)

        # --- sample each leaf conditioned on hub ---
        # Choose conditional probability per sample
        leaf_prob = np.where(hub, a1, a0)             # shape (pool_size,)

        leaf1 = rng.random(pool_size) < leaf_prob
        leaf2 = rng.random(pool_size) < leaf_prob
        leaf3 = rng.random(pool_size) < leaf_prob

        # Combine into (pool_size, 4) array: [hub, leaf1, leaf2, leaf3]
        self._patterns = np.column_stack(
            [hub, leaf1, leaf2, leaf3]
        ).astype(bool)

    # ------------------------------------------------------------------ #
    # Public API  (identical interface to ShotPool)                       #
    # ------------------------------------------------------------------ #

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """
        Draw a random immunity pattern from the pool.

        Parameters
        ----------
        rng : np.random.Generator
            Caller-supplied RNG so each ensemble seed produces different
            draw sequences.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(4,)``.  ``True`` = immune.
        """
        idx = rng.integers(0, len(self._patterns))
        return self._patterns[idx].copy()

    @property
    def patterns(self) -> np.ndarray:
        """All patterns in the pool, shape ``(pool_size, 4)``."""
        return self._patterns
