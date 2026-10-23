"""
Quantum shot pool: pre-computes correlated-immunity patterns via Qiskit / Aer.

The circuit is built once and executed on ``AerSimulator(seed_simulator=seed)``
to produce *pool_size* shots.  Each shot is a 4-bit boolean pattern
(True = immune) for one 2×2 cluster.

Patterns are drawn **randomly** from the pool via a caller-supplied
``np.random.Generator``, so different ensemble seeds yield different sequences
of draws.

Usage
-----
>>> pool = ShotPool(p=0.3, epsilon=0.15, pool_size=4096, seed=42)
>>> rng  = np.random.default_rng(123)
>>> pattern = pool.draw(rng)          # shape (4,), dtype bool
"""

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from qsim.cluster_circuit import build_cluster_circuit


class ShotPool:
    """
    Pre-computed pool of correlated immunity patterns from a quantum circuit.

    Parameters
    ----------
    p : float
        Marginal immunity probability per cell.
    epsilon : float
        Hub–neighbour correlation.
    pool_size : int
        Number of shots to pre-compute.
    seed : int
        Seed for ``AerSimulator`` (reproducibility).
    """

    def __init__(self, p: float, epsilon: float,
                 pool_size: int = 4096, seed: int = 42):
        self.p = p
        self.epsilon = epsilon
        self.pool_size = pool_size

        # Build and transpile the circuit for the Aer backend
        circuit = build_cluster_circuit(p, epsilon)
        backend = AerSimulator(seed_simulator=seed)
        compiled = transpile(circuit, backend)

        # Execute all shots in one job
        result = backend.run(compiled, shots=pool_size).result()
        counts = result.get_counts()

        # Convert counts dict → (pool_size, 4) boolean array.
        # Qiskit bitstrings are printed MSB-first for the *classical* register,
        # so we reverse to get [q0, q1, q2, q3] ordering.
        rows: list[list[bool]] = []
        for bitstring, count in counts.items():
            bits = [c == "1" for c in reversed(bitstring)]
            rows.extend([bits] * count)

        self._patterns = np.array(rows, dtype=bool)  # (pool_size, 4)

        # Shuffle so the ordering doesn't depend on dict-iteration order
        rng_shuffle = np.random.default_rng(seed)
        rng_shuffle.shuffle(self._patterns)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
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
