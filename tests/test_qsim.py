"""
Tests for the quantum and classical correlated-immunity sampling layer.

Test strategy
-------------
1. Circuit construction: correct qubit/clbit count, rejects invalid params.
2. ShotPool: marginals ≈ p, hub–neighbour correlation ≈ ε, correct shape.
3. ClassicalPool: same checks.
4. Quantum ↔ Classical agreement: chi-squared test of homogeneity on the
   16 bitstring frequencies (change 8).  We test two parameter sets:
   (p=0.3, ε=0.15) and (p=0.4, ε=0.8).
"""

import numpy as np
import pytest
from scipy.stats import chi2_contingency

from qsim.cluster_circuit import build_cluster_circuit
from qsim.shot_pool import ShotPool
from qsim.classical_pool import ClassicalPool
from qsim.validate import validate_pool


# ---------------------------------------------------------------------------
# Two parameter sets for broad coverage (change 8)
# ---------------------------------------------------------------------------
PARAM_SETS = [
    (0.3, 0.15),   # moderate p, weak hub–neighbour correlation
    (0.4, 0.8),    # higher p, strong hub–neighbour correlation
]

POOL_SIZE = 8192           # large enough for chi-squared to be meaningful
ATOL_MARGINAL = 0.03       # absolute tolerance on marginals
ATOL_CORR = 0.05           # slightly looser for hub–neighbour correlation estimates


# ===================================================================== #
# Circuit construction                                                   #
# ===================================================================== #

class TestClusterCircuit:
    """Basic structural checks on the quantum circuit."""

    def test_circuit_has_4_qubits(self):
        qc = build_cluster_circuit(0.3, 0.15)
        assert qc.num_qubits == 4

    def test_circuit_has_measurements(self):
        qc = build_cluster_circuit(0.3, 0.15)
        assert qc.num_clbits == 4

    def test_invalid_params_raise(self):
        # epsilon = 2.0 makes a1 > 1 for any 0 < p < 1
        with pytest.raises(ValueError):
            build_cluster_circuit(0.5, 2.0)


# ===================================================================== #
# ShotPool (quantum)                                                     #
# ===================================================================== #

class TestShotPool:
    """Statistical checks on the quantum shot pool."""

    @pytest.mark.parametrize("p, epsilon", PARAM_SETS)
    def test_marginals(self, p, epsilon):
        pool = ShotPool(p, epsilon, pool_size=POOL_SIZE, seed=42)
        result = validate_pool(pool, p, epsilon, atol=ATOL_MARGINAL)
        assert result["marginals_pass"], (
            f"Marginals {result['marginals']} "
            f"not within {ATOL_MARGINAL} of {p}"
        )

    @pytest.mark.parametrize("p, epsilon", PARAM_SETS)
    def test_hub_neighbour_correlation(self, p, epsilon):
        pool = ShotPool(p, epsilon, pool_size=POOL_SIZE, seed=42)
        result = validate_pool(pool, p, epsilon, atol=ATOL_CORR)
        assert result["correlations_pass"], (
            f"Hub–neighbour correlations {result['hub_leaf_correlations']} "
            f"not within {ATOL_CORR} of {epsilon}"
        )

    def test_pool_shape(self):
        pool = ShotPool(0.3, 0.15, pool_size=1024, seed=42)
        assert pool.patterns.shape == (1024, 4)
        assert pool.patterns.dtype == bool

    def test_draw_returns_copy(self):
        """draw() must return a copy so mutations don't corrupt the pool."""
        pool = ShotPool(0.3, 0.15, pool_size=1024, seed=42)
        rng = np.random.default_rng(99)
        pattern = pool.draw(rng)
        pattern[:] = False                     # mutate the returned array
        assert pool.patterns.any()             # pool is unchanged

    def test_draw_uses_rng(self):
        """Different RNG seeds should (usually) produce different draws."""
        pool = ShotPool(0.3, 0.15, pool_size=4096, seed=42)
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        draws_a = [pool.draw(rng_a).tobytes() for _ in range(20)]
        draws_b = [pool.draw(rng_b).tobytes() for _ in range(20)]
        # Very unlikely to be identical sequences
        assert draws_a != draws_b


# ===================================================================== #
# ClassicalPool (control)                                                #
# ===================================================================== #

class TestClassicalPool:
    """Statistical checks on the classical control pool."""

    @pytest.mark.parametrize("p, epsilon", PARAM_SETS)
    def test_marginals(self, p, epsilon):
        pool = ClassicalPool(p, epsilon, pool_size=POOL_SIZE, seed=42)
        result = validate_pool(pool, p, epsilon, atol=ATOL_MARGINAL)
        assert result["marginals_pass"], (
            f"Marginals {result['marginals']} "
            f"not within {ATOL_MARGINAL} of {p}"
        )

    @pytest.mark.parametrize("p, epsilon", PARAM_SETS)
    def test_hub_neighbour_correlation(self, p, epsilon):
        pool = ClassicalPool(p, epsilon, pool_size=POOL_SIZE, seed=42)
        result = validate_pool(pool, p, epsilon, atol=ATOL_CORR)
        assert result["correlations_pass"], (
            f"Hub–neighbour correlations {result['hub_leaf_correlations']} "
            f"not within {ATOL_CORR} of {epsilon}"
        )

    def test_pool_shape(self):
        pool = ClassicalPool(0.3, 0.15, pool_size=1024, seed=42)
        assert pool.patterns.shape == (1024, 4)
        assert pool.patterns.dtype == bool


# ===================================================================== #
# Quantum ↔ Classical agreement (chi-squared, change 8)                  #
# ===================================================================== #

class TestQuantumClassicalAgreement:
    """
    Chi-squared test of homogeneity on the 16 bitstring frequencies.

    Both pools should produce statistically indistinguishable distributions.
    We use alpha = 0.01 (p-value threshold) — strict enough to catch real
    differences but forgiving of sampling noise.
    """

    @staticmethod
    def _bitstring_counts(patterns: np.ndarray) -> np.ndarray:
        """
        Convert (N, 4) bool array to frequency counts over 16 bitstrings.

        Bitstring index = 8*q0 + 4*q1 + 2*q2 + 1*q3  (big-endian).
        """
        weights = np.array([8, 4, 2, 1])
        indices = patterns.astype(int) @ weights     # shape (N,)
        return np.bincount(indices, minlength=16)    # shape (16,)

    @pytest.mark.parametrize("p, epsilon", PARAM_SETS)
    def test_chi_squared(self, p, epsilon):
        """
        Build a 2×16 contingency table (quantum row, classical row) and
        run scipy.stats.chi2_contingency.  Expect p-value > 0.01.
        """
        q_pool = ShotPool(p, epsilon, pool_size=POOL_SIZE, seed=42)
        c_pool = ClassicalPool(p, epsilon, pool_size=POOL_SIZE, seed=42)

        q_counts = self._bitstring_counts(q_pool.patterns)
        c_counts = self._bitstring_counts(c_pool.patterns)

        # 2 × 16 contingency table
        contingency = np.array([q_counts, c_counts])

        # Add pseudocount of 1 to cells with 0 to avoid chi-sq issues
        contingency = contingency + 1

        chi2, pvalue, dof, _ = chi2_contingency(contingency)

        assert pvalue > 0.01, (
            f"Quantum and classical pools differ significantly: "
            f"chi2={chi2:.2f}, p-value={pvalue:.6f}, dof={dof}"
        )
