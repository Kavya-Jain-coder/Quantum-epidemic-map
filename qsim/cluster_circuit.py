"""
4-qubit quantum circuit for correlated immunity in a 2×2 cell cluster.

Circuit design
--------------
We want 4 binary random variables (one per cell in a 2×2 cluster) where:
  - Each variable has marginal P(immune) = p
  - Hub–neighbour correlation = epsilon
  - Neighbour–neighbour correlation = epsilon² (via conditional independence
    given the hub)

Qubit layout
  q0 = hub   (e.g. top-left cell of the 2×2 block)
  q1, q2, q3 = leaves (other 3 cells)

Gate sequence
  q0:  RY(θ_p)                     → P(q0 = |1⟩) = p
  qi:  RY(θ(a0))  then  CRY(δ, q0) → P(qi = |1⟩ | hub=0) = a0
                                      P(qi = |1⟩ | hub=1) = a1
where
  a1 = p + ε·(1 − p)      conditional prob if hub is immune
  a0 = p − ε·p             conditional prob if hub is not immune
  θ(a) = 2·arcsin(√a)      standard RY↔probability mapping
  δ    = θ(a1) − θ(a0)     controlled rotation angle

Proof: marginal of each leaf = p
  P(leaf=1) = p·a1 + (1−p)·a0
            = p·[p + ε(1−p)] + (1−p)·[p − εp]
            = p² + εp(1−p) + p(1−p) − εp(1−p)
            = p

Proof: hub–neighbour Pearson correlation = ε
  Cov(hub, leaf) = p·a1 − p² = εp(1−p)
  Var(hub) = Var(leaf) = p(1−p)
  Corr = ε
"""

import numpy as np
from qiskit import QuantumCircuit


def _theta(a: float) -> float:
    """Convert probability *a* to RY rotation angle: θ = 2·arcsin(√a)."""
    return 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_cluster_circuit(p: float, epsilon: float) -> QuantumCircuit:
    """
    Build a 4-qubit circuit encoding correlated immunity for a 2×2 cluster.

    Parameters
    ----------
    p : float
        Marginal immunity probability for each cell (0 < p < 1).
    epsilon : float
        Hub–neighbour correlation (0 ≤ epsilon ≤ 1).
        Neighbour–neighbour correlation will be epsilon².

    Returns
    -------
    QuantumCircuit
        A measured 4-qubit circuit ready for execution on a backend.

    Raises
    ------
    ValueError
        If parameters produce conditional probabilities outside [0, 1].
    """
    # ---------- conditional probabilities ----------
    a1 = p + epsilon * (1.0 - p)   # P(leaf=1 | hub=1)
    a0 = p - epsilon * p           # P(leaf=1 | hub=0)

    if not (0.0 <= a0 <= 1.0):
        raise ValueError(
            f"a0 = {a0:.4f} is outside [0, 1]. "
            f"Check p={p}, epsilon={epsilon}."
        )
    if not (0.0 <= a1 <= 1.0):
        raise ValueError(
            f"a1 = {a1:.4f} is outside [0, 1]. "
            f"Check p={p}, epsilon={epsilon}."
        )

    # ---------- rotation angles ----------
    theta_p  = _theta(p)
    theta_a0 = _theta(a0)
    theta_a1 = _theta(a1)
    delta    = theta_a1 - theta_a0  # CRY angle

    # ---------- build circuit ----------
    qc = QuantumCircuit(4)

    # Hub qubit: marginal = p
    qc.ry(theta_p, 0)

    # Each leaf: RY(θ(a0)) then CRY(δ) controlled by hub
    for leaf in (1, 2, 3):
        qc.ry(theta_a0, leaf)
        qc.cry(delta, 0, leaf)

    # Measure all qubits into a classical register
    qc.measure_all()

    return qc
