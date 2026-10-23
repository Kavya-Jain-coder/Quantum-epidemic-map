"""
Ensemble runner — drives the SEIR automaton for a given number of ticks
and records per-tick state counts.

Usage
-----
>>> from engine.grid import WorldGrid
>>> from engine.runner import run_simulation
>>> grid = WorldGrid(50, 50)
>>> grid.seed_infection(25, 25)
>>> rng = np.random.default_rng(42)
>>> history = run_simulation(grid, n_ticks=80, rng=rng)
>>> print(history['I'])   # infectious count per tick
"""

from __future__ import annotations

import numpy as np

from engine.seir import seir_step


def run_simulation(
    grid,
    n_ticks: int,
    rng: np.random.Generator,
    *,
    well_mixed: bool = False,
) -> dict[str, np.ndarray]:
    """
    Run the SEIR automaton for *n_ticks* steps and record state counts.

    Parameters
    ----------
    grid : WorldGrid
        The grid to simulate (modified in place).
    n_ticks : int
        Number of simulation ticks.
    rng : np.random.Generator
        Source of randomness.
    well_mixed : bool
        If ``True``, use well-mixed contact (see ``seir_step``).

    Returns
    -------
    dict mapping ``'S'``, ``'E'``, ``'I'``, ``'R'``, ``'D'`` to
    ``np.ndarray`` of shape ``(n_ticks + 1,)`` — counts at tick 0
    (initial state) through tick *n_ticks*.
    Land-only counts (ocean cells excluded, change 6).
    """
    keys = ("S", "E", "I", "R", "D")
    history: dict[str, list[int]] = {k: [] for k in keys}

    # Record initial state (tick 0)
    c = grid.counts()
    for k in keys:
        history[k].append(c[k])

    # Simulate
    for _ in range(n_ticks):
        seir_step(grid, rng, well_mixed=well_mixed)
        c = grid.counts()
        for k in keys:
            history[k].append(c[k])

    # Convert to numpy arrays for easy analysis
    return {k: np.array(v) for k, v in history.items()}
