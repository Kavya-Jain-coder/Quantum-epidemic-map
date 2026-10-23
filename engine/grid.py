"""
World grid for the SEIR automaton.

Each cell represents a *region where an outbreak may become established*
(not a single person).  Population density scales the local transmission
rate β.

State encoding (int8)
---------------------
  S = 0   Susceptible
  E = 1   Exposed (latent, not yet infectious)
  I = 2   Infectious
  R = 3   Recovered
  D = 4   Dead

Ocean cells are permanently set to R and excluded from all counts (change 6).
"""

from __future__ import annotations

import numpy as np


# ── state codes ───────────────────────────────────────────────────────
class State:
    """SEIR+D state codes, stored as ``np.int8`` in the grid arrays."""
    S = np.int8(0)
    E = np.int8(1)
    I = np.int8(2)   # noqa: E741  — standard epidemiology notation
    R = np.int8(3)
    D = np.int8(4)


# ── grid ──────────────────────────────────────────────────────────────
class WorldGrid:
    """
    Holds the mutable state arrays for the SEIR cellular automaton.

    Parameters
    ----------
    H, W : int
        Grid height and width (rows × columns).
    density : ndarray (H, W), optional
        Population density in people / km².  If ``None``, every land cell
        gets ``density_ref`` (i.e. uniform β = β_base).
    land_mask : ndarray (H, W) bool, optional
        ``True`` = land, ``False`` = ocean.  Defaults to all-land.
    beta_base : float
        Baseline transmission strength at ``density_ref``.
    density_ref : float
        Reference density (people / km²) at which β = β_base.
    beta_cap : float
        Hard cap on β to prevent extreme values in tiny dense countries.
    latent_period : int
        Ticks spent in E before transitioning to I.
    infectious_period : int
        Ticks spent in I before transitioning to R or D.
    lethality : float
        Probability of death when exiting I (otherwise → R).
    imm_protection : float
        Protection level granted by immunity (change 4).  Infection
        probability uses ``(1 − imm_protection · immune_bit)``.
        Default 0.9 means an immune cell has 10 % residual susceptibility.
    """

    def __init__(
        self,
        H: int,
        W: int,
        *,
        density: np.ndarray | None = None,
        land_mask: np.ndarray | None = None,
        beta_base: float = 0.3,
        density_ref: float = 100.0,
        beta_cap: float = 1.0,
        latent_period: int = 3,
        infectious_period: int = 5,
        lethality: float = 0.10,
        imm_protection: float = 0.9,
    ):
        self.H, self.W = H, W

        # ---- parameters (stored for seir_step to read) ----
        self.latent_period = latent_period
        self.infectious_period = infectious_period
        self.lethality = lethality
        self.imm_protection = imm_protection

        # ---- land mask (change 6) ----
        if land_mask is None:
            self.land_mask = np.ones((H, W), dtype=bool)
        else:
            self.land_mask = np.asarray(land_mask, dtype=bool)

        # ---- state array ----
        # All land cells start Susceptible; ocean cells are permanently R
        # (inert — they never participate in any transition).
        self.state = np.full((H, W), State.S, dtype=np.int8)
        self.state[~self.land_mask] = State.R

        # ---- timer: ticks spent in current E or I state ----
        self.timer = np.zeros((H, W), dtype=np.int16)

        # ---- immunity (bool per cell, initially all non-immune) ----
        self.immunity = np.zeros((H, W), dtype=bool)

        # ---- transmission strength β ----
        # β = min(beta_base · density / density_ref, beta_cap)
        # Ocean cells get β = 0 so infection cannot cross water.
        if density is None:
            density_arr = np.full((H, W), density_ref, dtype=np.float32)
        else:
            density_arr = np.asarray(density, dtype=np.float32)

        self.beta = np.minimum(
            beta_base * density_arr / density_ref, beta_cap
        ).astype(np.float32)
        self.beta[~self.land_mask] = 0.0

    # ----------------------------------------------------------------- #
    # Convenience helpers                                                #
    # ----------------------------------------------------------------- #

    def seed_infection(self, row: int, col: int) -> None:
        """Set a single land cell to Infectious (I) with timer = 0."""
        if not self.land_mask[row, col]:
            raise ValueError(f"Cell ({row}, {col}) is ocean — cannot seed.")
        self.state[row, col] = State.I
        self.timer[row, col] = 0      # seir_step will increment on first tick

    def counts(self) -> dict[str, int]:
        """
        Count cells in each SEIR state, **land cells only** (change 6).

        Ocean cells are excluded so that S + E + I + R + D = total land cells.
        """
        land = self.land_mask
        return {
            "S": int(np.sum((self.state == State.S) & land)),
            "E": int(np.sum((self.state == State.E) & land)),
            "I": int(np.sum((self.state == State.I) & land)),
            "R": int(np.sum((self.state == State.R) & land)),
            "D": int(np.sum((self.state == State.D) & land)),
        }
