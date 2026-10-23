"""
SEIR step function — advances the automaton by one tick.

Transition rules
----------------
**Lattice mode** (default):
  S → E :  P = 1 − exp(−β · I_nbrs · (1 − IMM · immune_bit))    (change 4)
  E → I :  deterministic after ``latent_period`` ticks
  I → R :  deterministic after ``infectious_period`` ticks, prob 1 − lethality
  I → D :  deterministic after ``infectious_period`` ticks, prob lethality

**Well-mixed mode** (``well_mixed=True``):
  S → E :  same formula, but I_nbrs = global_infected_fraction × 8
  E → I :  geometric (memoryless), P = 1 / latent_period per tick
  I → R/D: geometric (memoryless), P = 1 / infectious_period per tick
  This matches the ODE's exponential-rate assumption (follow-up 1).

R, D  :  absorbing states

Neighbour contact
-----------------
* **Lattice mode**: 8-neighbour (Moore) contact on the grid.
  Longitude wraps at the date line (±180°); latitude does NOT wrap at the
  poles (change 6).
* **Well-mixed mode**: every cell sees the global infected fraction × 8.
"""

from __future__ import annotations

import numpy as np

from engine.grid import State


# =====================================================================
# Distance-based transmission kernel  (Phase 2 Step 2)
# =====================================================================

def compute_kernel_weights(H: int = 180, sigma: float = 111.0
                           ) -> dict[tuple[int, int], np.ndarray]:
    """
    Precompute per-row contact weights for distance-based transmission.

    For a receiver in row r with latitude phi:
      - lat = 90 - (r + 0.5) * (180 / H)
      - offset (di, dj) has physical distance:
          d = sqrt((111 * di)**2 + (111 * cos(phi) * dj)**2)
      - offsets with d <= 166 km (1.5 * 111) are included:
          di in {-1, 0, 1}, dj in [-4, 4]
      - weight: w = exp(-d / sigma)  (or 1.0 if sigma is inf)
      - normalized per row so that sum(w) == 8.0 (preserving beta's meaning
        across all latitudes).

    Returns a dict mapping (di, dj) -> float32 array of shape (H,).
    """
    kernel_weights: dict[tuple[int, int], np.ndarray] = {}
    for r in range(H):
        lat_deg = 90.0 - (r + 0.5) * (180.0 / H)
        phi = np.radians(lat_deg)
        cos_phi = float(np.cos(phi))
        row_w: dict[tuple[int, int], float] = {}

        for di in (-1, 0, 1):
            for dj in range(-4, 5):
                if di == 0 and dj == 0:
                    continue
                d = float(np.sqrt((111.0 * di) ** 2 + (111.0 * cos_phi * dj) ** 2))
                if d <= 166.0:
                    w = 1.0 if np.isinf(sigma) else float(np.exp(-d / sigma))
                    row_w[(di, dj)] = w

        tot = sum(row_w.values())
        norm = 8.0 / tot if tot > 0 else 1.0
        for k, v in row_w.items():
            if k not in kernel_weights:
                kernel_weights[k] = np.zeros(H, dtype=np.float32)
            kernel_weights[k][r] = v * norm

    return kernel_weights


def _count_infectious_neighbours(state: np.ndarray,
                                  land_mask: np.ndarray,
                                  kernel_weights: dict[tuple[int, int], np.ndarray] | None = None
                                  ) -> np.ndarray:
    """
    Count distance-weighted infectious (I) neighbours for every cell.

    Uses precomputed per-row weights across offsets (di, dj) with
    distance d <= 166 km. At the equator this reproduces the 8-neighbour
    stencil exactly.

    Longitude wraps at +/-180 deg (left <-> right columns). Latitude does NOT
    wrap at the poles — there is no landmass beyond 90 N / 90 S.

    Returns a float32 array of shape (H, W).
    """
    is_inf = (state == State.I).astype(np.float32)
    H, W = state.shape
    count = np.zeros((H, W), dtype=np.float32)

    if kernel_weights is None:
        kernel_weights = compute_kernel_weights(H)

    for (di, dj), w_row in kernel_weights.items():
        w_col = w_row.reshape(H, 1)
        if not np.any(w_col > 0):
            continue

        # np.roll(arr, (-di, -dj)) -> shifted[i,j] = arr[i+di, j+dj]
        # Longitude wrap (axis=1) connects column 0 to column W-1.
        shifted = np.roll(is_inf, (-di, -dj), axis=(0, 1))

        # Block latitude wrap at the poles:
        # di = -1 -> looking north -> row 0 would see row H-1 -> zero it
        # di = +1 -> looking south -> row H-1 would see row 0  -> zero it
        if di == -1:
            shifted[0, :] = 0.0
        elif di == 1:
            shifted[-1, :] = 0.0

        count += shifted * w_col

    return count


# =====================================================================
# One SEIR tick
# =====================================================================

def seir_step(grid, rng: np.random.Generator, *,
              well_mixed: bool = False) -> None:
    """
    Advance the SEIR automaton by one tick **in place**.

    All transition decisions are computed from the state at the *start*
    of the tick; then all changes are applied simultaneously.

    Parameters
    ----------
    grid : WorldGrid
        Modified in place (``grid.state``, ``grid.timer``).
    rng : np.random.Generator
        Source of randomness for infection and death rolls.
    well_mixed : bool
        If ``True``, use global infected fraction × 8 for contact AND
        geometric (memoryless) E→I / I→R,D durations so the dynamics
        match the mean-field SEIR ODE (follow-up 1).
    """
    state = grid.state
    timer = grid.timer
    land  = grid.land_mask
    H, W  = grid.H, grid.W

    # ---- 0. Advance tick counter ----
    grid.current_tick += 1

    # ---- 1. Snapshot current populations (before any mutation) ----
    is_s = (state == State.S) & land
    is_e = (state == State.E) & land
    is_i = (state == State.I) & land

    # ---- 2. Increment timers for E and I cells ----
    timer[is_e | is_i] += 1

    # ---- 3. E → I ----
    if well_mixed:
        # Geometric (memoryless): each tick, P(transition) = 1/latent_period
        # This matches the ODE's σ = 1/latent_period rate (follow-up 1).
        sigma = 1.0 / grid.latent_period
        e_rolls = rng.random((H, W), dtype=np.float32)
        e_to_i = is_e & (e_rolls < sigma)
    else:
        # Deterministic: transition after exactly latent_period ticks
        e_to_i = is_e & (timer >= grid.latent_period)

    # ---- 4. I → R / D ----
    if well_mixed:
        # Geometric (memoryless): P(exit) = 1/infectious_period per tick.
        gamma = 1.0 / grid.infectious_period
        i_exit_rolls = rng.random((H, W), dtype=np.float32)
        i_exit = is_i & (i_exit_rolls < gamma)
    else:
        # Deterministic: transition after exactly infectious_period ticks
        i_exit = is_i & (timer >= grid.infectious_period)

    die_rolls = rng.random((H, W), dtype=np.float32)
    i_to_d = i_exit & (die_rolls < grid.lethality)
    i_to_r = i_exit & (~i_to_d)

    # ---- 5. S → E  (stochastic, based on infectious neighbours) ----
    #
    # Uses the ORIGINAL state (is_i snapshot / unmodified state array) so
    # that cells transitioning I→R this tick still exert pressure.

    if well_mixed:
        # Every cell sees the global infected fraction × 8
        n_inf  = float(is_i.sum())
        n_land = float(max(land.sum(), 1))
        i_nbrs = np.full((H, W), (n_inf / n_land) * 8.0, dtype=np.float32)
    else:
        kw = getattr(grid, "kernel_weights", None)
        i_nbrs = _count_infectious_neighbours(state, land, kw)

    # Graded immunity (change 4):
    #   protection = 1 − IMM · immune_bit
    #   immune cell → protection = 1 − 0.9 = 0.1  (10 % residual risk)
    #   non-immune  → protection = 1.0
    protection = 1.0 - grid.imm_protection * grid.immunity.astype(np.float32)

    force = grid.beta * i_nbrs * protection
    prob  = 1.0 - np.exp(-force)

    inf_rolls = rng.random((H, W), dtype=np.float32)
    s_to_e = is_s & (inf_rolls < prob)

    # ---- 6. Record arrival time for newly exposed cells ----
    # Arrival-time convention:
    #   seed cell                      → tick 0 (set by seed_infection)
    #   cells infected on the 1st step → tick 1 (grid.current_tick == 1)
    #   cells infected on step k       → tick k
    #   never-infected cells           → -1
    new_exposure = s_to_e & (grid.first_infection_tick < 0)
    grid.first_infection_tick[new_exposure] = grid.current_tick

    # ---- 7. Apply all transitions simultaneously ----
    state[i_to_r] = State.R
    state[i_to_d] = State.D
    state[e_to_i] = State.I
    state[s_to_e] = State.E

    # Reset timers for every cell that changed state
    timer[i_to_r | i_to_d | e_to_i | s_to_e] = 0
