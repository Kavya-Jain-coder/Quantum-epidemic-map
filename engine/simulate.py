"""
High-level simulation entry point.

``simulate_from(lat, lon, ...)`` loads the pipeline data, creates a
WorldGrid, seeds infection at the given latitude/longitude, runs the
automaton, and returns a ``SimResult`` containing per-tick state frames
and the arrival-time array.

Usage
-----
>>> from engine.simulate import simulate_from
>>> result = simulate_from(28.6, 77.2, n_ticks=200, seed=42)
>>> print(result.arrival_time[result.seed_row, result.seed_col])  # 0
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.grid import WorldGrid, State
from engine.seir import seir_step


# ── paths to pipeline outputs ────────────────────────────────────────

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"


# ── result container ─────────────────────────────────────────────────

@dataclass
class SimResult:
    """Container for a completed simulation run."""

    frames: np.ndarray
    """uint8 array of shape ``(n_ticks + 1, H, W)`` — full SEIR state at
    each tick (tick 0 = initial state after seeding)."""

    arrival_time: np.ndarray
    """int16 array of shape ``(H, W)``.  Value = tick when the cell was
    first infected (S→E or seed).  ``-1`` = never infected."""

    counts: dict[str, np.ndarray]
    """Per-tick counts: ``'S'``, ``'E'``, ``'I'``, ``'R'``, ``'D'`` →
    arrays of shape ``(n_ticks + 1,)``.  Land cells only."""

    seed_row: int
    """Row where infection was seeded (may differ from input if snapped)."""

    seed_col: int
    """Column where infection was seeded."""

    seed_lat: float
    """Latitude of the seeded cell centre."""

    seed_lon: float
    """Longitude of the seeded cell centre."""

    seed_country: str
    """Country name of the seeded cell (from countries.csv), or 'Unknown'."""

    grid_shape: tuple[int, int]
    """(H, W) of the grid."""

    speed_ns_km_day: float = 0.0
    """Illustrative front speed north-south (km/day)."""

    speed_ew_km_day: float = 0.0
    """Illustrative front speed east-west (km/day)."""

    params: dict = field(default_factory=dict)
    """Echo of input parameters for the UI."""


# ── coordinate helpers ───────────────────────────────────────────────

def latlon_to_rowcol(lat: float, lon: float, H: int = 180, W: int = 360
                     ) -> tuple[int, int]:
    """
    Convert latitude/longitude to grid row/column.

    Grid convention:
      row 0   = 89.5 °N  (top of the 360×180 grid)
      row 179 = 89.5 °S
      col 0   = 179.5 °W
      col 359 = 179.5 °E

    Uses ``math.floor`` (not ``int()``) so that negative fractional
    values round correctly — e.g. lon = −180.5 → col 359, not 0.

    Returns (row, col), clamped to valid ranges.
    """
    row = math.floor(90 - lat)
    col = math.floor(lon + 180) % W

    # Clamp row to 0..H-1  (safety, per change-5)
    row = max(0, min(row, H - 1))
    return row, col


def _snap_to_nearest_land(row: int, col: int,
                          land_mask: np.ndarray) -> tuple[int, int]:
    """
    If (row, col) is ocean, find the nearest land cell (L1 distance).

    Searches in expanding rings up to radius 20.  If nothing is found,
    raises ValueError.
    """
    if land_mask[row, col]:
        return row, col

    H, W = land_mask.shape
    for radius in range(1, 21):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) != radius and abs(dc) != radius:
                    continue                   # only ring, not interior
                r = row + dr
                c = (col + dc) % W             # longitude wrap
                if 0 <= r < H and land_mask[r, c]:
                    return r, c

    raise ValueError(
        f"No land cell found within 20 cells of ({row}, {col})"
    )


def _country_name_at(row: int, col: int,
                     country_id: np.ndarray,
                     countries_df: pd.DataFrame) -> str:
    """Look up the country name for a grid cell."""
    cid = int(country_id[row, col])
    if cid < 0:
        return "Unknown"
    match = countries_df[countries_df["id"] == cid]
    if len(match) == 0:
        return "Unknown"
    return str(match.iloc[0]["name"])


# ── main entry point ─────────────────────────────────────────────────

def simulate_from(
    lat: float,
    lon: float,
    n_ticks: int = 200,
    seed: int = 42,
    *,
    beta_base: float = 0.3,
    beta_cap: float = 1.0,
    lethality: float = 0.10,
    latent_period: int = 3,
    infectious_period: int = 5,
    imm_protection: float = 0.9,
) -> SimResult:
    """
    Run a full SEIR simulation seeded at the given (lat, lon).

    Loads pipeline data from ``data/processed/``, creates a WorldGrid,
    seeds infection, runs ``n_ticks`` steps, and returns a SimResult.

    Parameters
    ----------
    lat, lon : float
        Click location.  If ocean, snaps to nearest land cell.
    n_ticks : int
        Number of simulation ticks (days).
    seed : int
        Random seed for reproducibility.
    beta_base, beta_cap, lethality, latent_period, infectious_period,
    imm_protection : float / int
        Engine parameters forwarded to WorldGrid.

    Returns
    -------
    SimResult
    """
    # ---- load pipeline data ----
    land_mask = np.load(_DATA_DIR / "land_mask.npy")
    density   = np.load(_DATA_DIR / "population_density.npy")
    country_id = np.load(_DATA_DIR / "country_id.npy")
    countries_df = pd.read_csv(_DATA_DIR / "countries.csv")

    H, W = land_mask.shape

    # ---- locate seed cell ----
    row, col = latlon_to_rowcol(lat, lon, H, W)
    row, col = _snap_to_nearest_land(row, col, land_mask)

    seed_lat = 90.0 - row - 0.5     # cell centre latitude
    seed_lon = col - 180.0 + 0.5    # cell centre longitude
    seed_country = _country_name_at(row, col, country_id, countries_df)

    # ---- build grid ----
    grid = WorldGrid(
        H, W,
        density=density,
        land_mask=land_mask,
        beta_base=beta_base,
        beta_cap=beta_cap,
        lethality=lethality,
        latent_period=latent_period,
        infectious_period=infectious_period,
        imm_protection=imm_protection,
    )
    grid.seed_infection(row, col)

    # ---- allocate storage ----
    frames = np.empty((n_ticks + 1, H, W), dtype=np.uint8)
    keys = ("S", "E", "I", "R", "D")
    count_lists: dict[str, list[int]] = {k: [] for k in keys}

    # ---- tick 0: initial state ----
    frames[0] = grid.state.astype(np.uint8)
    c = grid.counts()
    for k in keys:
        count_lists[k].append(c[k])

    # ---- run simulation ----
    rng = np.random.default_rng(seed)
    for t in range(1, n_ticks + 1):
        seir_step(grid, rng)
        frames[t] = grid.state.astype(np.uint8)
        c = grid.counts()
        for k in keys:
            count_lists[k].append(c[k])

    # ---- compute illustrative front speeds (km/day) ----
    # 1 tick = 1 day, 1 cell = ~111 km NS, ~111*cos(lat) km EW
    arr = grid.first_infection_tick
    reached = arr >= 0
    reached_rows, reached_cols = np.where(reached)
    if len(reached_rows) > 1:
        t_max = int(arr[reached].max())
        d_row_max = int(np.abs(reached_rows - row).max())
        col_diffs = np.abs(reached_cols - col)
        col_diffs = np.minimum(col_diffs, W - col_diffs)
        d_col_max = int(col_diffs.max())
        cos_seed = math.cos(math.radians(seed_lat))
        speed_ns = round((d_row_max * 111.0) / max(t_max, 1), 1)
        speed_ew = round((d_col_max * 111.0 * cos_seed) / max(t_max, 1), 1)
    else:
        speed_ns, speed_ew = 0.0, 0.0

    # ---- assemble result ----
    return SimResult(
        frames=frames,
        arrival_time=grid.first_infection_tick.copy(),
        counts={k: np.array(v) for k, v in count_lists.items()},
        seed_row=row,
        seed_col=col,
        seed_lat=seed_lat,
        seed_lon=seed_lon,
        seed_country=seed_country,
        grid_shape=(H, W),
        speed_ns_km_day=speed_ns,
        speed_ew_km_day=speed_ew,
        params={
            "lat": lat, "lon": lon,
            "seed_lat": seed_lat, "seed_lon": seed_lon,
            "n_ticks": n_ticks, "seed": seed,
            "beta_base": beta_base, "beta_cap": beta_cap,
            "lethality": lethality,
            "latent_period": latent_period,
            "infectious_period": infectious_period,
        },
    )
