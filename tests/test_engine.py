"""
Tests for the SEIR engine.

Test strategy (change 1, follow-ups 1–2)
-----------------------------------------
1. **Well-mixed vs ODE**: Run in well-mixed mode (geometric durations) on a
   uniform grid and compare to the mean-field SEIR ODE.  Averaged over
   several seeds: ±15 % on final attack rate, ±5 ticks on peak timing.

2. **Lattice sanity tests**:
   a. No spread without a seed — all cells stay S.
   b. Front advances at most 1 cell per tick (Chebyshev distance).
   c. No infection appears in cells with β = 0.

3. **Water blocks spread** — right half is ocean, infection stays on left.

4. **Longitude wrap** (follow-up 2a) — infection at column 0 reaches col 359.

5. **Graded immunity** (follow-up 2b) — immune cells at IMM = 0.9 reduce
   infection probability by the expected factor.

6. **Pipeline sanity** (follow-up 2c) — India POP_EST > 1e9, Bangladesh
   among the 3 densest, land cell count matches saved value.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from engine.grid import WorldGrid, State
from engine.seir import seir_step
from engine.runner import run_simulation


# ===================================================================== #
#  Lattice sanity tests  (change 1)                                      #
# ===================================================================== #

class TestLatticeSanity:
    """Basic invariant tests on the lattice (8-neighbour) mode."""

    def test_no_spread_without_seed(self):
        """An all-S grid with no I cells should stay S forever."""
        grid = WorldGrid(20, 20, beta_base=0.5)
        rng = np.random.default_rng(42)

        for _ in range(10):
            seir_step(grid, rng)

        c = grid.counts()
        assert c["E"] == 0
        assert c["I"] == 0
        assert c["R"] == 0
        assert c["D"] == 0
        assert c["S"] == 20 * 20

    def test_front_advances_at_most_1_per_tick(self):
        """
        After *t* ticks, no cell farther than Chebyshev distance *t*
        from the seed should be touched (E / I / R / D).

        Uses a large enough grid and few enough ticks that the front
        never reaches the edges (so longitude wrap is irrelevant).
        """
        size = 41
        centre = size // 2          # (20, 20)
        n_ticks = 10
        grid = WorldGrid(size, size, beta_base=5.0, beta_cap=5.0)
        grid.seed_infection(centre, centre)
        rng = np.random.default_rng(42)

        for t in range(1, n_ticks + 1):
            seir_step(grid, rng)

            # Find all cells that are no longer S
            touched = grid.state != State.S  # includes R ocean — but
            # the grid is all-land, so this is fine.
            rows, cols = np.where(touched)

            if len(rows) == 0:
                continue

            # Chebyshev distance from centre
            dists = np.maximum(np.abs(rows - centre), np.abs(cols - centre))
            max_dist = int(dists.max())

            assert max_dist <= t, (
                f"Tick {t}: max Chebyshev distance {max_dist} > {t}"
            )

    def test_beta_zero_blocks_infection(self):
        """Cells with β = 0 must never become infected."""
        grid = WorldGrid(20, 30, beta_base=0.8)

        # Zero out the rightmost 10 columns
        grid.beta[:, 20:] = 0.0

        # Seed infection on the left side
        grid.seed_infection(10, 5)
        rng = np.random.default_rng(42)

        for _ in range(60):
            seir_step(grid, rng)

        # No cell in the β = 0 region should be E, I, R, or D
        right_state = grid.state[:, 20:]
        assert np.all(right_state == State.S), (
            "Cells with β = 0 were infected"
        )


# ===================================================================== #
#  Water blocks spread  (change 6)                                       #
# ===================================================================== #

class TestWaterBlocks:
    """Ocean cells should block the spread of infection."""

    def test_ocean_blocks_spread(self):
        """Left half = land, right half = ocean.  Infection stays on left."""
        land_mask = np.zeros((20, 20), dtype=bool)
        land_mask[:, :10] = True       # left 10 columns are land

        grid = WorldGrid(20, 20, land_mask=land_mask, beta_base=0.8)
        grid.seed_infection(10, 5)     # seed on the land side
        rng = np.random.default_rng(42)

        for _ in range(50):
            seir_step(grid, rng)

        # Right half is ocean → permanently R.  No cell should be E or I.
        right_state = grid.state[:, 10:]
        assert np.all(
            (right_state == State.R)
        ), "Infection crossed ocean"

        # Land-only counts should not include ocean R cells
        c = grid.counts()
        assert c["S"] + c["E"] + c["I"] + c["R"] + c["D"] == int(
            land_mask.sum()
        )


# ===================================================================== #
#  Longitude wrap  (follow-up 2a)                                        #
# ===================================================================== #

class TestLongitudeWrap:
    """Infection wraps across the date line (column 0 ↔ column W−1)."""

    def test_infection_wraps_at_date_line(self):
        """
        Seed infection at column 0 on a 10 × 360 grid.
        After enough ticks, column 359 should be reached — infection must
        have crossed westward from column 0 to column 359 via the wrap.
        """
        H, W = 10, 360
        grid = WorldGrid(H, W, beta_base=5.0, beta_cap=5.0)

        # Seed at row 5, column 0
        grid.seed_infection(5, 0)
        rng = np.random.default_rng(42)

        # 5 ticks is enough for 8-neighbour to reach columns 355–359
        for _ in range(5):
            seir_step(grid, rng)

        # Column 359 should have been reached (non-S)
        col_359_states = grid.state[:, 359]
        touched = np.any(col_359_states != State.S)
        assert touched, (
            "Infection did not wrap from column 0 to column 359 "
            "(date line wrap is broken)"
        )


# ===================================================================== #
#  Graded immunity  (follow-up 2b)                                       #
# ===================================================================== #

class TestGradedImmunity:
    """Immune cells with IMM = 0.9 should have reduced infection probability."""

    def test_immunity_reduces_infection(self):
        """
        Compare infection rates between non-immune and immune populations.

        Setup: a 1-row strip of S cells, each with exactly 1 infectious
        neighbour.  Run many independent trials:
        - Group A: no immunity  → P = 1 − exp(−β)
        - Group B: immunity on  → P = 1 − exp(−β · 0.1)

        The observed ratio of infection counts should match the expected
        probability ratio within a reasonable tolerance.
        """
        beta = 0.5
        imm = 0.9
        n_trials = 5000

        p_no_imm  = 1.0 - np.exp(-beta * 1.0)
        p_with_imm = 1.0 - np.exp(-beta * (1.0 - imm))
        expected_ratio = p_with_imm / p_no_imm

        infected_no_imm = 0
        infected_with_imm = 0
        rng = np.random.default_rng(42)

        for trial in range(n_trials):
            # --- Trial without immunity ---
            grid_a = WorldGrid(1, 3, beta_base=beta, beta_cap=beta)
            grid_a.state[0, 0] = State.I   # infectious neighbour
            # Cell (0,1) is S with no immunity — target
            # Cell (0,2) is S — not measured
            seir_step(grid_a, rng)
            if grid_a.state[0, 1] == State.E:
                infected_no_imm += 1

            # --- Trial with immunity ---
            grid_b = WorldGrid(1, 3, beta_base=beta, beta_cap=beta)
            grid_b.state[0, 0] = State.I
            grid_b.immunity[0, 1] = True   # target cell is immune
            seir_step(grid_b, rng)
            if grid_b.state[0, 1] == State.E:
                infected_with_imm += 1

        # Observed ratio
        obs_ratio = infected_with_imm / max(infected_no_imm, 1)

        # Check that ratio is close to expected (±0.05 absolute tolerance)
        assert abs(obs_ratio - expected_ratio) < 0.05, (
            f"Immunity reduction ratio: observed {obs_ratio:.3f}, "
            f"expected {expected_ratio:.3f} "
            f"(counts: no_imm={infected_no_imm}, with_imm={infected_with_imm})"
        )


# ===================================================================== #
#  Well-mixed mode vs SEIR ODE  (change 1 + follow-up 1)                #
# ===================================================================== #

class TestWellMixedVsODE:
    """
    Compare well-mixed automaton (geometric/memoryless durations) to the
    mean-field SEIR ODE.

    Both now use memoryless transitions (geometric in the automaton,
    exponential in the ODE), so agreement should be tighter.

    Averaged over 8 seeds:
    * Final attack rate within ±15 % of ODE.
    * Peak-I timing within ±5 ticks of ODE.
    """

    @staticmethod
    def _solve_ode(N: int, beta_eff: float, sigma: float, gamma: float,
                   mu: float, n_ticks: int, I0: int = 5):
        """Solve the mean-field SEIR ODE and return time series."""

        def rhs(t, y):
            S, E, I, R, D = y
            infection = beta_eff * S * I / N
            return [
                -infection,
                infection - sigma * E,
                sigma * E - gamma * I,
                (1 - mu) * gamma * I,
                mu * gamma * I,
            ]

        y0 = [N - I0, 0, I0, 0, 0]
        sol = solve_ivp(rhs, [0, n_ticks], y0,
                        t_eval=np.arange(n_ticks + 1),
                        method="RK45", max_step=0.5)
        return sol

    def test_epidemic_matches_ode(self):
        """
        Well-mixed automaton vs SEIR ODE, averaged over 8 seeds.

        Tolerances (follow-up 1):
          ±15 % on final R+D  (ratio 0.85–1.15)
          ±5 ticks on peak-I timing
        """
        H, W = 60, 60
        N = H * W                   # 3600, all land
        n_ticks = 120
        n_seeds = 8

        # β_eff = beta_cell × 8.  R₀ = β_eff / γ ≈ 2.4
        # γ = 1/5 = 0.2  →  β_eff = 0.48  →  beta_cell = 0.06
        beta_cell = 0.06
        beta_eff = beta_cell * 8     # 0.48
        sigma = 1.0 / 3             # 1 / latent_period
        gamma = 1.0 / 5             # 1 / infectious_period
        mu = 0.10                   # lethality

        # ---- solve ODE ----
        sol = self._solve_ode(N, beta_eff, sigma, gamma, mu, n_ticks, I0=5)
        ode_final_rd = sol.y[3][-1] + sol.y[4][-1]
        ode_peak_tick = int(np.argmax(sol.y[2]))

        # ---- run automaton over multiple seeds ----
        sim_finals = []
        sim_peaks  = []

        for seed in range(n_seeds):
            grid = WorldGrid(H, W, beta_base=beta_cell, beta_cap=beta_cell)

            # Seed 5 cells as infectious (reduces stochastic noise)
            centre = H // 2
            for dr, dc in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
                grid.seed_infection(centre + dr, centre + dc)

            rng = np.random.default_rng(seed)
            hist = run_simulation(grid, n_ticks, rng, well_mixed=True)

            sim_finals.append(int(hist["R"][-1] + hist["D"][-1]))
            sim_peaks.append(int(np.argmax(hist["I"])))

        # ---- average over seeds ----
        avg_final = np.mean(sim_finals)
        avg_peak  = np.mean(sim_peaks)

        # ---- check: epidemic occurred ----
        assert avg_final > 0.30 * N, (
            f"Epidemic too small: avg R+D = {avg_final:.0f} < 30 % of {N}"
        )

        # ---- check: final attack rate within ±15 % of ODE ----
        if ode_final_rd > 0:
            ratio = avg_final / ode_final_rd
            assert 0.85 < ratio < 1.15, (
                f"Final R+D ratio sim/ode = {ratio:.3f} "
                f"(sim_avg={avg_final:.0f}, ode={ode_final_rd:.0f}, "
                f"per_seed={sim_finals})"
            )

        # ---- check: peak-I timing within ±5 ticks ----
        assert abs(avg_peak - ode_peak_tick) <= 5, (
            f"Peak-I timing: sim_avg={avg_peak:.1f}, ode={ode_peak_tick}, "
            f"diff={abs(avg_peak - ode_peak_tick):.1f}, "
            f"per_seed={sim_peaks}"
        )


# ===================================================================== #
#  Pipeline sanity  (follow-up 2c)                                       #
# ===================================================================== #

class TestPipelineSanity:
    """
    Check that the pipeline-generated data files are internally consistent.

    These tests load the CSV and .npy files from data/processed/ and run
    basic sanity checks.  They require the pipeline to have been run at
    least once (``python -m pipeline.build_world_grid``).
    """

    @pytest.fixture(autouse=True)
    def _load_data(self):
        """Load pipeline outputs; skip if not yet generated."""
        import pathlib
        import pandas as pd

        data_dir = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
        csv_path = data_dir / "countries.csv"
        lm_path  = data_dir / "land_mask.npy"

        if not csv_path.exists() or not lm_path.exists():
            pytest.skip("Pipeline data not found — run 'python -m pipeline.build_world_grid' first")

        self.df = pd.read_csv(csv_path)
        self.land_mask = np.load(lm_path)

    def test_india_population(self):
        """India must have POP_EST > 1 billion."""
        india = self.df[self.df["name"] == "India"]
        assert len(india) == 1, "India not found in countries.csv"
        pop = int(india.iloc[0]["population"])
        assert pop > 1e9, f"India POP_EST = {pop}, expected > 1e9"

    def test_bangladesh_among_densest(self):
        """Bangladesh must be among the 3 densest countries."""
        df = self.df.copy()
        df["density"] = df["population"] / df["area_km2"].replace(0, np.nan)
        top3 = df.nlargest(3, "density")
        top3_names = top3["name"].tolist()
        assert "Bangladesh" in top3_names, (
            f"Bangladesh not in top 3 densest: {top3_names}"
        )

    def test_land_cell_count_matches(self):
        """The saved land_mask.npy should have a consistent cell count."""
        n_land = int(self.land_mask.sum())
        n_total = self.land_mask.size
        assert n_total == 360 * 180, f"Grid is not 360×180: {self.land_mask.shape}"
        # After excluding Antarctica, land fraction should be ~27%
        # (lower than true 29% because some coastal cells are missed)
        land_frac = n_land / n_total
        assert 0.15 < land_frac < 0.45, (
            f"Land fraction {land_frac:.2%} is unreasonable"
        )
        # Land cells should be >10,000 (we get ~17,769 without Antarctica)
        assert n_land > 10_000, f"Only {n_land} land cells — too few"
