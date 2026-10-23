"""
Tests for the SEIR engine.

Test strategy (change 1, follow-ups 1–2, phase 2 step 1)
---------------------------------------------------------
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

7. **Arrival time** (phase 2 step 1):
   a. Seed cell has arrival tick 0.
   b. Arrival ticks increase with Chebyshev distance from the seed.
   c. Unreached cells remain −1.

8. **simulate_from** (phase 2 step 1):
   a. Returns correct shapes.
   b. Ocean click snaps to nearest land cell.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from engine.grid import WorldGrid, State
from engine.seir import seir_step
from engine.runner import run_simulation
from engine.simulate import latlon_to_rowcol, _snap_to_nearest_land


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
        After *t* ticks, no cell farther than (max_reach * t) from
        the seed should be touched (E / I / R / D).
        Max reach is 1 row/tick north-south, and max_col_reach columns/tick
        east-west (reach-limited version, phase 2 step 2).
        """
        size = 41
        centre = size // 2          # (20, 20)
        n_ticks = 10
        grid = WorldGrid(size, size, beta_base=5.0, beta_cap=5.0)
        grid.seed_infection(centre, centre)
        rng = np.random.default_rng(42)

        max_row_reach = 1
        max_col_reach = max(
            abs(dj) for (di, dj), w in grid.kernel_weights.items() if np.any(w > 0)
        )

        for t in range(1, n_ticks + 1):
            seir_step(grid, rng)

            # Find all cells that are no longer S
            touched = grid.state != State.S
            rows, cols = np.where(touched)

            if len(rows) == 0:
                continue

            row_dists = np.abs(rows - centre)
            col_dists = np.abs(cols - centre)

            assert int(row_dists.max()) <= t * max_row_reach, (
                f"Tick {t}: max row distance {row_dists.max()} > {t * max_row_reach}"
            )
            assert int(col_dists.max()) <= t * max_col_reach, (
                f"Tick {t}: max col distance {col_dists.max()} > {t * max_col_reach}"
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


# ===================================================================== #
#  Arrival-time tracking  (phase 2 step 1)                               #
# ===================================================================== #

class TestArrivalTime:
    """Verify the first_infection_tick array on a uniform all-land grid."""

    def test_seed_cell_arrival_is_zero(self):
        """The seeded cell must have arrival tick 0."""
        grid = WorldGrid(20, 20, beta_base=0.5)
        grid.seed_infection(10, 10)
        assert grid.first_infection_tick[10, 10] == 0

    def test_arrival_increases_with_distance(self):
        """
        On a uniform all-land grid with high β, cells at Chebyshev
        distance d from the seed should have arrival tick ≥ d.
        Cells reached at tick t should have Chebyshev distance ≤ t.
        """
        size = 41
        centre = size // 2     # (20, 20)
        n_ticks = 15

        grid = WorldGrid(size, size, beta_base=5.0, beta_cap=5.0)
        grid.seed_infection(centre, centre)
        rng = np.random.default_rng(42)

        for _ in range(n_ticks):
            seir_step(grid, rng)

        arr = grid.first_infection_tick

        # 1. Seed cell = tick 0
        assert arr[centre, centre] == 0

        # 2. For every reached cell, arrival_tick ≥ Chebyshev distance
        reached = arr >= 0
        rows, cols = np.where(reached)
        for r, c in zip(rows, cols):
            dist = max(abs(r - centre), abs(c - centre))
            tick = arr[r, c]
            assert tick >= dist, (
                f"Cell ({r},{c}): Chebyshev dist={dist} but arrival={tick}"
            )

        # 3. For every reached cell, Chebyshev distance ≤ arrival_tick
        #    (front can advance at most 1 cell per tick)
        for r, c in zip(rows, cols):
            dist = max(abs(r - centre), abs(c - centre))
            tick = arr[r, c]
            assert dist <= tick or (r == centre and c == centre), (
                f"Cell ({r},{c}): dist={dist} > tick={tick} — too fast"
            )

    def test_arrival_time_1d_strip(self):
        """
        Verify the arrival-time convention on a 1-D strip:
        - seed cell = tick 0
        - cells infected on the 1st step = tick 1
        - unreached cells = -1
        """
        # 1-row grid with 15 columns, high beta so transmission is deterministic
        grid = WorldGrid(1, 15, beta_base=10.0, beta_cap=10.0)
        grid.seed_infection(0, 7)

        # Before any step: seed cell is 0, all others -1
        assert grid.first_infection_tick[0, 7] == 0
        assert np.all(grid.first_infection_tick[0, :7] == -1)
        assert np.all(grid.first_infection_tick[0, 8:] == -1)

        # Advance by 1 tick
        rng = np.random.default_rng(42)
        seir_step(grid, rng)

        # Seed remains 0
        assert grid.first_infection_tick[0, 7] == 0
        # Immediate neighbours infected on step 1 must have arrival tick 1
        assert grid.first_infection_tick[0, 6] == 1
        assert grid.first_infection_tick[0, 8] == 1
        # All other cells must remain -1
        assert np.all(grid.first_infection_tick[0, :6] == -1)
        assert np.all(grid.first_infection_tick[0, 9:] == -1)

    def test_unreached_cells_are_minus_one(self):
        """
        Run a normal epidemic on a grid with an ocean strip.
        Every cell with arrival == -1 must be Susceptible (or ocean) at the end,
        and every non-susceptible land cell must have arrival >= 0.
        """
        # 20x40 grid, columns 18..22 are ocean (barrier width 5)
        grid = WorldGrid(20, 40, beta_base=0.8)
        grid.land_mask[:, 18:23] = False
        grid.beta[~grid.land_mask] = 0.0

        # Seed on the left continent
        grid.seed_infection(10, 5)
        rng = np.random.default_rng(42)

        for _ in range(50):
            seir_step(grid, rng)

        arr = grid.first_infection_tick
        state = grid.state
        land = grid.land_mask

        # Non-triviality: some land cells infected, some land cells across ocean unreached
        assert (arr >= 0).sum() > 1, "Epidemic did not spread at all"
        assert (land & (arr == -1)).sum() > 0, "No unreached land cells left"

        # 1. Every cell with arrival == -1 must be Susceptible (or ocean)
        unreached = (arr == -1)
        assert np.all((state[unreached] == State.S) | (~land[unreached])), (
            "Found cells with arrival == -1 that are neither Susceptible nor ocean"
        )

        # 2. Every non-susceptible land cell must have arrival >= 0
        infected_or_recovered = land & (state != State.S)
        assert np.all(arr[infected_or_recovered] >= 0), (
            "Found non-susceptible land cells with arrival < 0"
        )

    def test_arrival_time_monotonic_rings(self):
        """
        On a uniform high-β grid, the mean arrival time at Chebyshev
        distance d should be strictly increasing with d.
        """
        size = 31
        centre = size // 2
        n_ticks = 12

        grid = WorldGrid(size, size, beta_base=5.0, beta_cap=5.0)
        grid.seed_infection(centre, centre)
        rng = np.random.default_rng(42)

        for _ in range(n_ticks):
            seir_step(grid, rng)

        arr = grid.first_infection_tick
        prev_mean = -1.0

        for d in range(0, n_ticks):
            # Collect arrival times for cells at exactly Chebyshev distance d
            mask = np.zeros((size, size), dtype=bool)
            for r in range(size):
                for c in range(size):
                    if max(abs(r - centre), abs(c - centre)) == d:
                        mask[r, c] = True

            reached = mask & (arr >= 0)
            if not reached.any():
                break

            mean_t = arr[reached].mean()
            assert mean_t > prev_mean, (
                f"Ring d={d}: mean arrival {mean_t:.1f} "
                f"not > prev {prev_mean:.1f}"
            )
            prev_mean = mean_t


# ===================================================================== #
#  simulate_from + coordinate helpers  (phase 2 step 1)                  #
# ===================================================================== #

class TestSimulateFrom:
    """
    Tests for the simulate_from() entry point.

    These require pipeline data; skipped if not available.
    """

    @pytest.fixture(autouse=True)
    def _check_data(self):
        import pathlib
        data_dir = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
        if not (data_dir / "land_mask.npy").exists():
            pytest.skip("Pipeline data not found")

    def test_simulate_from_returns_correct_shape(self):
        """Check that frames and arrival_time have the expected shapes."""
        from engine.simulate import simulate_from

        n_ticks = 10
        result = simulate_from(28.6, 77.2, n_ticks=n_ticks, seed=42)

        H, W = result.grid_shape
        assert result.frames.shape == (n_ticks + 1, H, W)
        assert result.arrival_time.shape == (H, W)
        assert result.frames.dtype == np.uint8
        assert result.arrival_time.dtype == np.int16

        # Counts should have n_ticks + 1 entries
        for key in ("S", "E", "I", "R", "D"):
            assert len(result.counts[key]) == n_ticks + 1

    def test_seed_cell_in_result(self):
        """The seed cell in the result should have arrival_time 0."""
        from engine.simulate import simulate_from

        result = simulate_from(28.6, 77.2, n_ticks=5, seed=42)
        assert result.arrival_time[result.seed_row, result.seed_col] == 0

    def test_ocean_snaps_to_land(self):
        """Clicking the middle of the ocean should snap to a land cell."""
        # Middle of the Pacific: ~0°N, 180°W → likely ocean
        land_mask = np.load(
            pathlib.Path(__file__).resolve().parent.parent
            / "data" / "processed" / "land_mask.npy"
        )
        row, col = latlon_to_rowcol(0.0, -160.0)
        snapped_row, snapped_col = _snap_to_nearest_land(row, col, land_mask)
        assert land_mask[snapped_row, snapped_col], (
            f"Snapped cell ({snapped_row}, {snapped_col}) is still ocean"
        )

    def test_latlon_to_rowcol_equator(self):
        """Equator at Greenwich should map to row 90, col 180."""
        row, col = latlon_to_rowcol(0.0, 0.0)
        assert row == 90
        assert col == 180

    def test_latlon_to_rowcol_clamped(self):
        """Extreme latitudes should be clamped to valid row range."""
        row, _ = latlon_to_rowcol(95.0, 0.0)    # beyond north pole
        assert 0 <= row <= 179
        row, _ = latlon_to_rowcol(-95.0, 0.0)   # beyond south pole
        assert 0 <= row <= 179

    def test_latlon_to_rowcol_boundary_cases(self):
        """
        Boundary tests using math.floor:
        - lon = -180.5 -> col 359
        - lon = 180    -> col 0
        - lat = 90     -> row 0
        - lat = -90    -> row 179
        """
        _, col1 = latlon_to_rowcol(0.0, -180.5)
        assert col1 == 359, f"lon=-180.5 gave col={col1}, expected 359"

        _, col2 = latlon_to_rowcol(0.0, 180.0)
        assert col2 == 0, f"lon=180.0 gave col={col2}, expected 0"

        row1, _ = latlon_to_rowcol(90.0, 0.0)
        assert row1 == 0, f"lat=90.0 gave row={row1}, expected 0"

        row2, _ = latlon_to_rowcol(-90.0, 0.0)
        assert row2 == 179, f"lat=-90.0 gave row={row2}, expected 179"


# ===================================================================== #
#  Distance-based transmission kernel  (phase 2 step 2)                  #
# ===================================================================== #

class TestDistanceKernel:
    """Tests for the distance-based transmission kernel."""

    def test_equator_active_offsets_reproduce_moore_stencil(self):
        """
        At the equator (d <= 166 km), the kernel offsets with non-zero
        weight must match the 8-neighbour Moore stencil exactly.
        """
        from engine.seir import compute_kernel_weights
        kw = compute_kernel_weights(180, sigma=111.0)
        for eq_row in (89, 90):
            active = sorted([(di, dj) for (di, dj), w in kw.items() if w[eq_row] > 0])
            expected = sorted([
                (di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if not (di == 0 and dj == 0)
            ])
            assert active == expected, (
                f"Equator row {eq_row} active offsets {active} != Moore {expected}"
            )

    def test_equator_identical_epidemic_for_same_seed(self):
        """
        At the equator, verify that the new kernel reproduces the old
        stencil's epidemic tick-for-tick for the same seed:
        1. With uniform weights (sigma=inf), weights sum to 8 and are
           each exactly 1.0 at the equator. The simulation is 100% identical.
        2. With physical sigma=111.0 and deterministic transmission (beta=5.0),
           all cell states and arrival times are also 100% identical.
        """
        H, W = 180, 360

        def _step_old_8nbr(grid, rng):
            is_s = (grid.state == State.S) & grid.land_mask
            is_e = (grid.state == State.E) & grid.land_mask
            is_i = (grid.state == State.I) & grid.land_mask
            grid.current_tick += 1
            grid.timer[is_e | is_i] += 1
            e_to_i = is_e & (grid.timer >= grid.latent_period)
            i_exit = is_i & (grid.timer >= grid.infectious_period)
            die_rolls = rng.random((H, W), dtype=np.float32)
            i_to_d = i_exit & (die_rolls < grid.lethality)
            i_to_r = i_exit & (~i_to_d)

            is_inf = is_i.astype(np.float32)
            count = np.zeros((H, W), dtype=np.float32)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    sh = np.roll(is_inf, (-di, -dj), axis=(0, 1))
                    if di == -1:
                        sh[0, :] = 0.0
                    elif di == 1:
                        sh[-1, :] = 0.0
                    count += sh
            force = grid.beta * count
            prob = 1.0 - np.exp(-force)
            inf_rolls = rng.random((H, W), dtype=np.float32)
            s_to_e = is_s & (inf_rolls < prob)
            new_exp = s_to_e & (grid.first_infection_tick < 0)
            grid.first_infection_tick[new_exp] = grid.current_tick
            grid.state[i_to_r] = State.R
            grid.state[i_to_d] = State.D
            grid.state[e_to_i] = State.I
            grid.state[s_to_e] = State.E
            grid.timer[s_to_e] = 0
            grid.timer[e_to_i] = 0

        # Case 1: sigma = inf (exact 1.0 weights) at beta = 0.8
        g_old = WorldGrid(H, W, beta_base=0.8)
        g_old.seed_infection(90, 180)
        rng_old = np.random.default_rng(42)

        g_new = WorldGrid(H, W, beta_base=0.8, kernel_sigma=float("inf"))
        g_new.seed_infection(90, 180)
        rng_new = np.random.default_rng(42)

        for _ in range(12):
            _step_old_8nbr(g_old, rng_old)
            seir_step(g_new, rng_new)
            assert np.array_equal(g_old.state, g_new.state), "State diverged under sigma=inf"
            assert np.array_equal(g_old.first_infection_tick, g_new.first_infection_tick)

        # Case 2: default sigma = 111.0 at high beta = 5.0
        g_old2 = WorldGrid(H, W, beta_base=5.0, beta_cap=5.0)
        g_old2.seed_infection(90, 180)
        rng_old2 = np.random.default_rng(123)

        g_new2 = WorldGrid(H, W, beta_base=5.0, beta_cap=5.0, kernel_sigma=111.0)
        g_new2.seed_infection(90, 180)
        rng_new2 = np.random.default_rng(123)

        for _ in range(10):
            _step_old_8nbr(g_old2, rng_old2)
            seir_step(g_new2, rng_new2)
            assert np.array_equal(g_old2.state, g_new2.state), "State diverged under beta=5"
            assert np.array_equal(g_old2.first_infection_tick, g_new2.first_infection_tick)

    def test_ew_ns_extent_ratio_within_25_percent(self):
        """
        Averaged over 8 seeds on a uniform all-land grid, EW extent in km
        vs NS extent in km must be within 25% of 1.0 (ratio between 0.75 and 1.25)
        when seeded at 0, 45, 60, and 70 degrees N.
        """
        import math
        H, W = 180, 360
        n_seeds = 8
        n_ticks = 15

        for lat in (0, 45, 60, 70):
            seed_row = int(math.floor(90.0 - lat))
            seed_col = 180
            phi_c = math.radians(lat)
            ratios = []

            for s in range(n_seeds):
                grid = WorldGrid(H, W, beta_base=0.5, kernel_sigma=111.0)
                grid.seed_infection(seed_row, seed_col)
                rng = np.random.default_rng(200 + s)

                for _ in range(n_ticks):
                    seir_step(grid, rng)

                reached = grid.first_infection_tick >= 0
                rows, cols = np.where(reached)
                assert len(rows) > 1, f"Seed {s} at lat {lat} did not spread"

                ns_km = (rows.max() - rows.min()) * 111.0
                ew_km = (cols.max() - cols.min()) * 111.0 * math.cos(phi_c)
                assert ns_km > 0, "Zero NS extent"
                ratios.append(ew_km / ns_km)

            mean_ratio = float(np.mean(ratios))
            assert 0.75 <= mean_ratio <= 1.25, (
                f"Lat {lat}N: mean EW/NS ratio {mean_ratio:.3f} outside 25% tolerance [0.75, 1.25]"
            )

    def test_well_mixed_mode_unaffected_by_kernel(self):
        """
        Well-mixed mode uses global infected fraction * 8 and bypasses
        spatial kernel weights completely.
        """
        grid = WorldGrid(30, 30, beta_base=1.0)
        grid.seed_infection(15, 15)
        rng = np.random.default_rng(42)
        for _ in range(5):
            seir_step(grid, rng, well_mixed=True)
        counts = grid.counts()
        assert counts["I"] + counts["E"] + counts["R"] + counts["D"] > 1


import pathlib  # needed at module level for test_ocean_snaps_to_land
