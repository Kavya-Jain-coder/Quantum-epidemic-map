# Quantum-Classical Epidemic Spread Simulator

A hybrid quantum-classical cellular-automaton simulator that models how an
infectious disease spreads across a stylised world map.

> **Disclaimer — not a forecast.**
> Every number produced by this simulator is an *illustrative scenario*, not a
> prediction or public-health recommendation. Parameters are chosen for
> demonstration, not calibrated to any real pathogen. Do not use the outputs
> for decision-making.

## What it does

1. A **360 × 180 grid** (1° cells) covers the world. Each cell represents a
   *region in which an outbreak may become established* — it does **not**
   represent a single person. Population density (country-level average) scales
   the local transmission rate.

2. A **SEIR automaton** (Susceptible → Exposed → Infectious → Recovered / Dead)
   advances in discrete ticks. Infection probability for a susceptible cell is

       P = 1 − exp(−β · I_neighbours · (1 − IMM · immune_bit))

   where `I_neighbours` is the count of infectious 8-neighbours, `β` is a
   density-scaled, capped transmission strength, and `IMM` (default 0.9) is the
   protection level granted by immunity.

3. A **4-qubit quantum circuit** (one per 2 × 2 cluster of cells) encodes
   spatially-correlated immunity via Qiskit / Aer. A hub qubit receives
   `RY(θ(p))` and each of the three leaf qubits receives `RY(θ(a₀))` plus a
   controlled-`RY(θ(a₁) − θ(a₀))` from the hub, so that:
   - each qubit's marginal immunity probability is exactly **p**,
   - hub–neighbour correlation is **ε**,
   - neighbour–neighbour correlation is **ε²**.

   Immunity is assigned *lazily*: a cluster's shot is drawn from a pre-computed
   pool only when infection first reaches it ("lazy collapse").

4. A **classical control sampler** generates the same joint distribution with
   plain NumPy. Statistical tests (chi-squared on 16 bitstring frequencies)
   confirm the quantum and classical pools are indistinguishable.

## Honest limitations

- **No quantum advantage.** 4-qubit circuits are efficiently simulable on a
  classical computer. The quantum layer exists to explore the *formalism*, not
  to claim speedup.
- Population density is currently a **country-level average** (country
  population ÷ country area in km²). All cells within a country share the same
  density value, including overseas territories (e.g. French Guiana receives
  France's average density). Gridded datasets such as WorldPop are a planned
  improvement.
- The 1° grid treats every cell as equal-sized, ignoring the convergence of
  meridians toward the poles. This means the **land fraction by cell count
  (~27 %) is not the true ~29 %** of Earth's surface. Polar cells represent
  much less physical area than equatorial cells.
- Antarctica is excluded by default (no permanent population).
- Any "X million infected" figure shown in the UI is **scaled from sampled
  grid cells** and must be labelled as such.

## Project structure

```
quantum-epidemic-map/
  README.md              ← you are here
  LICENSE                ← MIT
  ATTRIBUTION.md         ← data credits & AI-assistance disclosure
  requirements.txt
  data/
    raw/                 ← downloaded shapefiles (not committed)
    processed/           ← .npy grids, preview PNGs
  qsim/                  ← quantum & classical sampling layer
  engine/                ← SEIR grid engine
  pipeline/              ← scripts that build the world grid
  app/                   ← map viewer (future)
  tests/
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest (for running tests)

# Build the world grid (downloads Natural Earth ~1 MB)
python3 -m pipeline.build_world_grid

# Run tests
python3 -m pytest tests/ -v
```

## Requirements

Tested on **Python 3.14** (should work on 3.10+). See `requirements.txt` for
packages and `requirements-lock.txt` for exact pinned versions. Everything runs
on a laptop with free tools only — no API keys, no cloud, no quantum hardware.
Results are reproducible via fixed seeds.

## License

MIT — see [LICENSE](LICENSE).
