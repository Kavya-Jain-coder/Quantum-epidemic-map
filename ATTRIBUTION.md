# Attribution & Data Sources

## AI Assistance

This project was developed with assistance from **Google Antigravity (Gemini)**,
an AI coding assistant. The AI helped with code generation, architecture design,
and documentation. All code was reviewed and approved by the project author.

AI tool: Google Antigravity IDE
Model: Antigravity (Google DeepMind), Claude Sonnet — via Google Antigravity IDE

## Data Sources

### Natural Earth (public domain)

- **Dataset**: Admin 0 – Countries (1:110m Cultural Vectors)
- **URL**: https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/
- **Direct download**: https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip
- **License**: Public domain. Natural Earth data is free for any use.
- **Fields used**:
  - `NAME` — country name
  - `ISO_A3` — ISO 3166-1 alpha-3 code
  - `ISO_A3_EH` — ISO code from "Extended Home"; used as fallback when
    `ISO_A3 == "-99"`. This resolves Norway and France (which have
    `-99` in `ISO_A3` due to overseas territory splits). Entries where
    *both* fields are `-99` (N. Cyprus, Somaliland, Kosovo) receive a
    synthetic code `"X" + 2-digit id`.
  - `POP_EST` — population estimate (source: UN / CIA World Factbook, as
    bundled by Natural Earth; values are approximate)
  - `geometry` — country polygons, used for rasterisation and area calculation
- **Area calculation**: country area in km² is computed from the Natural Earth
  `geometry` field using an equal-area projection (EPSG:6933). This is used
  to derive country-level average population density.
- **Antarctica**: excluded by default (no permanent population; POP_EST ≈ 4490
  represents transient research staff). Configurable via
  `exclude_antarctica=False` in the pipeline.

### Population density (country-level average)

Population density per grid cell is calculated as:

    density = POP_EST / area_km2

where `area_km2` comes from the projected geometry (see above). This is a
**country-level average** — all cells within the same country share the same
density value. This is a coarse approximation; gridded datasets such as
WorldPop (https://www.worldpop.org/) would improve realism. Replacing
country-average density with gridded data is a planned future improvement.

### Additional data sources

TODO — add sources here as they are incorporated (e.g., airport locations,
WorldPop gridded population, climate data).

## Licence compatibility

All data and software dependencies used in this project are compatible with
the MIT licence:

| Dependency    | Licence       |
|---------------|---------------|
| NumPy         | BSD-3-Clause  |
| SciPy         | BSD-3-Clause  |
| Qiskit        | Apache-2.0    |
| qiskit-aer    | Apache-2.0    |
| Matplotlib    | PSF / BSD     |
| GeoPandas     | BSD-3-Clause  |
| Shapely       | BSD-3-Clause  |
| Rasterio      | BSD-3-Clause  |
| Natural Earth | Public domain |
