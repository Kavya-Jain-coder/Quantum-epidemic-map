#!/usr/bin/env python3
"""
Build the world grid from Natural Earth 110m admin-0 countries.

Downloads the Natural Earth shapefile (public domain, ~4 MB), rasterises
country polygons onto a **360 × 180** grid (1° per cell), and saves:

  data/processed/country_id.npy            int16, −1 = ocean
  data/processed/land_mask.npy             bool
  data/processed/population_density.npy    float32, people / km²
  data/processed/countries.csv             id, name, iso_a3, population, area_km2
  data/processed/density_preview.png       matplotlib preview

Density methodology (change 3)
------------------------------
    density = POP_EST / area_km2

*area_km2* is the area of each country's geometry projected to **EPSG:6933**
(World Cylindrical Equal Area).  We use the projected geometry rather than
counting land cells because cell area varies with cos(latitude), so a simple
cell count would over-estimate density near the poles and under-estimate it
near the equator.

This is a **country-level average** — every cell within the same country gets
the same density value.  For realistic sub-national variation, replace with
gridded data such as WorldPop (https://www.worldpop.org/).

Data source (change 2)
----------------------
Natural Earth Admin 0 – Countries (1:110m)
  URL : https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
  Direct : https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip
  Licence: Public domain
  Fields : NAME, ISO_A3, ISO_A3_EH (fallback when ISO_A3 == "-99"), POP_EST, geometry

Antarctica (follow-up 3)
------------------------
Antarctica is excluded by default because it has no permanent population
(POP_EST ≈ 4490 represents transient research staff).  Pass
``exclude_antarctica=False`` to ``main()`` to include it.

Run
---
    python -m pipeline.build_world_grid
"""

from __future__ import annotations

import sys
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from rasterio.transform import from_bounds


# ── paths ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RAW_DIR     = ROOT / "data" / "raw"
OUT_DIR     = ROOT / "data" / "processed"

SHAPEFILE_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/"
    "ne_110m_admin_0_countries.zip"
)
ZIP_PATH     = RAW_DIR / "ne_110m_admin_0_countries.zip"
EXTRACT_DIR  = RAW_DIR / "ne_110m_admin_0_countries"

# ── grid dimensions ───────────────────────────────────────────────────
GRID_W, GRID_H = 360, 180            # 1° cells, full globe


# =====================================================================
# Download & extract
# =====================================================================

def download_shapefile() -> Path:
    """Download and extract Natural Earth shapefile if not already present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Already extracted?
    shp_files = list(EXTRACT_DIR.glob("*.shp")) if EXTRACT_DIR.exists() else []
    if shp_files:
        print(f"  Shapefile already present: {shp_files[0].name}")
        return shp_files[0]

    # Download zip
    if not ZIP_PATH.exists():
        print(f"  Downloading Natural Earth 110m countries …")
        print(f"    URL: {SHAPEFILE_URL}")
        urllib.request.urlretrieve(SHAPEFILE_URL, ZIP_PATH)
        size_mb = ZIP_PATH.stat().st_size / 1e6
        print(f"    Saved: {ZIP_PATH.name} ({size_mb:.1f} MB)")

    # Extract
    print("  Extracting …")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(EXTRACT_DIR)

    shp_files = list(EXTRACT_DIR.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError("No .shp file found in extracted archive.")
    print(f"  Extracted: {shp_files[0].name}")
    return shp_files[0]


# =====================================================================
# Build countries.csv from Natural Earth attributes  (change 2)
# =====================================================================

def build_countries_csv(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Extract country metadata from Natural Earth attributes.

    * **Population** comes from the ``POP_EST`` field (UN / CIA estimates as
      bundled by Natural Earth).
    * **Area** is computed from the geometry projected to EPSG:6933
      (World Cylindrical Equal Area), not from cell counts (change 3).
    * When ``ISO_A3 == "-99"``, we fall back to ``ISO_A3_EH`` (Extended
      Home).  If that is also ``"-99"`` (e.g. N. Cyprus, Somaliland,
      Kosovo), we assign a synthetic code ``"X" + 2-digit id``
      (follow-up 3).
    """
    # Project once to equal-area CRS for accurate areas
    gdf_ea = gdf.to_crs(epsg=6933)

    rows: list[dict] = []
    for idx in range(len(gdf)):
        name = str(gdf.at[idx, "NAME"])

        # --- ISO code: prefer ISO_A3, fall back to ISO_A3_EH ---
        iso = str(gdf.at[idx, "ISO_A3"])
        if iso in ("-99", "", "nan", "-099"):
            iso_eh = str(gdf.at[idx, "ISO_A3_EH"]) if "ISO_A3_EH" in gdf.columns else "-99"
            if iso_eh not in ("-99", "", "nan", "-099"):
                iso = iso_eh
            else:
                iso = f"X{idx:02d}"   # synthetic, last resort

        # --- population ---
        pop = gdf.at[idx, "POP_EST"]
        if pop is None or (isinstance(pop, (int, float)) and pop < 0):
            pop = 0
        pop = int(pop)

        # --- area in km² from equal-area geometry ---
        area_km2 = round(gdf_ea.geometry.iloc[idx].area / 1e6, 1)

        rows.append({
            "id":         idx,
            "name":       name,
            "iso_a3":     iso,
            "population": pop,
            "area_km2":   area_km2,
        })

    return pd.DataFrame(rows)


# =====================================================================
# Rasterise
# =====================================================================

def rasterize_countries(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Burn country polygons onto a 360 × 180 lat/lon grid.

    Returns an ``int16`` array of shape ``(180, 360)``.
    Ocean cells are −1.  ``all_touched=True`` ensures small island nations
    appear on the grid.
    """
    # Affine transform: top-left = (−180, 90), pixel size = 1° × 1°
    transform = from_bounds(-180, -90, 180, 90, GRID_W, GRID_H)

    shapes = [(geom, cid) for cid, geom in enumerate(gdf.geometry)]

    country_id = rasterize(
        shapes,
        out_shape=(GRID_H, GRID_W),
        transform=transform,
        fill=-1,
        dtype=np.int16,
        all_touched=True,
    )
    return country_id


# =====================================================================
# Density grid  (change 3)
# =====================================================================

def build_density_grid(
    country_id: np.ndarray,
    countries_df: pd.DataFrame,
) -> np.ndarray:
    """
    Build a per-cell population density grid (people / km²).

    Every land cell receives its country's average density::

        density = POP_EST / area_km2

    where ``area_km2`` comes from equal-area projected geometry (EPSG:6933).
    Ocean cells are 0.

    **This is a country-level average.**  All cells within one country share
    the same value.  Replace with WorldPop for sub-national resolution.
    """
    density = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    for _, row in countries_df.iterrows():
        cid   = row["id"]
        area  = row["area_km2"]
        pop   = row["population"]

        d = (pop / area) if (area > 0 and pop > 0) else 0.0

        density[country_id == cid] = d

    return density


# =====================================================================
# Preview PNG
# =====================================================================

def save_preview(
    density: np.ndarray,
    land_mask: np.ndarray,
    out_path: Path,
) -> None:
    """Save a log-scale population-density world map as PNG."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Log₁₀ density; NaN for ocean so it appears grey
    vis = np.full_like(density, np.nan)
    vis[land_mask] = np.log10(np.maximum(density[land_mask], 0.1))

    im = ax.imshow(
        vis,
        extent=[-180, 180, -90, 90],
        origin="upper",
        cmap="YlOrRd",
        aspect="auto",
    )
    fig.colorbar(im, ax=ax, shrink=0.6, label="log₁₀(people / km²)")
    ax.set_title(
        "Population Density  (country-level average, Natural Earth POP_EST / EPSG:6933 area)\n"
        "Illustrative only — not a forecast",
        fontsize=11,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Preview saved: {out_path.name}")


# =====================================================================
# Sanity checks
# =====================================================================

def print_sanity_checks(
    countries_df: pd.DataFrame,
    land_mask: np.ndarray,
) -> None:
    """Print quick sanity-check statistics."""
    n_countries = len(countries_df)
    n_land      = int(land_mask.sum())
    n_total     = land_mask.size
    land_frac   = n_land / n_total

    print()
    print("=" * 55)
    print("  SANITY CHECKS")
    print("=" * 55)
    print(f"  Countries mapped        : {n_countries}")
    print(f"  Land cells              : {n_land} / {n_total}  ({land_frac:.1%})")
    print(f"  Expected ≈29% land      : {'OK' if 0.20 < land_frac < 0.45 else 'WARN'}")

    # Top-5 densest countries (by people/km²)
    df = countries_df.copy()
    df["density"] = df["population"] / df["area_km2"].replace(0, np.nan)
    top5 = df.nlargest(5, "density")
    print()
    print("  Top 5 densest countries (people / km²):")
    for _, row in top5.iterrows():
        print(f"    {row['name']:30s}  {row['density']:>10,.1f}")

    # Countries with synthetic ISO codes (both ISO_A3 and ISO_A3_EH were -99)
    synth = countries_df[countries_df["iso_a3"].str.startswith("X")]
    if len(synth) > 0:
        print()
        print(f"  Countries with synthetic ISO code "
              f"(ISO_A3 and ISO_A3_EH both '-99'): {len(synth)}")
        for _, row in synth.iterrows():
            print(f"    {row['iso_a3']:>5s}  {row['name']}")

    print("=" * 55)
    print()


# =====================================================================
# Main
# =====================================================================

def main(*, exclude_antarctica: bool = True) -> None:
    """
    Download → rasterise → compute density → save.

    Parameters
    ----------
    exclude_antarctica : bool
        If ``True`` (default), drop Antarctica before rasterisation.
        Antarctica has no permanent population (POP_EST ≈ 4490 is
        transient research staff) and would distort density stats.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download / locate shapefile
    print("[1/6] Shapefile")
    shp_path = download_shapefile()

    # 2. Load with geopandas (WGS-84 lat/lon)
    print("[2/6] Loading shapefile …")
    gdf = gpd.read_file(shp_path)

    # Optionally exclude Antarctica (follow-up 3)
    if exclude_antarctica:
        ant_mask = gdf["ISO_A3"].isin(["ATA"]) | gdf["NAME"].str.contains(
            "Antarctica", case=False, na=False
        )
        n_dropped = int(ant_mask.sum())
        if n_dropped > 0:
            gdf = gdf[~ant_mask].copy()
            print(f"  Excluded Antarctica ({n_dropped} feature(s))")

    gdf = gdf.reset_index(drop=True)       # 0-based country IDs
    print(f"  {len(gdf)} features loaded")

    # 3. Build countries.csv from attributes (not hand-written)
    print("[3/6] Building countries.csv from Natural Earth attributes …")
    print("  Population field : POP_EST")
    print("  Area method      : geometry → EPSG:6933 (equal-area)")
    countries_df = build_countries_csv(gdf)
    csv_path = OUT_DIR / "countries.csv"
    countries_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path.name}  ({len(countries_df)} rows)")

    # 4. Rasterise country polygons
    print("[4/6] Rasterising onto 360×180 grid …")
    country_id = rasterize_countries(gdf)
    land_mask  = country_id >= 0
    print(f"  Land cells: {land_mask.sum()}")

    # 5. Compute density grid
    print("[5/6] Computing population density (country-level average) …")
    density = build_density_grid(country_id, countries_df)
    print(f"  Max density: {density.max():,.1f} people/km²")

    # 6. Save everything
    print("[6/6] Saving outputs …")
    np.save(OUT_DIR / "country_id.npy",          country_id)
    np.save(OUT_DIR / "land_mask.npy",           land_mask)
    np.save(OUT_DIR / "population_density.npy",  density)
    print("  Saved: country_id.npy, land_mask.npy, population_density.npy")

    save_preview(density, land_mask, OUT_DIR / "density_preview.png")

    # Sanity checks
    print_sanity_checks(countries_df, land_mask)

    print("Pipeline complete ✓")


if __name__ == "__main__":
    main()
