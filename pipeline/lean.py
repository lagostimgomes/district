"""
pipeline/lean.py

Post-hoc partisan lean computation using VEST 2020 precinct-level data.

For any state with a completed district map, this module computes
Biden% per district by spatially joining VEST precinct centroids to
district polygons and summing G20PREDBID / (G20PREDBID + G20PRERTRU).

This is strictly post-hoc — the redistricting algorithm never sees
partisan or demographic data.

Usage:
    from pipeline.lean import district_lean, lean_label, VEST_DIR

    lean = district_lean("MD", compact_gdf)
    # lean = {0: 0.871, 1: 0.727, ...}

    label, colour = lean_label(0.871)
    # ("D+74", "#1259a6")
"""

from pathlib import Path

import geopandas as gpd
import warnings

VEST_DIR = Path("data/vest")

# Biden blue / Trump red with enough opacity for badge backgrounds
D_COLOUR = "#1259a6"
R_COLOUR = "#c0152b"


def vest_path(abbr: str) -> Path | None:
    """Return path to VEST 2020 shapefile for *abbr*, or None if not present."""
    p = VEST_DIR / abbr.lower() / f"{abbr.lower()}_2020.shp"
    return p if p.exists() else None


def district_lean(abbr: str, districts: gpd.GeoDataFrame) -> dict[int, float]:
    """
    Compute true precinct-level Biden% per district for *abbr*.

    Parameters
    ----------
    abbr      : 2-letter state abbreviation.
    districts : GeoDataFrame with a 'district_id' column (0-based int).
                Must already be in the state's projection (cfg.crs).

    Returns
    -------
    dict mapping district_id (int) → Biden fraction [0, 1].
    Empty dict if VEST data is not available for this state.
    """
    shp = vest_path(abbr)
    if shp is None:
        return {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vest = gpd.read_file(shp)

    if "G20PREDBID" not in vest.columns or "G20PRERTRU" not in vest.columns:
        return {}

    vest = vest[["G20PREDBID", "G20PRERTRU", "geometry"]].copy()
    vest["G20PREDBID"] = vest["G20PREDBID"].fillna(0).astype(float)
    vest["G20PRERTRU"] = vest["G20PRERTRU"].fillna(0).astype(float)
    vest = vest.to_crs(districts.crs)

    # Centroid join: avoids double-counting precincts that straddle district lines.
    cents = vest.copy()
    cents["geometry"] = vest.geometry.centroid

    joined = gpd.sjoin(
        cents,
        districts[["district_id", "geometry"]],
        how="left",
        predicate="within",
    )

    lean: dict[int, float] = {}
    for did, grp in joined.groupby("district_id"):
        bid   = grp["G20PREDBID"].sum()
        tru   = grp["G20PRERTRU"].sum()
        total = bid + tru
        lean[int(did)] = float(bid / total) if total > 0 else 0.5

    return lean


def lean_label(biden_pct: float) -> tuple[str, str]:
    """
    Return (label, hex_colour) for a district given its Biden fraction.

    Examples
    --------
    >>> lean_label(0.871)
    ('D+74', '#1259a6')
    >>> lean_label(0.435)
    ('R+13', '#c0152b')
    """
    margin = round(abs(biden_pct - 0.5) * 200)
    if biden_pct >= 0.5:
        return f"D+{margin}", D_COLOUR
    else:
        return f"R+{margin}", R_COLOUR
