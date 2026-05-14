"""
render_us_map.py

Combines all completed state redistricting results into a single US map.
Completed states show their best-compact district plan.
Failed / not-yet-run states show as light grey outlines.
At-large states (K=1) show as a single solid district.

Output: data/us_map.png
"""

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT   = Path("data")
COUNTY_SHP  = Path("data/maryland/counties/tl_2020_us_county.shp")
OUTPUT      = DATA_ROOT / "us_map.png"
CRS_MAIN    = "EPSG:5070"   # Albers Equal Area — contiguous US
CRS_AK      = "EPSG:3338"
CRS_HI      = "EPSG:26904"

DISTRICT_COLOURS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC", "#D37295", "#FABFD2",
    "#B6992D", "#499894", "#86BCB6", "#E15759",
]

NO_DATA_COLOUR  = "#D8D8D8"
AT_LARGE_COLOUR = "#B8CCE4"
STATE_EDGE      = "#FFFFFF"
COUNTY_EDGE     = "#AAAAAA"
STATE_LABEL_CLR = "#333333"

# Continental bounding box computed dynamically from state data (see main()).
# These are overwritten at runtime; kept here as fallback defaults.
MAIN_XLIM = (-2.35e6, 2.35e6)
MAIN_YLIM = (-1.55e6, 1.35e6)

# ---------------------------------------------------------------------------
# State metadata
# ---------------------------------------------------------------------------

from state_configs import ALL_STATES, STATES_BY_ABBR

AT_LARGE = {cfg.fips for cfg in ALL_STATES.values() if cfg.k == 1}

# Map abbr → data directory (handle both "md" and "maryland" legacy)
def find_state_dir(abbr: str) -> Path | None:
    d = DATA_ROOT / abbr.lower()
    gpkg = d / "final" / "best_map_compact.gpkg"
    if gpkg.exists():
        return gpkg
    # legacy Maryland path
    if abbr.upper() == "MD":
        leg = DATA_ROOT / "maryland" / "final" / "best_map_compact.gpkg"
        if leg.exists():
            return leg
    return None


# ---------------------------------------------------------------------------
# Load state boundaries (dissolve counties → states)
# ---------------------------------------------------------------------------

def load_state_boundaries() -> gpd.GeoDataFrame:
    counties = gpd.read_file(COUNTY_SHP)
    # Keep only US states (exclude territories: STATEFP > 56 except 60,66,69,72,78)
    us_fips = {str(i).zfill(2) for i in range(1, 57) if i not in {3, 7, 14, 43, 52}}
    col = next(c for c in counties.columns if "STATEFP" in c.upper())
    counties = counties[counties[col].isin(us_fips)].copy()
    counties = counties.rename(columns={col: "STATEFP"})
    states = counties.dissolve(by="STATEFP").reset_index()[["STATEFP", "geometry"]]
    states = states.to_crs(CRS_MAIN)
    # Add abbr
    fips_to_abbr = {cfg.fips: cfg.abbr for cfg in ALL_STATES.values()}
    states["abbr"] = states["STATEFP"].map(fips_to_abbr)
    return states


# ---------------------------------------------------------------------------
# Load one state's districts
# ---------------------------------------------------------------------------

def load_districts(abbr: str, gpkg_path: Path) -> gpd.GeoDataFrame | None:
    try:
        gdf = gpd.read_file(gpkg_path).to_crs(CRS_MAIN)
        gdf["abbr"] = abbr.upper()
        return gdf
    except Exception as e:
        print(f"  [{abbr}] Failed to load {gpkg_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def main():
    print("Loading state boundaries…")
    states = load_state_boundaries()

    # Separate Alaska and Hawaii for insets
    states_main = states[~states["abbr"].isin(["AK", "HI"])].copy()
    states_ak   = states[states["abbr"] == "AK"].copy()
    states_hi   = states[states["abbr"] == "HI"].copy()

    print("Loading completed district maps…")
    all_districts = []
    completed = set()
    failed    = set()

    for cfg in sorted(ALL_STATES.values(), key=lambda c: c.abbr):
        abbr = cfg.abbr
        if cfg.k == 1:
            continue  # at-large — handled via state polygon

        gpkg = find_state_dir(abbr)
        if gpkg is None:
            failed.add(abbr)
            continue

        gdf = load_districts(abbr, gpkg)
        if gdf is None:
            failed.add(abbr)
        else:
            all_districts.append(gdf)
            completed.add(abbr)
            print(f"  [{abbr}] {len(gdf)} districts loaded")

    # At-large states
    at_large_abbrs = {cfg.abbr for cfg in ALL_STATES.values() if cfg.k == 1}

    print(f"\nCompleted: {len(completed)}  |  Failed/missing: {len(failed)}  |  At-large: {len(at_large_abbrs)}")

    # ── Colour assignment ──────────────────────────────────────────────────
    # For each state, rotate the colour palette so adjacent states
    # less likely to share identical colours on boundaries.
    rotation = {abbr: i % 8 for i, abbr in enumerate(sorted(completed))}

    # ── Compute tight bounding box from actual state data ─────────────────
    global MAIN_XLIM, MAIN_YLIM
    minx, miny, maxx, maxy = states_main.total_bounds
    pad_x = (maxx - minx) * 0.02
    pad_y = (maxy - miny) * 0.04
    MAIN_XLIM = (minx - pad_x, maxx + pad_x)
    MAIN_YLIM = (miny - pad_y, maxy + pad_y)

    # ── Build figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(30, 19), facecolor="#EEF2F7")

    # Title block — in figure space, above the map axes.
    fig.text(0.455, 0.975,
             "US Congressional Redistricting — Blind Algorithmic Maps",
             ha="center", va="top", fontsize=20, fontweight="bold", color="#1F3864")
    fig.text(0.455, 0.945,
             "Best-compact plan per state  ·  Weighted ReCom MCMC  ·  Geography only",
             ha="center", va="top", fontsize=11, color="#555555", style="italic")
    fig.text(0.455, 0.922,
             "Districts drawn using population balance and geographic boundaries only — zero partisan or demographic data",
             ha="center", va="top", fontsize=9.5, color="#888888")

    # Main axes (contiguous 48) — sits below the title block.
    ax = fig.add_axes([0.01, 0.10, 0.85, 0.80])
    ax.set_facecolor("#C8DCF0")
    ax.set_xlim(*MAIN_XLIM)
    ax.set_ylim(*MAIN_YLIM)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # ── Draw districts ─────────────────────────────────────────────────────
    for gdf in all_districts:
        abbr = gdf["abbr"].iloc[0]
        if abbr in ("AK", "HI"):
            continue
        rot = rotation.get(abbr, 0)
        gdf_sorted = gdf.sort_values("district_id")
        gdf_sorted["_color"] = [DISTRICT_COLOURS[(int(r["district_id"]) + rot) % len(DISTRICT_COLOURS)]
                                 for _, r in gdf_sorted.iterrows()]
        gdf_sorted.plot(ax=ax, color=gdf_sorted["_color"].tolist(),
                        edgecolor=STATE_EDGE, linewidth=0.3)

    # ── Failed / not-run states ────────────────────────────────────────────
    failed_states = states_main[states_main["abbr"].isin(failed)]
    if not failed_states.empty:
        failed_states.plot(ax=ax, color=NO_DATA_COLOUR, edgecolor=STATE_EDGE, linewidth=0.5)

    # ── At-large states ────────────────────────────────────────────────────
    at_large_main = states_main[states_main["abbr"].isin(at_large_abbrs)]
    if not at_large_main.empty:
        at_large_main.plot(ax=ax, color=AT_LARGE_COLOUR, edgecolor=STATE_EDGE, linewidth=0.5)

    # ── State boundary outlines ────────────────────────────────────────────
    states_main.boundary.plot(ax=ax, edgecolor="#333333", linewidth=0.7, alpha=0.85)

    # ── State abbreviation labels — clipped to axis bounds ─────────────────
    label_skip = {"RI", "CT", "DE", "NJ", "MA", "MD", "DC"}
    xl0, xl1 = MAIN_XLIM
    yl0, yl1 = MAIN_YLIM
    for _, row in states_main.iterrows():
        if row["abbr"] is None or row["abbr"] in label_skip:
            continue
        cx = row.geometry.centroid.x
        cy = row.geometry.centroid.y
        # Skip labels whose centroid falls outside the visible axes area.
        if not (xl0 < cx < xl1 and yl0 < cy < yl1):
            continue
        ax.text(cx, cy, row["abbr"], ha="center", va="center",
                fontsize=6.5, fontweight="bold", color="white",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="#00000055", lw=0))

    # ── Alaska inset (bottom-left) ─────────────────────────────────────────
    ax_ak = fig.add_axes([0.01, 0.01, 0.22, 0.20])
    ax_ak.set_facecolor("#C8DCF0")
    ax_ak.set_aspect("equal")
    ax_ak.set_axis_off()
    ax_ak.patch.set_linewidth(0.8)

    if not states_ak.empty:
        ak_proj = states_ak.to_crs(CRS_AK)
        ak_proj.plot(ax=ax_ak, color=AT_LARGE_COLOUR, edgecolor="#444444", linewidth=0.6)
        cx = ak_proj.geometry.iloc[0].centroid.x
        cy = ak_proj.geometry.iloc[0].centroid.y
        ax_ak.text(cx, cy, "AK\n(at-large)", ha="center", va="center",
                   fontsize=8, fontweight="bold", color="#333333")

    # ── Hawaii inset (right of Alaska) ─────────────────────────────────────
    ax_hi = fig.add_axes([0.23, 0.01, 0.14, 0.14])
    ax_hi.set_facecolor("#C8DCF0")
    ax_hi.set_aspect("equal")
    ax_hi.set_axis_off()

    hi_gpkg = find_state_dir("HI")
    if hi_gpkg and states_hi is not None and not states_hi.empty:
        hi_dist = load_districts("HI", hi_gpkg)
        if hi_dist is not None:
            hi_proj = hi_dist.to_crs(CRS_HI)
            rot = rotation.get("HI", 0)
            hi_proj["_color"] = [DISTRICT_COLOURS[(int(r["district_id"]) + rot) % len(DISTRICT_COLOURS)]
                                  for _, r in hi_proj.iterrows()]
            hi_proj.plot(ax=ax_hi, color=hi_proj["_color"].tolist(),
                         edgecolor=STATE_EDGE, linewidth=0.4)
            states_hi.to_crs(CRS_HI).boundary.plot(ax=ax_hi, edgecolor="#333333", linewidth=0.6)
    elif not states_hi.empty:
        states_hi.to_crs(CRS_HI).plot(ax=ax_hi, color=NO_DATA_COLOUR,
                                       edgecolor="#333333", linewidth=0.5)
        cx = states_hi.to_crs(CRS_HI).geometry.iloc[0].centroid.x
        cy = states_hi.to_crs(CRS_HI).geometry.iloc[0].centroid.y
        ax_hi.text(cx, cy, "HI\n(no data)", ha="center", va="center",
                   fontsize=7, color="#666666")

    # ── Legend (right panel) ───────────────────────────────────────────────
    legend_ax = fig.add_axes([0.86, 0.10, 0.13, 0.80])
    legend_ax.set_axis_off()

    def _legend_row(y, colour, label):
        patch = mpatches.FancyBboxPatch(
            (0.04, y - 0.028), 0.18, 0.056,
            boxstyle="round,pad=0.01",
            facecolor=colour, edgecolor="#666666", linewidth=0.8,
            transform=legend_ax.transAxes,
        )
        legend_ax.add_patch(patch)
        legend_ax.text(0.28, y, label, fontsize=9, va="center",
                       transform=legend_ax.transAxes, color="#222222",
                       linespacing=1.4)

    legend_ax.text(0.04, 0.97, "Legend", fontsize=13, fontweight="bold",
                   transform=legend_ax.transAxes, va="top", color="#1F3864")
    legend_ax.plot([0.04, 0.96], [0.93, 0.93], color="#BBBBBB", linewidth=0.8,
                   transform=legend_ax.transAxes)

    _legend_row(0.87, DISTRICT_COLOURS[0],  "Congressional\ndistricts\n(best-compact)")
    _legend_row(0.74, AT_LARGE_COLOUR,       "At-large state\n(K=1) — no\nredistricting")
    _legend_row(0.61, NO_DATA_COLOUR,        f"No data\n({len(failed)} states)")

    # Status box
    legend_ax.text(
        0.04, 0.52,
        "Status",
        fontsize=10, fontweight="bold", va="top",
        transform=legend_ax.transAxes, color="#333333",
    )
    legend_ax.text(
        0.04, 0.48,
        f"  Completed : {len(completed)}\n"
        f"  At-large  : {len(at_large_abbrs)}\n"
        f"  No data   : {len(failed)}\n"
        f"  Total     : 50",
        fontsize=9, va="top", transform=legend_ax.transAxes,
        color="#444444", family="monospace", linespacing=1.6,
    )

    # No-data state list
    if failed:
        failed_cols = sorted(failed)
        # split into two columns if long
        mid = (len(failed_cols) + 1) // 2
        col1 = "\n".join(failed_cols[:mid])
        col2 = "\n".join(failed_cols[mid:])
        legend_ax.text(0.04, 0.28, "No data:", fontsize=9, fontweight="bold",
                       va="top", transform=legend_ax.transAxes, color="#555555")
        legend_ax.text(0.04, 0.24, col1, fontsize=8.5, va="top",
                       transform=legend_ax.transAxes, color="#888888",
                       family="monospace", linespacing=1.5)
        if col2:
            legend_ax.text(0.54, 0.24, col2, fontsize=8.5, va="top",
                           transform=legend_ax.transAxes, color="#888888",
                           family="monospace", linespacing=1.5)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
