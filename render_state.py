"""
render_state.py — Render district maps for any completed state.

Usage:
    python render_state.py MD
    python render_state.py TN
    python render_state.py MD TN

Produces (in data/{abbr}/final/):
    map_compact.png
    map_fewest_splits.png
    map_comparison.png   (side-by-side compact vs fewest-splits, with lean)
    map_comparison_enacted.png   (MD only — vs enacted CD118)
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from state_configs import STATES_BY_ABBR
from pipeline.lean import district_lean, lean_label

COUNTY_SHP = Path("data/maryland/counties/tl_2020_us_county.shp")
CURRENT_DIST_MD = Path("data/maryland/current_districts/tl_2022_24_cd118.shp")

DISTRICT_COLOURS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC", "#D37295", "#FABFD2",
    "#B6992D", "#499894", "#86BCB6", "#E15759",
]


def load_counties(fips: str, crs: str) -> gpd.GeoDataFrame | None:
    if not COUNTY_SHP.exists():
        return None
    counties = gpd.read_file(COUNTY_SHP)
    col = next((c for c in counties.columns if "STATEFP" in c.upper()), None)
    if col:
        counties = counties[counties[col] == fips]
    return counties.to_crs(crs) if not counties.empty else None


def load_stats(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_map(ax, districts, counties, title, stats, lean_data: dict | None = None,
             n_colors: int | None = None):
    k = n_colors or len(districts)
    colors = [DISTRICT_COLOURS[int(r["district_id"]) % len(DISTRICT_COLOURS)]
              for _, r in districts.sort_values("district_id").iterrows()]

    districts.sort_values("district_id").plot(
        ax=ax, color=colors, linewidth=0.8, edgecolor="white",
    )
    if counties is not None:
        counties.boundary.plot(ax=ax, linewidth=0.35, color="#444444", alpha=0.45)

    for _, row in districts.iterrows():
        did = int(row["district_id"])
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        va = "bottom" if (lean_data and did in lean_data) else "center"
        ax.annotate(str(did + 1), xy=(cx, cy), ha="center", va=va,
                    fontsize=8, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, lw=0))
        if lean_data and did in lean_data:
            lbl, lcolor = lean_label(lean_data[did])
            ax.annotate(lbl, xy=(cx, cy), xytext=(0, -13), textcoords="offset points",
                        ha="center", va="top", fontsize=7, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.15", fc=lcolor, alpha=0.85, lw=0))

    ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
    ax.set_axis_off()

    if stats:
        pp   = stats.get("pp_mean", float("nan"))
        cs   = stats.get("county_splits", float("nan"))
        pdev = stats.get("pop_dev_max", float("nan"))
        label = (f"PP mean: {pp:.3f}   County splits: {cs:.0f}   "
                 f"Max pop dev: {pdev*100:.2f}%")
        ax.text(0.02, 0.02, label, transform=ax.transAxes, fontsize=7,
                color="#333333", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, lw=0.5))


def render_state(abbr: str) -> None:
    abbr = abbr.upper()
    cfg = STATES_BY_ABBR[abbr]
    crs = cfg.crs
    fips = cfg.fips
    k = cfg.k

    final_dir = Path(f"data/{abbr.lower()}/final")
    compact_path = final_dir / "best_map_compact.gpkg"
    splits_path  = final_dir / "best_map_fewest_splits.gpkg"
    compact_stats_path = final_dir / "best_map_compact_stats.json"
    splits_stats_path  = final_dir / "best_map_fewest_splits_stats.json"

    if not compact_path.exists():
        # MD legacy path
        if abbr == "MD":
            final_dir = Path("data/maryland/final")
            compact_path = final_dir / "best_map_compact.gpkg"
            splits_path  = final_dir / "best_map_fewest_splits.gpkg"
            compact_stats_path = final_dir / "best_map_compact_stats.json"
            splits_stats_path  = final_dir / "best_map_fewest_splits_stats.json"
        if not compact_path.exists():
            print(f"[{abbr}] No final maps found — skipping.")
            return

    print(f"[{abbr}] Loading maps…")
    compact = gpd.read_file(compact_path).to_crs(crs)
    splits  = gpd.read_file(splits_path).to_crs(crs)
    compact_stats = load_stats(compact_stats_path)
    splits_stats  = load_stats(splits_stats_path)
    counties = load_counties(fips, crs)

    print(f"[{abbr}] Computing partisan lean (VEST 2020)…")
    compact_lean = district_lean(abbr, compact)
    splits_lean  = district_lean(abbr, splits)

    if compact_lean:
        for did in sorted(compact_lean):
            lbl, _ = lean_label(compact_lean[did])
            print(f"  Compact  D{did+1}: {lbl}  ({compact_lean[did]*100:.1f}% Biden)")
    else:
        print("  (no VEST data available)")

    dem_patch = mpatches.Patch(color="#1259a6", label="D (Biden majority)")
    rep_patch = mpatches.Patch(color="#c0152b", label="R (Trump majority)")
    footnote = (
        f"{cfg.name} Congressional Districts — Blind Redistricting  |  "
        "Drawn with geography only. Partisan lean is post-hoc: "
        "VEST 2020 precinct-level presidential results."
    )

    # ── Individual maps ────────────────────────────────────────────────────
    for gdf, stats, lean, fname, title in [
        (compact, compact_stats, compact_lean,
         "map_compact", "Best Compact\n(highest Polsby–Popper mean)"),
        (splits, splits_stats, splits_lean,
         "map_fewest_splits", "Fewest County Splits\n(minimum county fragmentation)"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#F5F5F0")
        ax.set_facecolor("#C8DCF0")
        plot_map(ax, gdf, counties, title, stats, lean_data=lean)
        if lean:
            fig.legend(handles=[dem_patch, rep_patch], loc="upper right",
                       fontsize=8, framealpha=0.9, edgecolor="#cccccc")
        fig.text(0.5, 0.01, footnote, ha="center", fontsize=6.5, color="#666666")
        out = final_dir / f"{fname}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved: {out}")

    # ── Side-by-side comparison ────────────────────────────────────────────
    ncols = 3 if (abbr == "MD" and CURRENT_DIST_MD.exists()) else 2
    fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, 8))
    fig.patch.set_facecolor("#F5F5F0")
    for ax in axes:
        ax.set_facecolor("#C8DCF0")

    col = 0
    if ncols == 3:
        current = gpd.read_file(CURRENT_DIST_MD).to_crs(crs)
        current["district_id"] = current["CD118FP"].astype(int) - 1
        current_lean = district_lean("MD", current)
        plot_map(axes[col], current, counties,
                 "Enacted Map (CD118, 2022)\nCurrent congressional districts",
                 None, lean_data=current_lean)
        axes[col].text(0.02, 0.02, "Enacted — drawn by Maryland legislature",
                       transform=axes[col].transAxes, fontsize=7, color="#333333",
                       va="bottom",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, lw=0.5))
        col += 1

    plot_map(axes[col], compact, counties,
             "Algorithmic: Best Compact\n(highest Polsby–Popper mean)",
             compact_stats, lean_data=compact_lean)
    plot_map(axes[col + 1], splits, counties,
             "Algorithmic: Fewest Splits\n(minimum county fragmentation)",
             splits_stats, lean_data=splits_lean)

    if compact_lean:
        fig.legend(handles=[dem_patch, rep_patch], loc="upper right",
                   fontsize=9, framealpha=0.9, edgecolor="#cccccc")

    subtitle = ("Enacted vs. Blind Algorithmic Maps  ·  " if ncols == 3
                else "Blind Algorithmic Maps  ·  ")
    fig.suptitle(
        f"{cfg.name} Congressional Districts — {subtitle}"
        "Post-hoc 2020 Presidential Lean Overlay",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.text(0.5, -0.02,
             "Weighted ReCom MCMC · geography only — zero partisan data used in drawing  ·  "
             "Lean: VEST 2020 precinct-level presidential results",
             ha="center", fontsize=7.5, color="#555555")
    plt.tight_layout(pad=1.5)
    out = final_dir / "map_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")
    print(f"[{abbr}] Done.\n")


if __name__ == "__main__":
    states = sys.argv[1:] if len(sys.argv) > 1 else ["MD"]
    for s in states:
        render_state(s)
