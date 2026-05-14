"""
render_maps.py — Visualise the two selected Maryland district maps.

Produces:
    data/maryland/final/map_compact.png
    data/maryland/final/map_fewest_splits.png
    data/maryland/final/map_comparison.png  (side-by-side)
    data/maryland/final/pareto_frontier.png

Post-hoc partisan lean overlay uses VEST 2020 precinct-level presidential
election results. The districts were drawn with zero partisan data.
"""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

from pipeline.lean import district_lean as vest_district_lean, lean_label as vest_lean_label

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FINAL_DIR    = Path("data/maryland/final")
COUNTY_DIR   = Path("data/maryland/counties")
PRECINCT_GDF = Path("data/maryland/graph/md_precincts_pop.gpkg")
PLANS_PATH   = Path("data/maryland/ensemble/plans.parquet")

COMPACT_PATH    = FINAL_DIR / "best_map_compact.gpkg"
SPLITS_PATH     = FINAL_DIR / "best_map_fewest_splits.gpkg"
COMPACT_STATS   = FINAL_DIR / "best_map_compact_stats.json"
SPLITS_STATS    = FINAL_DIR / "best_map_fewest_splits_stats.json"
PARETO_PATH     = FINAL_DIR / "pareto_frontier.csv"
REPORT_PATH     = FINAL_DIR / "report.json"
CURRENT_DIST    = Path("data/maryland/current_districts/tl_2022_24_cd118.shp")

CRS = "EPSG:32618"

# 8 visually distinct colours for districts (colourblind-friendly palette).
DISTRICT_COLOURS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
]

# ---------------------------------------------------------------------------
# 2020 Maryland Presidential Election — official certified county results
# Source: Maryland State Board of Elections canvass
# ---------------------------------------------------------------------------

# county_fips (3-digit) → (biden_votes, trump_votes)
COUNTY_FIPS_TO_NAME = {
    "001": "Allegany", "003": "Anne Arundel", "005": "Baltimore",
    "510": "Baltimore City", "009": "Calvert", "011": "Caroline",
    "013": "Carroll", "015": "Cecil", "017": "Charles",
    "019": "Dorchester", "021": "Frederick", "023": "Garrett",
    "025": "Harford", "027": "Howard", "029": "Kent",
    "031": "Montgomery", "033": "Prince George's", "035": "Queen Anne's",
    "037": "St. Mary's", "039": "Somerset", "041": "Talbot",
    "043": "Washington", "045": "Wicomico", "047": "Worcester",
}

_COUNTY_RESULTS_BY_NAME = {
    "Allegany":       (9158,   20886),
    "Anne Arundel":   (172823, 127821),
    "Baltimore":      (258409, 146202),
    "Baltimore City": (207260,  25374),
    "Calvert":        (22587,   25346),
    "Caroline":       (5095,    10283),
    "Carroll":        (36456,   60218),
    "Cecil":          (16809,   29439),
    "Charles":        (62171,   25579),
    "Dorchester":     (6857,     8764),
    "Frederick":      (77675,   63682),
    "Garrett":        (3281,    12002),
    "Harford":        (63095,   80930),
    "Howard":         (129433,  48390),
    "Kent":           (5329,     5195),
    "Montgomery":     (419569, 101222),
    "Prince George's":(379208,  37090),
    "Queen Anne's":   (10709,   18741),
    "St. Mary's":     (23138,   30826),
    "Somerset":       (4241,     5739),
    "Talbot":         (11062,   10946),
    "Washington":     (26044,   40224),
    "Wicomico":       (22054,   22944),
    "Worcester":      (12560,   18571),
}

# Pre-compute Biden vote share (0–1) per 3-digit county FIPS.
COUNTY_BIDEN_PCT: dict[str, float] = {}
for fips, name in COUNTY_FIPS_TO_NAME.items():
    b, t = _COUNTY_RESULTS_BY_NAME[name]
    COUNTY_BIDEN_PCT[fips] = b / (b + t)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_counties():
    shp_files = list(COUNTY_DIR.rglob("*.shp"))
    if not shp_files:
        return None
    counties = gpd.read_file(shp_files[0]).to_crs(CRS)
    state_col = next((c for c in counties.columns if c.upper() in ("STATEFP20", "STATEFP")), None)
    if state_col:
        counties = counties[counties[state_col] == "24"]
    return counties


def load_stats(path):
    with open(path) as f:
        return json.load(f)


def district_cmap(n=8):
    return ListedColormap(DISTRICT_COLOURS[:n])


# ---------------------------------------------------------------------------
# Partisan lean computation  (delegates to pipeline/lean.py — VEST precinct data)
# ---------------------------------------------------------------------------

def compute_district_lean(districts: gpd.GeoDataFrame) -> dict[int, float]:
    """
    Returns {district_id: biden_pct} using VEST 2020 precinct-level data.
    Falls back to an empty dict if VEST data is unavailable.
    """
    return vest_district_lean("MD", districts)


def load_current_districts() -> tuple[gpd.GeoDataFrame, dict[int, float]]:
    """Load enacted CD118 districts and compute VEST-based partisan lean."""
    current = gpd.read_file(CURRENT_DIST).to_crs(CRS)
    current["district_id"] = current["CD118FP"].astype(int) - 1
    lean = vest_district_lean("MD", current)
    return current, lean


def lean_label(biden_pct: float) -> tuple[str, str]:
    return vest_lean_label(biden_pct)


def plot_map(ax, districts, counties, title, stats, lean: dict | None = None):
    """Draw one district map on the given axes."""
    cmap = district_cmap(len(districts))

    districts = districts.sort_values("district_id")
    districts.plot(
        ax=ax,
        column="district_id",
        cmap=cmap,
        linewidth=0.8,
        edgecolor="white",
        legend=False,
    )

    if counties is not None:
        counties.boundary.plot(ax=ax, linewidth=0.4, color="#444444", alpha=0.5)

    for _, row in districts.iterrows():
        dist_id = int(row["district_id"])
        centroid = row.geometry.centroid

        # District number label.
        ax.annotate(
            str(dist_id + 1),
            xy=(centroid.x, centroid.y),
            ha="center", va="center" if lean is None else "bottom",
            fontsize=9, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, lw=0),
        )

        # Party lean label (below district number).
        if lean and dist_id in lean:
            label, lcolor = lean_label(lean[dist_id])
            ax.annotate(
                label,
                xy=(centroid.x, centroid.y),
                xytext=(0, -14), textcoords="offset points",
                ha="center", va="top",
                fontsize=7.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc=lcolor, alpha=0.85, lw=0),
            )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_axis_off()

    pp_mean = stats.get("pp_mean", 0)
    splits  = stats.get("county_splits", "?")
    pop_dev = stats.get("pop_dev_max", 0)
    ax.text(
        0.02, 0.02,
        f"PP mean: {pp_mean:.3f}   County splits: {splits:.0f}   Max pop dev: {pop_dev*100:.2f}%",
        transform=ax.transAxes,
        fontsize=7.5, color="#333333",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, lw=0.5),
    )


def legend_patches(n=8):
    return [
        mpatches.Patch(color=DISTRICT_COLOURS[i], label=f"District {i+1}")
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    compact = gpd.read_file(COMPACT_PATH).to_crs(CRS)
    splits  = gpd.read_file(SPLITS_PATH).to_crs(CRS)
    counties = load_counties()
    compact_stats = load_stats(COMPACT_STATS)
    splits_stats  = load_stats(SPLITS_STATS)
    report = load_stats(REPORT_PATH)

    compact_row = report["best_map_compact"]["plan_row"]
    splits_row  = report["best_map_fewest_splits"]["plan_row"]

    print(f"  Compact map   : pp_mean={compact_stats['pp_mean']:.3f}, "
          f"county_splits={compact_stats['county_splits']:.0f}")
    print(f"  Fewest-splits : pp_mean={splits_stats['pp_mean']:.3f}, "
          f"county_splits={splits_stats['county_splits']:.0f}")

    print("Computing post-hoc partisan lean (VEST 2020 precinct-level data)…")
    compact_lean = compute_district_lean(compact)
    splits_lean  = compute_district_lean(splits)
    current_districts, current_lean = load_current_districts()

    for dist_id, biden_pct in sorted(compact_lean.items()):
        lbl, _ = lean_label(biden_pct)
        print(f"  Compact  D{dist_id+1}: {lbl}  ({biden_pct*100:.1f}% Biden)")
    for dist_id, biden_pct in sorted(splits_lean.items()):
        lbl, _ = lean_label(biden_pct)
        print(f"  Splits   D{dist_id+1}: {lbl}  ({biden_pct*100:.1f}% Biden)")

    FOOTNOTE = (
        "Maryland Congressional Districts — Blind Redistricting v1.0 | "
        "Districts drawn with geography only. Partisan lean is post-hoc: "
        "VEST 2020 precinct-level presidential results (Biden vs Trump)."
    )

    # ── Individual maps ────────────────────────────────────────────────────

    for gdf, stats, lean, name, label in [
        (compact, compact_stats, compact_lean, "map_compact",
         "Best Compact Map\n(highest Polsby–Popper mean)"),
        (splits, splits_stats, splits_lean, "map_fewest_splits",
         "Fewest County Splits Map\n(minimum county fragmentation)"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        fig.patch.set_facecolor("#F5F5F0")
        ax.set_facecolor("#D0E8F5")
        plot_map(ax, gdf, counties, label, stats, lean=lean)
        fig.legend(
            handles=legend_patches(len(gdf)),
            loc="lower right", ncol=2, fontsize=8,
            framealpha=0.9, edgecolor="#cccccc",
        )
        # Party lean legend
        dem_patch = mpatches.Patch(color="#1259a6", label="D (Biden majority)")
        rep_patch = mpatches.Patch(color="#c0152a", label="R (Trump majority)")
        fig.legend(
            handles=[dem_patch, rep_patch],
            loc="upper right", fontsize=8, framealpha=0.9, edgecolor="#cccccc",
        )
        fig.text(0.5, 0.01, FOOTNOTE, ha="center", fontsize=6.5, color="#666666",
                 wrap=True)
        out = FINAL_DIR / f"{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved: {out}")

    # ── Three-way comparison ───────────────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.patch.set_facecolor("#F5F5F0")
    for ax in axes:
        ax.set_facecolor("#D0E8F5")

    # Current enacted map — no Stats object, build a minimal one.
    current_stats = {"pp_mean": float("nan"), "county_splits": float("nan"), "pop_dev_max": float("nan")}

    plot_map(axes[0], current_districts, counties,
             "Enacted Map (CD118, 2022)\nCurrent congressional districts",
             current_stats, lean=current_lean)
    # Override stats box with simpler label for enacted map.
    axes[0].texts[-1].set_visible(False)
    axes[0].text(
        0.02, 0.02, "Enacted map — drawn by Maryland legislature",
        transform=axes[0].transAxes, fontsize=7.5, color="#333333",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, lw=0.5),
    )

    plot_map(axes[1], compact, counties,
             "Algorithmic: Best Compact\n(highest Polsby–Popper mean)",
             compact_stats, lean=compact_lean)
    plot_map(axes[2], splits,  counties,
             "Algorithmic: Fewest Splits\n(minimum county fragmentation)",
             splits_stats, lean=splits_lean)

    fig.legend(
        handles=legend_patches(8),
        loc="lower center", ncol=8, fontsize=9,
        framealpha=0.9, edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.02),
    )
    dem_patch = mpatches.Patch(color="#1259a6", label="D (Biden majority)")
    rep_patch = mpatches.Patch(color="#c0152a", label="R (Trump majority)")
    fig.legend(
        handles=[dem_patch, rep_patch],
        loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#cccccc",
    )
    fig.suptitle(
        "Maryland Congressional Districts — Enacted vs. Blind Algorithmic Maps\n"
        "Post-hoc 2020 Presidential Lean Overlay",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.06,
        "Algorithmic maps: Weighted ReCom MCMC · 10,000-plan ensemble · geography only — zero partisan data used in drawing · "
        "Lean labels: VEST 2020 precinct-level presidential results",
        ha="center", fontsize=7.5, color="#555555",
    )

    plt.tight_layout(pad=1.5)
    out = FINAL_DIR / "map_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── Pareto scatter ─────────────────────────────────────────────────────

    import pandas as pd
    pareto = pd.read_csv(PARETO_PATH)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#F5F5F0")
    ax.set_facecolor("#FAFAF8")

    ax.scatter(pareto["county_splits"], pareto["pp_mean"],
               c="#4E79A7", s=60, alpha=0.85, zorder=3, label="Pareto-optimal plan")

    # Highlight the two selected plans.
    best_compact = pareto.loc[pareto["pp_mean"].idxmax()]
    best_splits  = pareto.loc[pareto["county_splits"].idxmin()]

    ax.scatter(best_compact["county_splits"], best_compact["pp_mean"],
               c="#E15759", s=120, zorder=5, label="Selected: best compact")
    ax.scatter(best_splits["county_splits"],  best_splits["pp_mean"],
               c="#59A14F", s=120, marker="D", zorder=5, label="Selected: fewest splits")

    ax.annotate("Best compact", xy=(best_compact["county_splits"], best_compact["pp_mean"]),
                xytext=(4, 6), textcoords="offset points", fontsize=8, color="#E15759")
    ax.annotate("Fewest splits", xy=(best_splits["county_splits"], best_splits["pp_mean"]),
                xytext=(4, -12), textcoords="offset points", fontsize=8, color="#59A14F")

    ax.set_xlabel("County splits (fewer = better)", fontsize=10)
    ax.set_ylabel("PP mean — compactness (higher = better)", fontsize=10)
    ax.set_title("Pareto Frontier: Compactness vs. County Splits", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.text(0.5, -0.02,
             "Each point is a non-dominated plan from the 10,000-plan ensemble.",
             ha="center", fontsize=8, color="#666666")

    out = FINAL_DIR / "pareto_frontier.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── Convergence diagnostics ────────────────────────────────────────────

    print("Building convergence diagnostics…")
    metrics = pd.read_parquet(Path("data/maryland/ensemble/metrics.parquet"))

    # Hard-filter mask (same as Script 4).
    PP_MIN_THRESH  = 0.10
    POP_DEV_THRESH = 0.005
    valid = (metrics["pp_min"] >= PP_MIN_THRESH) & (metrics["pop_dev_max"] <= POP_DEV_THRESH)

    steps = metrics["step"].values

    # Running best of each key metric (over ALL plans, then over valid-only).
    run_max_pp   = metrics["pp_mean"].cummax()
    run_min_cs   = metrics["cut_edges"].cummin()   # proxy for county splits (not stored)
    valid_pp     = metrics["pp_mean"].where(valid)
    run_max_pp_v = valid_pp.cummax()

    # Running fraction of valid plans.
    run_valid_frac = valid.cumsum() / (steps + 1)

    # Pareto-frontier size over time: non-dominated in (pp_mean, cut_edges) among valid.
    # Compute at logarithmically-spaced checkpoints for speed.
    checkpoints = np.unique(np.geomspace(1, len(metrics) - 1, 60).astype(int))
    pareto_sizes = []
    for cp in checkpoints:
        sub = metrics.iloc[: cp + 1]
        sub_v = sub[valid.iloc[: cp + 1]]
        if sub_v.empty:
            pareto_sizes.append(0)
            continue
        pp  = sub_v["pp_mean"].values
        cs  = sub_v["cut_edges"].values
        dominated = np.zeros(len(pp), dtype=bool)
        for i in range(len(pp)):
            if dominated[i]:
                continue
            dominated |= (pp >= pp[i]) & (cs <= cs[i])
            dominated[i] = False
        pareto_sizes.append(int((~dominated).sum()))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor("#F5F5F0")
    for ax in axes.flat:
        ax.set_facecolor("#FAFAF8")
        ax.grid(True, alpha=0.25, linestyle="--")

    # 1. Running-best pp_mean (all vs valid).
    ax = axes[0, 0]
    ax.plot(steps, run_max_pp,   color="#4E79A7", lw=1.5, label="All plans")
    ax.plot(steps, run_max_pp_v, color="#E15759", lw=1.5, label="Valid plans only")
    ax.axhline(metrics["pp_mean"].max(), color="#E15759", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("MCMC step", fontsize=9)
    ax.set_ylabel("Running max PP mean", fontsize=9)
    ax.set_title("Compactness (PP mean) convergence", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # 2. Running-min cut edges.
    ax = axes[0, 1]
    ax.plot(steps, run_min_cs, color="#59A14F", lw=1.5)
    ax.axhline(metrics["cut_edges"].min(), color="#59A14F", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("MCMC step", fontsize=9)
    ax.set_ylabel("Running min cut edges", fontsize=9)
    ax.set_title("Cut-edge minimisation convergence", fontsize=10, fontweight="bold")

    # 3. Running fraction of valid plans.
    ax = axes[1, 0]
    ax.plot(steps, run_valid_frac * 100, color="#EDC948", lw=1.5)
    ax.axhline(valid.mean() * 100, color="#EDC948", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("MCMC step", fontsize=9)
    ax.set_ylabel("Valid plans (%)", fontsize=9)
    ax.set_title("Fraction of plans passing hard filters", fontsize=10, fontweight="bold")

    # 4. Pareto-frontier size over time.
    ax = axes[1, 1]
    ax.plot(steps[checkpoints], pareto_sizes, color="#B07AA1", lw=1.5, marker="o",
            markersize=3)
    ax.set_xlabel("MCMC step", fontsize=9)
    ax.set_ylabel("Pareto-frontier size", fontsize=9)
    ax.set_title("Pareto frontier growth over chain", fontsize=10, fontweight="bold")

    fig.suptitle(
        "MCMC Convergence Diagnostics — Does More Steps Help?",
        fontsize=12, fontweight="bold",
    )
    fig.text(
        0.5, -0.01,
        "Curves that flatten early → current ensemble is sufficient. "
        "Curves still rising at step 10,000 → more steps would improve results.",
        ha="center", fontsize=8, color="#555555",
    )
    plt.tight_layout(pad=1.8)
    out = FINAL_DIR / "convergence_diagnostics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")

    print("\nDone. Open the PNGs in data/maryland/final/")


if __name__ == "__main__":
    main()
