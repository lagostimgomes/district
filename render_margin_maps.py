"""Render 3 national maps with competitive districts highlighted."""
import json, pickle
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from render_us_map import (load_state_boundaries, load_districts,
                           find_state_dir, AT_LARGE, CRS_MAIN, CRS_AK, CRS_HI)
from state_configs import ALL_STATES

DATA_ROOT = Path("data")
HIGHLIGHT  = "#FFD700"
MUTED_DIST = "#3a3f47"
STATE_EDGE = "#555555"
BG         = "#0d1117"
SURFACE    = "#161b22"

with open(DATA_ROOT / "lean_2024.json") as f:
    lean = json.load(f)

def get_highlight_set(threshold):
    """Return set of (abbr, district_id_int) for districts with margin <= threshold."""
    s = set()
    for abbr, v in lean.items():
        if abbr in ("totals", "margins"): continue
        for did, d in v["districts"].items():
            if d["margin"] <= threshold:
                s.add((abbr.upper(), int(did)))
    return s

def render_margin_map(threshold, out_path, title_suffix):
    highlight_set = get_highlight_set(threshold)
    n_highlighted = len(highlight_set)
    print(f"  Highlighting {n_highlighted} districts (margin <= {threshold}%)")

    states = load_state_boundaries()
    states_main = states[~states["abbr"].isin(["AK","HI"])].copy()
    states_ak   = states[states["abbr"] == "AK"].copy()

    all_districts = []
    completed = set()
    for cfg in sorted(ALL_STATES.values(), key=lambda c: c.abbr):
        if cfg.k == 1: continue
        gpkg = find_state_dir(cfg.abbr)
        if gpkg is None: continue
        gdf = load_districts(cfg.abbr, gpkg)
        if gdf is not None:
            all_districts.append(gdf)
            completed.add(cfg.abbr)

    at_large_abbrs = {cfg.abbr for cfg in ALL_STATES.values() if cfg.k == 1}

    minx, miny, maxx, maxy = states_main.total_bounds
    pad_x = (maxx - minx) * 0.02
    pad_y = (maxy - miny) * 0.04
    xlim = (minx - pad_x, maxx + pad_x)
    ylim = (miny - pad_y, maxy + pad_y)

    fig = plt.figure(figsize=(30, 19), facecolor=BG)
    fig.text(0.455, 0.975, "US Congressional Redistricting — Blind Algorithmic Maps",
             ha="center", va="top", fontsize=20, fontweight="bold", color="#e6edf3")
    fig.text(0.455, 0.945, title_suffix,
             ha="center", va="top", fontsize=12, color=HIGHLIGHT, fontweight="bold")
    fig.text(0.455, 0.922, f"{n_highlighted} districts highlighted in gold — all others shown in grey",
             ha="center", va="top", fontsize=9.5, color="#888888")

    ax = fig.add_axes([0.01, 0.10, 0.85, 0.80])
    ax.set_facecolor("#1a2030")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.set_axis_off()

    for gdf in all_districts:
        abbr = gdf["abbr"].iloc[0]
        if abbr in ("AK", "HI"): continue
        colors = []
        for _, row in gdf.iterrows():
            key = (abbr, int(row["district_id"]))
            colors.append(HIGHLIGHT if key in highlight_set else MUTED_DIST)
        gdf_sorted = gdf.sort_values("district_id")
        gdf_sorted.plot(ax=ax, color=colors, edgecolor="#222222", linewidth=0.2)

    # Failed/missing states in dark grey
    failed = {cfg.abbr for cfg in ALL_STATES.values()
              if cfg.k > 1 and cfg.abbr not in completed}
    if failed:
        states_main[states_main["abbr"].isin(failed)].plot(
            ax=ax, color="#222222", edgecolor=STATE_EDGE, linewidth=0.5)

    # At-large states
    at_large_main = states_main[states_main["abbr"].isin(at_large_abbrs)]
    # Check if their single district is highlighted
    def at_large_color(abbr):
        key = (abbr, 0)
        return HIGHLIGHT if key in highlight_set else "#2a3040"
    for _, row in at_large_main.iterrows():
        color = at_large_color(row["abbr"])
        gpd.GeoDataFrame([row], geometry=[row.geometry], crs=CRS_MAIN).plot(
            ax=ax, color=color, edgecolor=STATE_EDGE, linewidth=0.5)

    states_main.boundary.plot(ax=ax, edgecolor="#444444", linewidth=0.7, alpha=0.9)

    # Alaska inset
    ax_ak = fig.add_axes([0.01, 0.01, 0.22, 0.20])
    ax_ak.set_facecolor("#1a2030"); ax_ak.set_aspect("equal"); ax_ak.set_axis_off()
    if not states_ak.empty:
        ak_proj = states_ak.to_crs(CRS_AK)
        ak_col = at_large_color("AK")
        ak_proj.plot(ax=ax_ak, color=ak_col, edgecolor="#444444", linewidth=0.6)
        cx = ak_proj.geometry.iloc[0].centroid.x
        cy = ak_proj.geometry.iloc[0].centroid.y
        ax_ak.text(cx, cy, "AK\n(at-large)", ha="center", va="center",
                   fontsize=8, fontweight="bold", color="#aaa")

    # Legend
    legend_ax = fig.add_axes([0.86, 0.10, 0.13, 0.80])
    legend_ax.set_axis_off()
    def _lrow(y, c, lbl):
        p = mpatches.FancyBboxPatch((0.04,y-0.028),0.18,0.056,
            boxstyle="round,pad=0.01",facecolor=c,edgecolor="#666",linewidth=0.8,
            transform=legend_ax.transAxes)
        legend_ax.add_patch(p)
        legend_ax.text(0.28,y,lbl,fontsize=9,va="center",
                       transform=legend_ax.transAxes,color="#dddddd",linespacing=1.4)
    legend_ax.text(0.04,0.97,"Legend",fontsize=13,fontweight="bold",
                   transform=legend_ax.transAxes,va="top",color="#e6edf3")
    _lrow(0.88,HIGHLIGHT,f"Competitive\n(≤{threshold}% margin)\n{n_highlighted} districts")
    _lrow(0.73,MUTED_DIST,"Other\ndistricts")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")

for thresh, suffix, fname in [
    (2, "≤ 2% margin highlighted — 20 toss-up districts", "us_map_margin2.png"),
    (5, "≤ 5% margin highlighted — 50 competitive districts", "us_map_margin5.png"),
    (8, "≤ 8% margin highlighted — lean districts", "us_map_margin8.png"),
]:
    print(f"\nRendering margin{thresh} map...")
    render_margin_map(thresh, DATA_ROOT / fname, suffix)

print("\nDone.")
