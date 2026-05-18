"""
Render a national partisan-spectrum map:
every district coloured on a continuous blue→white→red gradient
based on its 2024 Harris two-party share.
"""
import json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import geopandas as gpd

from render_us_map import (load_state_boundaries, load_districts,
                           find_state_dir, CRS_MAIN, CRS_AK, CRS_HI)
from state_configs import ALL_STATES

DATA_ROOT = Path("data")

# ── Lean data ─────────────────────────────────────────────────────────────────
with open(DATA_ROOT / "lean_2024.json") as f:
    lean_data = json.load(f)

# Build (abbr_upper, district_id_int) → harris_pct
lean_map: dict[tuple[str,int], float] = {}
for abbr, v in lean_data.items():
    if abbr in ("totals","margins"): continue
    for did, d in v["districts"].items():
        lean_map[(abbr.upper(), int(did))] = d["harris_pct"]

# ── Colour map ────────────────────────────────────────────────────────────────
# Custom diverging: deep red → white → deep blue, centred at 0.5
# Mirrored so that equal distance from 0.5 gives equal saturation.
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "partisan",
    [
        (0.00, "#8B0000"),   # deep red   (R+100)
        (0.15, "#C0152B"),   # strong red (R+70)
        (0.30, "#E06060"),   # moderate red
        (0.42, "#F4B8B8"),   # light red / lean R
        (0.48, "#F9E0E0"),   # very faint red
        (0.50, "#F5F0F8"),   # near-neutral / white-purple
        (0.52, "#DCE8F9"),   # very faint blue
        (0.58, "#AACCED"),   # light blue / lean D
        (0.70, "#4A90D9"),   # moderate blue
        (0.85, "#1259A6"),   # strong blue (D+70)
        (1.00, "#003087"),   # deep blue  (D+100)
    ],
    N=512,
)

def district_color(abbr: str, district_id: int) -> str:
    hp = lean_map.get((abbr.upper(), district_id), 0.5)
    return mcolors.to_hex(CMAP(hp))

# ── Load geometry ─────────────────────────────────────────────────────────────
states = load_state_boundaries()
states_main = states[~states["abbr"].isin(["AK","HI"])].copy()
states_ak   = states[states["abbr"] == "AK"].copy()
states_hi   = states[states["abbr"] == "HI"].copy()

all_districts, completed = [], set()
for cfg in sorted(ALL_STATES.values(), key=lambda c: c.abbr):
    if cfg.k == 1: continue
    gpkg = find_state_dir(cfg.abbr)
    if not gpkg: continue
    gdf = load_districts(cfg.abbr, gpkg)
    if gdf is not None:
        all_districts.append(gdf)
        completed.add(cfg.abbr)

at_large_abbrs = {cfg.abbr for cfg in ALL_STATES.values() if cfg.k == 1}

# Bounding box
minx, miny, maxx, maxy = states_main.total_bounds
px, py = (maxx-minx)*0.02, (maxy-miny)*0.04
xlim = (minx-px, maxx+px)
ylim = (miny-py, maxy+py)

# ── Figure ────────────────────────────────────────────────────────────────────
BG = "#0d1117"
fig = plt.figure(figsize=(30, 19), facecolor=BG)

fig.text(0.455, 0.975, "US Congressional Redistricting — Partisan Lean (2024 Presidential)",
         ha="center", va="top", fontsize=20, fontweight="bold", color="#e6edf3")
fig.text(0.455, 0.945,
         "Colour = two-party Harris share · Blind algorithm districts shaped by geography only",
         ha="center", va="top", fontsize=11, color="#aaaaaa", style="italic")
fig.text(0.455, 0.922,
         "Partisan lean applied post-hoc from VEST 2020 precinct data adjusted for 2024 state-level swing · "
         "The redistricting algorithm never saw this data",
         ha="center", va="top", fontsize=9.5, color="#666666")

ax = fig.add_axes([0.01, 0.10, 0.83, 0.80])
ax.set_facecolor("#1a2030")
ax.set_xlim(*xlim); ax.set_ylim(*ylim)
ax.set_aspect("equal"); ax.set_axis_off()

# Draw districts
for gdf in all_districts:
    abbr = gdf["abbr"].iloc[0]
    if abbr in ("AK","HI"): continue
    colors = [district_color(abbr, int(row["district_id"])) for _, row in gdf.iterrows()]
    gdf.sort_values("district_id").plot(ax=ax, color=colors,
                                        edgecolor="#00000055", linewidth=0.25)

# At-large states
for _, row in states_main[states_main["abbr"].isin(at_large_abbrs)].iterrows():
    color = district_color(row["abbr"], 0)
    gpd.GeoDataFrame([row], geometry=[row.geometry], crs=CRS_MAIN).plot(
        ax=ax, color=color, edgecolor="#33333388", linewidth=0.5)

# State outlines
states_main.boundary.plot(ax=ax, edgecolor="#33333399", linewidth=0.65, alpha=0.9)

# State labels
label_skip = {"RI","CT","DE","NJ","MA","MD","DC"}
xl0,xl1 = xlim; yl0,yl1 = ylim
for _, row in states_main.iterrows():
    if not row["abbr"] or row["abbr"] in label_skip: continue
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    if not (xl0 < cx < xl1 and yl0 < cy < yl1): continue
    ax.text(cx, cy, row["abbr"], ha="center", va="center",
            fontsize=6.5, fontweight="bold", color="white", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="#00000066", lw=0))

# ── Alaska inset ──────────────────────────────────────────────────────────────
ax_ak = fig.add_axes([0.01, 0.01, 0.22, 0.20])
ax_ak.set_facecolor("#1a2030"); ax_ak.set_aspect("equal"); ax_ak.set_axis_off()
if not states_ak.empty:
    ak_proj = states_ak.to_crs(CRS_AK)
    ak_color = district_color("AK", 0)
    ak_proj.plot(ax=ax_ak, color=ak_color, edgecolor="#444444", linewidth=0.6)
    cx = ak_proj.geometry.iloc[0].centroid.x
    cy = ak_proj.geometry.iloc[0].centroid.y
    ax_ak.text(cx, cy, "AK\n(at-large)", ha="center", va="center",
               fontsize=8, fontweight="bold", color="#cccccc")

# ── Hawaii inset ──────────────────────────────────────────────────────────────
ax_hi = fig.add_axes([0.23, 0.01, 0.14, 0.14])
ax_hi.set_facecolor("#1a2030"); ax_hi.set_aspect("equal"); ax_hi.set_axis_off()
hi_gpkg = find_state_dir("HI")
if hi_gpkg and not states_hi.empty:
    hi_dist = load_districts("HI", hi_gpkg)
    if hi_dist is not None:
        hi_proj = hi_dist.to_crs(CRS_HI)
        colors = [district_color("HI", int(r["district_id"])) for _, r in hi_proj.iterrows()]
        hi_proj.plot(ax=ax_hi, color=colors, edgecolor="#ffffff55", linewidth=0.4)
        states_hi.to_crs(CRS_HI).boundary.plot(ax=ax_hi, edgecolor="#444444", linewidth=0.6)

# ── Colour-bar legend ─────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.845, 0.12, 0.018, 0.72])
norm = mcolors.Normalize(vmin=0, vmax=1)
sm = matplotlib.cm.ScalarMappable(cmap=CMAP, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.ax.yaxis.set_tick_params(color="#aaaaaa", labelsize=8.5)
cb.outline.set_edgecolor("#444444")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="#aaaaaa")

# Custom tick labels
ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
tick_labels = ["R+100","R+80","R+60","R+40","R+20","Even",
               "D+20","D+40","D+60","D+80","D+100"]
cb.set_ticks(ticks)
cb.set_ticklabels(tick_labels)
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.set_tick_params(color="#888888")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="#cccccc")

# Side legend text
fig.text(0.874, 0.865, "Harris\ntwo-party\nshare", ha="center", va="top",
         fontsize=8.5, color="#aaaaaa", linespacing=1.5)
fig.text(0.874, 0.125, "2024\nPresidential\nresults", ha="center", va="bottom",
         fontsize=8.5, color="#aaaaaa", linespacing=1.5)

# Count labels
fig.text(0.874, 0.835, f"D  204", ha="center", va="top",
         fontsize=9, fontweight="bold", color="#4A90D9")
fig.text(0.874, 0.155, f"R  231", ha="center", va="bottom",
         fontsize=9, fontweight="bold", color="#C0152B")

out = DATA_ROOT / "us_map_partisan.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved: {out}")
