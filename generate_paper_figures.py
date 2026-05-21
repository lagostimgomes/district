"""
Generate all four main figures for research_paper_nature.pdf.
Saves PNGs to figures/fig1.png … fig4.png.
Run: python generate_paper_figures.py
"""
import warnings; warnings.filterwarnings("ignore")
import json, math
from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from shapely.ops import unary_union
from scipy.stats import gaussian_kde

# ── Paths ────────────────────────────────────────────────────────────────────
FIGS     = Path("figures");  FIGS.mkdir(exist_ok=True)
DATA     = Path("data")
COUNTY_SHP = DATA / "maryland" / "counties" / "tl_2020_us_county.shp"

# ── Load data ────────────────────────────────────────────────────────────────
blind   = json.load(open(DATA / "lean_2024.json"))
enacted = json.load(open(DATA / "enacted_lean_2024.json"))

from state_configs import ALL_STATES

# Per-state pp_mean and county_splits from report.json
state_stats = {}
for cfg in ALL_STATES.values():
    if cfg.k == 1: continue
    rpt = DATA / cfg.abbr.lower() / "final" / "report.json"
    if not rpt.exists(): continue
    d = json.load(open(rpt))
    bmc = d.get("best_map_compact", {})
    state_stats[cfg.abbr] = {
        "pp_mean":       bmc.get("pp_mean", 0),
        "county_splits": bmc.get("county_splits", 0),
        "k":             cfg.k,
        "fips":          cfg.fips,
        "name":          cfg.name,
    }

# Seat counts per state
def seat_counts(data):
    out = {}
    for s, v in data.items():
        if isinstance(v, dict) and "D" in v:
            out[s] = {"D": v["D"], "R": v["R"]}
        elif isinstance(v, str):
            out[s] = {"D": 1 if v=="D" else 0, "R": 1 if v=="R" else 0}
    return out

blind_seats   = seat_counts(blind)
enacted_seats = seat_counts(enacted)

# Deviation: blind D − enacted D (positive = blind more D)
deviations = {
    s: blind_seats[s]["D"] - enacted_seats[s]["D"]
    for s in blind_seats if s in enacted_seats
}

# Competitive seat counts from district-level margin data
def competitive(data, threshold):
    n = 0
    for v in data.values():
        if isinstance(v, dict) and "districts" in v:
            for d in v["districts"].values():
                if d["margin"] <= threshold:
                    n += 1
    return n

comp_blind_5  = competitive(blind, 5)
comp_blind_10 = competitive(blind, 10)
comp_enacted_5  = competitive(enacted, 5)
comp_enacted_10 = competitive(enacted, 10)

# All district margins
def all_margins(data):
    m = []
    for v in data.values():
        if isinstance(v, dict) and "districts" in v:
            for d in v["districts"].values():
                m.append(d["margin"] / 100)
    return np.array(m)

blind_margins   = all_margins(blind)
enacted_margins = all_margins(enacted)

# ── Style constants ───────────────────────────────────────────────────────────
BLUE  = "#2166ac"
RED   = "#d6604d"
GRAY  = "#aaaaaa"
DARK  = "#1a1a1a"
LIGHT = "#f5f5f5"
ACC   = "#c0392b"   # accent red (matches CSS)

plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "figure.dpi":       150,
})

# ── Load state boundaries ─────────────────────────────────────────────────────
print("Loading state boundaries…")
counties_all = gpd.read_file(COUNTY_SHP).to_crs(epsg=5070)

# Skip non-contiguous / territories
SKIP_FIPS = {"02","15","60","66","69","72","78"}   # AK, HI, territories
states_cont = (
    counties_all[~counties_all["STATEFP"].isin(SKIP_FIPS)]
    .dissolve(by="STATEFP")
    .reset_index()[["STATEFP", "geometry"]]
)

# Build abbr→fips mapping
fips2abbr = {cfg.fips.zfill(2): cfg.abbr for cfg in ALL_STATES.values()}
states_cont["abbr"] = states_cont["STATEFP"].map(fips2abbr)
states_cont = states_cont.dropna(subset=["abbr"])

# State centroids for labels
states_cont["centroid"] = states_cont.geometry.centroid


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — National comparison: deviation choropleth + seat-count bar
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 1…")

fig1, axes = plt.subplots(
    1, 2, figsize=(11, 4.4),
    gridspec_kw={"width_ratios": [1.65, 1]},
    facecolor="white"
)

# ── Panel a: choropleth ───────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#d4e8f7")
ax.set_axis_off()
ax.set_title("a   Partisan seat deviation  (blind − enacted Democratic seats)",
             loc="left", fontsize=9, fontweight="bold", pad=6)

cmap = LinearSegmentedColormap.from_list(
    "bwr", [(0.0, "#c0392b"), (0.5, "#f5f5f5"), (1.0, "#2166ac")]
)
norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)

states_cont["dev"] = states_cont["abbr"].map(deviations).fillna(0)

# Base fill
states_cont.plot(ax=ax, color="#e8e8e8", edgecolor="white", linewidth=0.3)
# Deviation fill
sc = states_cont.plot(
    ax=ax, column="dev", cmap=cmap, norm=norm,
    edgecolor="white", linewidth=0.4, missing_kwds={"color": "#e0e0e0"}
)

# State labels for those with nonzero deviation
label_offsets = {
    "IL": (0, 0), "NY": (60000, -40000), "PA": (0, 0),
    "TX": (0, 0), "WI": (0, 0),
}
for _, row in states_cont.iterrows():
    abbr = row["abbr"]
    dev  = deviations.get(abbr, 0)
    if abs(dev) >= 1:
        cx, cy = row["centroid"].x, row["centroid"].y
        dx, dy = label_offsets.get(abbr, (0, 0))
        ax.annotate(
            f"{abbr}\n{'+' if dev>0 else ''}{dev}D",
            xy=(cx + dx, cy + dy),
            fontsize=6.5, ha="center", va="center",
            color="white" if abs(dev) >= 2 else DARK,
            fontweight="bold"
        )

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig1.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035,
                     pad=0.02, aspect=28, ticks=[-3,-2,-1,0,1,2,3])
cbar.ax.set_xticklabels(["-3D","-2D","-1D","0","+1D","+2D","+3D"], fontsize=7)
cbar.ax.tick_params(size=0)
cbar.outline.set_visible(False)

# ── Panel b: seat totals + competitive bars ───────────────────────────────────
ax2 = axes[1]
ax2.set_title("b   Seat distribution and electoral competition",
              loc="left", fontsize=9, fontweight="bold", pad=6)

categories = ["National seats", "Competitive\n(≤5%)", "Competitive\n(≤10%)"]
blind_vals   = [204,  comp_blind_5,  comp_blind_10]
enacted_vals = [203,  comp_enacted_5, comp_enacted_10]
xs = np.arange(len(categories))
w  = 0.32

b1 = ax2.bar(xs - w/2, blind_vals,   w, label="Blind maps",   color=BLUE, alpha=0.85)
b2 = ax2.bar(xs + w/2, enacted_vals, w, label="Enacted maps", color=RED,  alpha=0.85)

# Value labels
for bar in list(b1) + list(b2):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             str(int(bar.get_height())), ha="center", va="bottom",
             fontsize=7.5, color=DARK)

# Difference annotation on competitive bars
for i in range(1, 3):
    diff = blind_vals[i] - enacted_vals[i]
    pct  = diff / enacted_vals[i] * 100
    ax2.annotate(
        f"+{diff} ({pct:.0f}%)",
        xy=(xs[i], max(blind_vals[i], enacted_vals[i]) + 5),
        ha="center", fontsize=7, color=BLUE, fontweight="bold"
    )

ax2.set_xticks(xs)
ax2.set_xticklabels(categories, fontsize=8)
ax2.set_ylabel("Number of seats", fontsize=8)
ax2.set_ylim(0, max(blind_vals) * 1.18)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(25))
ax2.legend(fontsize=7.5, loc="upper right", framealpha=0.7)
ax2.axhline(0, color="black", linewidth=0.5)

# D/R split label for national bar
ax2.text(xs[0] - w/2, blind_vals[0] / 2, "D",
         ha="center", va="center", fontsize=8, color="white", fontweight="bold")
ax2.text(xs[0] + w/2, enacted_vals[0] / 2, "D",
         ha="center", va="center", fontsize=8, color="white", fontweight="bold")

plt.tight_layout(pad=1.2)
fig1.savefig(FIGS / "fig1.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print("  ✓ fig1.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — pp_mean vs state geometric compactness
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 2…")

# Compute state-level PP score from dissolved county geometry
def poly_pp(geom):
    import math
    area = geom.area
    peri = geom.length
    if peri == 0: return 0
    return 4 * math.pi * area / (peri ** 2)

state_pp = {}
for _, row in states_cont.iterrows():
    abbr = row["abbr"]
    state_pp[abbr] = poly_pp(row.geometry)

# Census regions
REGIONS = {
    "Northeast": ["CT","ME","MA","NH","NJ","NY","PA","RI","VT"],
    "Midwest":   ["IL","IN","IA","KS","MI","MN","MO","NE","OH","ND","SD","WI"],
    "South":     ["AL","AR","DE","FL","GA","KY","LA","MD","MS","NC","OK","SC","TN","TX","VA","WV"],
    "West":      ["AK","AZ","CA","CO","HI","ID","MT","NV","NM","OR","UT","WA","WY"],
}
abbr2region = {a: r for r, abbrs in REGIONS.items() for a in abbrs}
region_colors = {"Northeast": "#e41a1c", "Midwest": "#377eb8",
                 "South": "#ff7f00", "West": "#4daf4a"}

xs, ys, colors, labels = [], [], [], []
for abbr, stats in state_stats.items():
    sp = state_pp.get(abbr)
    if sp is None: continue
    xs.append(sp)
    ys.append(stats["pp_mean"])
    colors.append(region_colors.get(abbr2region.get(abbr, ""), GRAY))
    labels.append(abbr)

xs, ys = np.array(xs), np.array(ys)

fig2, ax = plt.subplots(figsize=(7, 5), facecolor="white")
ax.set_title("State geometric complexity predicts district compactness",
             fontsize=10, fontweight="bold", pad=8)

for r, c in region_colors.items():
    ax.scatter([], [], color=c, s=50, label=r, alpha=0.9)

ax.scatter(xs, ys, c=colors, s=60, alpha=0.85, zorder=3, edgecolors="white", linewidth=0.5)

# 1:1 reference line
lim = (0, max(xs.max(), ys.max()) * 1.08)
ax.plot(lim, lim, "--", color=GRAY, linewidth=1, alpha=0.7, label="1:1 reference")

# Fit line
m, b = np.polyfit(xs, ys, 1)
xfit = np.linspace(xs.min(), xs.max(), 100)
ax.plot(xfit, m * xfit + b, "-", color=DARK, linewidth=1.2, alpha=0.6)

# R²
r2 = np.corrcoef(xs, ys)[0,1]**2
ax.text(0.97, 0.08, f"$R^2 = {r2:.2f}$", transform=ax.transAxes,
        ha="right", fontsize=9, color=DARK)

# Label selected states
highlight = {"WV","RI","GA","LA","TN","NE","KS","MT","NV"}
for i, abbr in enumerate(labels):
    if abbr in highlight:
        ax.annotate(abbr, (xs[i], ys[i]),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=7, color=DARK)

ax.set_xlabel("State polygon Polsby-Popper score (geometric complexity)", fontsize=9)
ax.set_ylabel("Mean district Polsby-Popper score (pp_mean)", fontsize=9)
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.legend(fontsize=8, loc="upper left", framealpha=0.8)
ax.grid(True, alpha=0.2, linewidth=0.5)

plt.tight_layout()
fig2.savefig(FIGS / "fig2.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print("  ✓ fig2.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Competitive seats: grouped bar + margin KDE
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 3…")

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), facecolor="white",
                                  gridspec_kw={"width_ratios": [1, 1.2]})

# ── Panel a: grouped bar ──────────────────────────────────────────────────────
ax1.set_title("a   Competitive seat counts", loc="left",
              fontsize=9, fontweight="bold", pad=5)

cats = ["≤5% margin\n(highly competitive)", "≤10% margin\n(competitive)"]
bv   = [comp_blind_5, comp_blind_10]
ev   = [comp_enacted_5, comp_enacted_10]
xx   = np.arange(2)
wb   = 0.35

bars_b = ax1.bar(xx - wb/2, bv, wb, color=BLUE, alpha=0.85, label="Blind maps",   zorder=3)
bars_e = ax1.bar(xx + wb/2, ev, wb, color=RED,  alpha=0.85, label="Enacted maps", zorder=3)

for bar in list(bars_b) + list(bars_e):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             str(int(bar.get_height())),
             ha="center", va="bottom", fontsize=8.5, color=DARK, fontweight="bold")

# Delta arrows
for i, (bval, eval_) in enumerate(zip(bv, ev)):
    diff = bval - eval_
    pct  = diff / eval_ * 100
    top  = max(bval, eval_) + 4
    ax1.annotate("", xy=(xx[i]-wb/2, top+0.5), xytext=(xx[i]+wb/2, top+0.5),
                 arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.5))
    ax1.text(xx[i], top + 2.5, f"+{diff} (+{pct:.0f}%)",
             ha="center", fontsize=7.5, color=BLUE, fontweight="bold")

ax1.set_xticks(xx)
ax1.set_xticklabels(cats, fontsize=8.5)
ax1.set_ylabel("Number of seats", fontsize=8.5)
ax1.set_ylim(0, max(bv + ev) * 1.28)
ax1.legend(fontsize=8, loc="upper right", framealpha=0.75)
ax1.grid(axis="y", alpha=0.25, linewidth=0.5)

# ── Panel b: margin KDE ───────────────────────────────────────────────────────
ax2.set_title("b   Distribution of district competitiveness margins",
              loc="left", fontsize=9, fontweight="bold", pad=5)

for margins, color, label in [
    (blind_margins,   BLUE, "Blind maps"),
    (enacted_margins, RED,  "Enacted maps"),
]:
    kde = gaussian_kde(margins, bw_method=0.12)
    xg  = np.linspace(0, 0.55, 300)
    yg  = kde(xg)
    ax2.plot(xg, yg, color=color, lw=2, label=label)
    ax2.fill_between(xg, yg, alpha=0.12, color=color)

# Shade competitive zones
for thresh, alpha, label in [(0.05, 0.08, ""), (0.10, 0.05, "")]:
    ax2.axvspan(0, thresh, alpha=alpha, color=DARK, zorder=0)

ax2.axvline(0.05, color=GRAY, lw=0.8, linestyle=":", alpha=0.8)
ax2.axvline(0.10, color=GRAY, lw=0.8, linestyle="--", alpha=0.8)
ax2.text(0.05, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 8,
         "  5%", fontsize=7, color=GRAY, va="top")
ax2.text(0.10, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 8,
         "  10%", fontsize=7, color=GRAY, va="top")

ax2.set_xlabel("Victory margin (|Dem share − 0.5|)", fontsize=8.5)
ax2.set_ylabel("Density", fontsize=8.5)
ax2.set_xlim(0, 0.55)
ax2.legend(fontsize=8, framealpha=0.75)
ax2.grid(alpha=0.2, linewidth=0.5)

plt.tight_layout(pad=1.5)
fig3.savefig(FIGS / "fig3.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig3)
print("  ✓ fig3.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — State-level partisan seat deviation (horizontal bar chart)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 4…")

devs = {s: d for s, d in deviations.items() if d != 0}
devs_sorted = sorted(devs.items(), key=lambda x: x[1])

states_all_sorted = sorted(deviations.items(), key=lambda x: x[1])
names  = [s for s, _ in states_all_sorted]
values = [v for _, v in states_all_sorted]

colors_bar = [BLUE if v > 0 else RED if v < 0 else GRAY for v in values]
# Highlight large ones
edge_colors = ["#1a1a1a" if abs(v) >= 2 else "none" for v in values]

fig4, ax = plt.subplots(figsize=(7, 8), facecolor="white")
ax.set_title("State-level partisan seat deviation\n(blind maps − enacted 118th Congress)",
             fontsize=10, fontweight="bold", pad=8)

ypos = np.arange(len(names))
bars = ax.barh(ypos, values, color=colors_bar, edgecolor=edge_colors,
               linewidth=0.7, height=0.72, zorder=3)

# Value labels
for i, (v, bar) in enumerate(zip(values, bars)):
    if v == 0: continue
    xpos = v + (0.05 if v > 0 else -0.05)
    ha   = "left" if v > 0 else "right"
    ax.text(xpos, i, f"{'+' if v>0 else ''}{v}",
            va="center", ha=ha, fontsize=7.5, color=DARK, fontweight="bold" if abs(v)>=2 else "normal")

ax.set_yticks(ypos)
ax.set_yticklabels(names, fontsize=7.5)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Seat deviation  (positive = blind maps elect more Democrats)", fontsize=8.5)
ax.set_xlim(-4, 4)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.grid(axis="x", alpha=0.2, linewidth=0.5)

# Legend
leg_elems = [
    mpatches.Patch(color=BLUE, label="Blind maps elect more Democrats\n(enacted map may favor Republicans)"),
    mpatches.Patch(color=RED,  label="Enacted maps elect more Democrats\n(enacted map may favor Democrats)"),
    mpatches.Patch(color=GRAY, label="No deviation"),
]
ax.legend(handles=leg_elems, fontsize=7.5, loc="lower right", framealpha=0.8)

# Annotate key states
annotations = {
    "IL": "D gerrymander", "WI": "R gerrymander",
    "TX": "R gerrymander", "NY": "Court-drawn\n(conservative)",
    "PA": "Court-drawn\n(D-leaning)"
}
for abbr, note in annotations.items():
    if abbr in names:
        i = names.index(abbr)
        v = values[i]
        ax.annotate(note,
            xy=(v, i), xytext=(v + (0.35 if v>0 else -0.35), i),
            fontsize=6.5, va="center",
            ha="left" if v > 0 else "right",
            color="#555",
            arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6)
        )

plt.tight_layout()
fig4.savefig(FIGS / "fig4.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig4)
print("  ✓ fig4.png")

print(f"\nAll figures saved to {FIGS}/")
