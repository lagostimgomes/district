"""Render blind-vs-enacted side-by-side comparison maps for all 44 multi-district states."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from state_configs import ALL_STATES, STATES_BY_ABBR

DATA_ROOT  = Path("data")
COUNTY_SHP = Path("data/maryland/counties/tl_2020_us_county.shp")
counties_all = gpd.read_file(COUNTY_SHP)

PALETTE = ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F","#EDC948",
           "#B07AA1","#FF9DA7","#9C755F","#BAB0AC","#D37295","#FABFD2",
           "#B6992D","#499894","#86BCB6","#E15759"]
BG, SURFACE = "#0d1117", "#161b22"

def color_by_id(district_id):
    return PALETTE[int(district_id) % len(PALETTE)]

def render_comparison(cfg):
    abbr = cfg.abbr.lower()
    proposed_gpkg = DATA_ROOT / abbr / "final" / "best_map_compact.gpkg"
    enacted_pat   = list((DATA_ROOT / abbr / "current_districts").glob("tl_*.shp")) if \
                    (DATA_ROOT / abbr / "current_districts").exists() else []
    out_path = DATA_ROOT / abbr / "final" / "map_vs_enacted.png"

    if not proposed_gpkg.exists():
        return False, "no proposed gpkg"

    try:
        proposed = gpd.read_file(proposed_gpkg)
        crs = proposed.crs

        # State boundary for county overlay
        fips2 = cfg.fips.zfill(2)
        state_counties = counties_all[counties_all["STATEFP"] == fips2].to_crs(crs)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG)
        fig.suptitle(f"{cfg.name}", fontsize=16, fontweight="bold", color="#e6edf3", y=0.99)

        for ax, gdf, title, id_col in [
            (axes[0], proposed, "Blind Algorithm\n(Geography Only)", "district_id"),
            (axes[1], None,     "Enacted 118th Congress", "CD118FP"),
        ]:
            ax.set_facecolor(SURFACE)
            ax.set_axis_off()
            ax.set_title(title, color="#e6edf3", fontsize=11, pad=6)

            if ax is axes[1]:
                if not enacted_pat:
                    ax.text(0.5, 0.5, "Enacted map\nnot available",
                            ha="center", va="center", transform=ax.transAxes,
                            color="#555", fontsize=12)
                    continue
                try:
                    gdf = gpd.read_file(enacted_pat[0]).to_crs(crs)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Load error:\n{e}", ha="center", va="center",
                            transform=ax.transAxes, color="#555", fontsize=9)
                    continue

            colors = [color_by_id(row[id_col]) for _, row in gdf.iterrows()]
            gdf.plot(ax=ax, color=colors, edgecolor="#ffffff", linewidth=0.3)
            if not state_counties.empty:
                state_counties.boundary.plot(ax=ax, edgecolor="#666666",
                                             linewidth=0.25, alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        return True, str(out_path)
    except Exception as e:
        plt.close("all")
        return False, str(e)

ok, fail = 0, 0
for cfg in sorted(ALL_STATES.values(), key=lambda c: c.abbr):
    if cfg.k == 1: continue
    print(f"  [{cfg.abbr}]", end=" ", flush=True)
    success, msg = render_comparison(cfg)
    if success:
        print("OK")
        ok += 1
    else:
        print(f"FAIL: {msg}")
        fail += 1

print(f"\nDone: {ok} OK, {fail} failed")
