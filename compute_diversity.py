"""
compute_diversity.py

Computes racial diversity (Herfindahl index) for:
  1. Blind/geography-only proposed districts (best_map_compact.gpkg)
  2. Enacted 118th Congress districts (tl_2022_XX_cd118.shp)

Uses 2020 Census PL94-171 redistricting data at the census-tract level.
"""

import os
import sys
import json
import zipfile
import logging
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
DATA = BASE / "data"
DOCS_MAPS = BASE / "docs" / "maps"
DOCS_MAPS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# State configs
# ---------------------------------------------------------------------------
sys.path.insert(0, str(BASE))
from state_configs import ALL_STATES, STATES_BY_ABBR

# At-large states (k=1, whole state = 1 district)
AT_LARGE = {"AK", "DE", "ND", "SD", "VT", "WY"}

# The 'maryland' directory holds PL94-171 data; 'md' holds the district maps
# We unify: use abbr.lower() for district data, but 'maryland' for PL cache
def pl_dir(abbr: str) -> Path:
    """Return PL94-171 cache directory for a state."""
    if abbr.upper() == "MD":
        return DATA / "maryland" / "census_pl94171"
    return DATA / abbr.lower() / "census_pl94171"

def state_dir(abbr: str) -> Path:
    return DATA / abbr.lower()

# ---------------------------------------------------------------------------
# State name → URL-safe name (underscores, title-case)
# ---------------------------------------------------------------------------
STATE_URL_NAMES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New_Hampshire",
    "NJ": "New_Jersey",
    "NM": "New_Mexico",
    "NY": "New_York",
    "NC": "North_Carolina",
    "ND": "North_Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode_Island",
    "SC": "South_Carolina",
    "SD": "South_Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West_Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "redistrict-diversity/1.0 (research)"


def download_file(url: str, dest: Path) -> bool:
    """Download url → dest. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = SESSION.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        return True
    except Exception as e:
        log.warning("  Download failed %s: %s", url, e)
        if dest.exists():
            dest.unlink()
        return False


def ensure_pl94171(abbr: str) -> bool:
    """Download and extract PL94-171 if not already cached. Returns True if available."""
    cache = pl_dir(abbr)
    geo_file = cache / f"{abbr.lower()}geo2020.pl"
    seg_file = cache / f"{abbr.lower()}000012020.pl"

    if geo_file.exists() and seg_file.exists():
        return True

    url_name = STATE_URL_NAMES[abbr.upper()]
    abbr_lower = abbr.lower()
    zip_url = (
        f"https://www2.census.gov/programs-surveys/decennial/2020/data/"
        f"01-Redistricting_File--PL_94-171/{url_name}/{abbr_lower}2020.pl.zip"
    )
    zip_path = cache / f"{abbr_lower}2020.pl.zip"
    cache.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        log.info("  Downloading PL94-171 for %s ...", abbr)
        ok = download_file(zip_url, zip_path)
        if not ok:
            return False

    log.info("  Extracting PL94-171 for %s ...", abbr)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache)
        return geo_file.exists() and seg_file.exists()
    except Exception as e:
        log.warning("  Extraction failed for %s: %s", abbr, e)
        return False


def ensure_tract_shp(abbr: str, fips: str) -> bool:
    """Download TIGER 2020 tract shapefile if not already present."""
    tract_dir = state_dir(abbr) / "census_tracts"
    shp_file = tract_dir / f"tl_2020_{fips}_tract.shp"

    if shp_file.exists():
        return True

    tract_dir.mkdir(parents=True, exist_ok=True)
    zip_url = f"https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_{fips}_tract.zip"
    zip_path = tract_dir / f"tl_2020_{fips}_tract.zip"

    if not zip_path.exists():
        log.info("  Downloading tract shapefile for %s (%s) ...", abbr, fips)
        ok = download_file(zip_url, zip_path)
        if not ok:
            return False

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tract_dir)
        return shp_file.exists()
    except Exception as e:
        log.warning("  Tract shp extraction failed for %s: %s", abbr, e)
        return False


# ---------------------------------------------------------------------------
# PL94-171 parsing
# ---------------------------------------------------------------------------
def parse_pl94171(abbr: str) -> pd.DataFrame | None:
    """
    Parse PL94-171 files and return a DataFrame of census-tract-level
    race group counts with columns:
        geoid, total, hispanic, nh_white, nh_black, nh_aian,
        nh_asian, nh_nhpi, nh_other
    Returns None if files not available.
    """
    cache = pl_dir(abbr)
    geo_path = cache / f"{abbr.lower()}geo2020.pl"
    seg_path = cache / f"{abbr.lower()}000012020.pl"

    if not geo_path.exists() or not seg_path.exists():
        return None

    # ---- Parse geo file: extract SUMLEV=140, get LOGRECNO → GEOID ----
    geo_rows = []
    with open(geo_path, "r", encoding="latin-1") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("|")
            if len(fields) < 10:
                continue
            sumlev = fields[2].strip()
            if sumlev != "140":
                continue
            logrecno = fields[7].strip()
            geoid_full = fields[8].strip()  # e.g. "1400000US24001000100"
            # Extract 11-digit tract GEOID: last 11 chars of the numeric part
            # geoid_full format: "1400000US{state2}{county3}{tract6}"
            if "US" in geoid_full:
                geoid = geoid_full.split("US")[1]  # "24001000100"
            else:
                geoid = geoid_full[-11:]
            geo_rows.append({"logrecno": logrecno, "geoid": geoid})

    if not geo_rows:
        log.warning("  No SUMLEV=140 rows found for %s", abbr)
        return None

    geo_df = pd.DataFrame(geo_rows)
    geo_df["logrecno"] = geo_df["logrecno"].str.lstrip("0").str.zfill(7)

    # ---- Parse segment 1 file: extract P2 fields ----
    # Fields (1-indexed):
    #   5: LOGRECNO
    #   77: P2_001N total
    #   78: P2_002N hispanic
    #   81: P2_005N nh_white
    #   82: P2_006N nh_black
    #   83: P2_007N nh_aian
    #   84: P2_008N nh_asian
    #   85: P2_009N nh_nhpi
    #   86: P2_010N nh_other (Other alone)
    #   87: P2_011N nh_two+ (Two or more)
    # We combine nh_other + nh_two+ into a single "nh_other" group per spec

    seg_rows = []
    with open(seg_path, "r", encoding="latin-1") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("|")
            if len(fields) < 87:
                continue
            logrecno = fields[4].strip().lstrip("0").zfill(7)
            try:
                total    = int(fields[76])
                hispanic = int(fields[77])
                nh_white = int(fields[80])
                nh_black = int(fields[81])
                nh_aian  = int(fields[82])
                nh_asian = int(fields[83])
                nh_nhpi  = int(fields[84])
                nh_other = int(fields[85])
                nh_two   = int(fields[86])
            except (ValueError, IndexError):
                continue
            seg_rows.append({
                "logrecno": logrecno,
                "total":    total,
                "hispanic": hispanic,
                "nh_white": nh_white,
                "nh_black": nh_black,
                "nh_aian":  nh_aian,
                "nh_asian": nh_asian,
                "nh_nhpi":  nh_nhpi,
                "nh_other": nh_other + nh_two,
            })

    seg_df = pd.DataFrame(seg_rows)

    merged = geo_df.merge(seg_df, on="logrecno", how="inner")
    log.info("  %s: %d census tracts parsed", abbr, len(merged))
    return merged


# ---------------------------------------------------------------------------
# Diversity computation
# ---------------------------------------------------------------------------
RACE_COLS = ["hispanic", "nh_white", "nh_black", "nh_aian", "nh_asian", "nh_nhpi", "nh_other"]


def herfindahl(counts: np.ndarray) -> float:
    """Compute Herfindahl diversity index for an array of group counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return float(1.0 - (p ** 2).sum())


def tracts_to_district_diversity(
    tracts_gdf: gpd.GeoDataFrame,
    tract_pop: pd.DataFrame,
    districts_gdf: gpd.GeoDataFrame,
    district_id_col: str,
) -> list[dict]:
    """
    Spatially join tracts → districts, then aggregate race counts
    weighted by total population.

    Returns list of {district: label, diversity: float}.
    """
    # Use tract centroids for spatial join (faster, avoids edge issues)
    tracts_pts = tracts_gdf.copy()
    tracts_pts.geometry = tracts_gdf.geometry.centroid

    # Reproject to match districts CRS
    if tracts_pts.crs != districts_gdf.crs:
        tracts_pts = tracts_pts.to_crs(districts_gdf.crs)

    # Spatial join
    joined = gpd.sjoin(
        tracts_pts[["geoid", "geometry"]],
        districts_gdf[[district_id_col, "geometry"]],
        how="left",
        predicate="within",
    )

    # Merge with population data
    joined = joined.merge(tract_pop, on="geoid", how="left")

    results = []
    for dist_label, grp in joined.groupby(district_id_col):
        counts = grp[RACE_COLS].fillna(0).sum().values.astype(float)
        d = herfindahl(counts)
        results.append({"district": dist_label, "diversity": round(d, 6)})

    return results


# ---------------------------------------------------------------------------
# Per-state processing
# ---------------------------------------------------------------------------
def process_state(cfg) -> dict:
    """
    Process one state. Returns dict with keys:
        abbr, blind_districts, enacted_districts, error
    """
    abbr = cfg.abbr
    fips = cfg.fips
    result = {"abbr": abbr, "blind_districts": [], "enacted_districts": [], "error": None}

    # 1. Ensure PL94-171 data
    ok = ensure_pl94171(abbr)
    if not ok:
        result["error"] = f"PL94-171 unavailable for {abbr}"
        log.warning("  Skipping %s: PL94-171 not available", abbr)
        return result

    # 2. Parse census data
    tract_pop = parse_pl94171(abbr)
    if tract_pop is None or len(tract_pop) == 0:
        result["error"] = f"No tract data for {abbr}"
        return result

    # 3. Ensure tract shapefile
    ok = ensure_tract_shp(abbr, fips)
    if not ok:
        # Try from the at-large state path
        result["error"] = f"Tract shapefile unavailable for {abbr}"
        log.warning("  Skipping %s: tract shapefile not available", abbr)
        return result

    # 4. Load tract geometries
    tract_shp = state_dir(abbr) / "census_tracts" / f"tl_2020_{fips}_tract.shp"
    try:
        tracts_gdf = gpd.read_file(tract_shp)
    except Exception as e:
        result["error"] = str(e)
        return result

    # Build GEOID column (11-digit: state2+county3+tract6)
    if "GEOID" in tracts_gdf.columns:
        tracts_gdf["geoid"] = tracts_gdf["GEOID"].str.strip()
    elif "TRACTCE" in tracts_gdf.columns:
        tracts_gdf["geoid"] = (
            tracts_gdf["STATEFP"].str.zfill(2)
            + tracts_gdf["COUNTYFP"].str.zfill(3)
            + tracts_gdf["TRACTCE"].str.zfill(6)
        )
    else:
        result["error"] = "No GEOID column in tract shapefile"
        return result

    # Merge pop counts into tracts
    tracts_gdf = tracts_gdf.merge(
        tract_pop[["geoid"] + RACE_COLS + ["total"]],
        on="geoid",
        how="left",
    )
    tracts_gdf[RACE_COLS + ["total"]] = tracts_gdf[RACE_COLS + ["total"]].fillna(0)

    # ----------------------------------------------------------------
    # 5. At-large states: whole state is 1 district
    # ----------------------------------------------------------------
    if cfg.k == 1:
        counts = tracts_gdf[RACE_COLS].sum().values.astype(float)
        d = herfindahl(counts)
        entry = {"district": 1, "diversity": round(d, 6)}
        result["blind_districts"] = [entry]
        result["enacted_districts"] = [entry]
        log.info("  %s (at-large): diversity=%.4f", abbr, d)
        return result

    # ----------------------------------------------------------------
    # 6. Blind districts
    # ----------------------------------------------------------------
    blind_path = state_dir(abbr) / "final" / "best_map_compact.gpkg"
    if blind_path.exists():
        try:
            blind_gdf = gpd.read_file(blind_path)
            # Find district ID column
            dist_col = None
            for col in ["district_id", "district", "DISTRICT", "dist_id", "id", "CD"]:
                if col in blind_gdf.columns:
                    dist_col = col
                    break
            if dist_col is None:
                # Use first non-geometry column
                dist_col = [c for c in blind_gdf.columns if c != "geometry"][0]

            blind_results = tracts_to_district_diversity(
                tracts_gdf, tract_pop, blind_gdf, dist_col
            )
            result["blind_districts"] = blind_results
            log.info("  %s blind: %d districts", abbr, len(blind_results))
        except Exception as e:
            log.warning("  %s blind map error: %s", abbr, e)
    else:
        log.warning("  %s: no blind map found at %s", abbr, blind_path)

    # ----------------------------------------------------------------
    # 7. Enacted 118th Congress districts
    # ----------------------------------------------------------------
    enacted_path = state_dir(abbr) / "current_districts" / f"tl_2022_{fips}_cd118.shp"
    if enacted_path.exists():
        try:
            enacted_gdf = gpd.read_file(enacted_path)
            # Find district ID col
            dist_col = None
            for col in ["CD118FP", "GEOID20", "DISTRICT", "district", "CD", "GEOID"]:
                if col in enacted_gdf.columns:
                    dist_col = col
                    break
            if dist_col is None:
                dist_col = [c for c in enacted_gdf.columns if c != "geometry"][0]

            enacted_results = tracts_to_district_diversity(
                tracts_gdf, tract_pop, enacted_gdf, dist_col
            )
            result["enacted_districts"] = enacted_results
            log.info("  %s enacted: %d districts", abbr, len(enacted_results))
        except Exception as e:
            log.warning("  %s enacted map error: %s", abbr, e)
    else:
        log.info("  %s: no enacted cd118 shapefile", abbr)

    return result


# ---------------------------------------------------------------------------
# Histogram plot
# ---------------------------------------------------------------------------
def make_histogram(blind_scores: list[float], enacted_scores: list[float], out_path: Path):
    BG = "#0d1117"
    FG = "white"
    BLUE = "#4c9be8"
    RED  = "#e85c4c"

    bins = np.arange(0.0, 1.051, 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig.patch.set_facecolor(BG)

    def _plot_panel(ax, scores, color, title):
        arr = np.array(scores)
        ax.set_facecolor(BG)
        n, _, patches = ax.hist(arr, bins=bins, color=color, edgecolor=BG, linewidth=0.5)
        med = float(np.median(arr))
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        ax.axvline(med, color=FG, linestyle="--", linewidth=1.4, label=f"Median={med:.3f}")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Diversity Index (Herfindahl)", color=FG, fontsize=11)
        ax.set_ylabel("Number of Districts", color=FG, fontsize=11)
        ax.set_title(title, color=FG, fontsize=12, pad=10)
        subtitle = f"mean={mean:.3f} ± {std:.3f},  N={len(arr)}"
        ax.text(0.5, -0.14, subtitle, transform=ax.transAxes,
                ha="center", va="top", color="#aaaaaa", fontsize=9)
        ax.tick_params(colors=FG, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(frameon=False, labelcolor=FG, fontsize=9)

    _plot_panel(axes[0], blind_scores, BLUE, "Blind Algorithm (Geography Only)")
    _plot_panel(axes[1], enacted_scores, RED, "Enacted 118th Congress Districts")

    fig.suptitle("Racial Diversity of Congressional Districts", color=FG, fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Saved histogram → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Build list of all 50 states to process (skip 'maryland' duplicate)
    states_to_process = [
        cfg for cfg in ALL_STATES.values()
        # 'maryland' dir is used only for PL cache; 'md' is the canonical state dir
    ]

    log.info("Processing %d states ...", len(states_to_process))

    # Download all PL94-171 in parallel, then process sequentially to avoid memory spikes
    log.info("=== Phase 1: Parallel downloads ===")

    def download_state(cfg):
        abbr = cfg.abbr
        fips = cfg.fips
        pl_ok = ensure_pl94171(abbr)
        tr_ok = ensure_tract_shp(abbr, fips)
        return abbr, pl_ok, tr_ok

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_state, cfg): cfg for cfg in states_to_process}
        download_status = {}
        for fut in as_completed(futures):
            abbr, pl_ok, tr_ok = fut.result()
            download_status[abbr] = (pl_ok, tr_ok)
            log.info("  Download done: %s  PL=%s  TRACT=%s", abbr, pl_ok, tr_ok)

    log.info("=== Phase 2: Compute diversity ===")

    all_blind = []
    all_enacted = []
    errors = []

    for cfg in states_to_process:
        abbr = cfg.abbr
        log.info("--- %s ---", abbr)
        state_result = process_state(cfg)

        if state_result["error"]:
            errors.append(f"{abbr}: {state_result['error']}")

        for entry in state_result["blind_districts"]:
            all_blind.append({"state": abbr, **entry})

        for entry in state_result["enacted_districts"]:
            all_enacted.append({"state": abbr, **entry})

    # ----------------------------------------------------------------
    # Stats
    # ----------------------------------------------------------------
    def stats(scores: list[float]) -> dict:
        arr = np.array(scores)
        if len(arr) == 0:
            return {"mean": None, "median": None, "std": None, "n": 0}
        return {
            "mean":   round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std":    round(float(np.std(arr)), 4),
            "n":      int(len(arr)),
        }

    blind_scores   = [e["diversity"] for e in all_blind]
    enacted_scores = [e["diversity"] for e in all_enacted]

    output = {
        "blind":          all_blind,
        "enacted":        all_enacted,
        "blind_stats":    stats(blind_scores),
        "enacted_stats":  stats(enacted_scores),
        "errors":         errors,
    }

    out_json = DATA / "diversity_results.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved JSON → %s", out_json)

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    if blind_scores and enacted_scores:
        make_histogram(blind_scores, enacted_scores, DOCS_MAPS / "diversity_comparison.png")
    else:
        log.warning("Not enough data to plot (blind=%d, enacted=%d)",
                    len(blind_scores), len(enacted_scores))

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DIVERSITY RESULTS SUMMARY")
    print("=" * 60)
    bs = output["blind_stats"]
    es = output["enacted_stats"]
    print(f"Blind (Geography Only):  mean={bs['mean']:.4f}  median={bs['median']:.4f}"
          f"  std={bs['std']:.4f}  N={bs['n']}")
    print(f"Enacted 118th Congress:  mean={es['mean']:.4f}  median={es['median']:.4f}"
          f"  std={es['std']:.4f}  N={es['n']}")
    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(f"  {e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
