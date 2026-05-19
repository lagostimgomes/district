"""
compute_lean_2024.py

Post-hoc partisan lean for completed states using VEST 2020 precinct data
adjusted to 2024 presidential results via a uniform state-level swing.

Method
------
1. Load VEST 2020 precinct shapefile for the state.
2. Compute Biden 2020 two-party share per precinct.
3. Compute state-level 2020 and 2024 two-party Harris/Biden shares.
4. Apply uniform swing: harris_est(precinct) = biden(precinct) + swing_state
5. Spatially join precinct centroids → district polygons.
6. Sum votes per district → Harris two-party share per district.
7. District > 50% = D, else R.

2024 certified presidential results used (AP / state canvass):
  MD  Harris 65.2%  Trump 32.6%  → two-party Harris 66.6%
  CA  Harris 59.3%  Trump 38.5%  → two-party Harris 60.6%
  NY  Harris 55.5%  Trump 42.6%  → two-party Harris 56.5%
  TX  Harris 43.0%  Trump 55.3%  → two-party Harris 43.7%
  FL  Harris 43.1%  Trump 55.2%  → two-party Harris 43.8%

Output: data/lean_2024.json
  {
    "MD": {"districts": {0: {"harris_pct": 0.87, "lean": "D", "margin": 74}}, "D": 7, "R": 1},
    ...
    "totals": {"D": ..., "R": ...}
  }
"""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np

# ---------------------------------------------------------------------------
# 2024 certified two-party Harris share by state
# ---------------------------------------------------------------------------

# Two-party Harris share from 2024 certified presidential results (AP / state canvasses)
RESULTS_2024 = {
    "AL": 0.326, "AR": 0.323, "AZ": 0.471, "CA": 0.606, "CO": 0.564,
    "CT": 0.574, "FL": 0.434, "GA": 0.489, "HI": 0.710, "IA": 0.419,
    "ID": 0.284, "IL": 0.544, "IN": 0.367, "KS": 0.352, "KY": 0.312,
    "LA": 0.341, "MA": 0.637, "MD": 0.666, "ME": 0.540, "MI": 0.493,
    "MN": 0.533, "MO": 0.397, "MS": 0.367, "MT": 0.392, "NC": 0.483,
    "NE": 0.366, "NH": 0.518, "NJ": 0.544, "NM": 0.554, "NV": 0.482,
    "NY": 0.565, "OH": 0.439, "OK": 0.322, "OR": 0.594, "PA": 0.491,
    "RI": 0.597, "SC": 0.411, "TN": 0.331, "TX": 0.432, "UT": 0.392,
    "VA": 0.548, "WA": 0.601, "WI": 0.494, "WV": 0.248,
}

VEST_DIR   = Path("data/vest")
DATA_ROOT  = Path("data")


def vest_path(abbr: str) -> Path | None:
    p = VEST_DIR / abbr.lower() / f"{abbr.lower()}_2020.shp"
    return p if p.exists() else None


def compute_state_lean(abbr: str) -> dict | None:
    shp = vest_path(abbr)
    if shp is None:
        print(f"  [{abbr}] No VEST data found")
        return None

    gpkg = DATA_ROOT / abbr.lower() / "final" / "best_map_compact.gpkg"
    if not gpkg.exists():
        print(f"  [{abbr}] No final map found")
        return None

    print(f"  [{abbr}] Loading VEST 2020…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vest = gpd.read_file(shp)

    if "G20PREDBID" not in vest.columns or "G20PRERTRU" not in vest.columns:
        print(f"  [{abbr}] Missing Biden/Trump columns")
        return None

    vest = vest[["G20PREDBID", "G20PRERTRU", "geometry"]].copy()
    vest["bid"] = vest["G20PREDBID"].fillna(0).astype(float)
    vest["tru"] = vest["G20PRERTRU"].fillna(0).astype(float)
    vest["total_2p"] = vest["bid"] + vest["tru"]

    # State-level 2020 Biden two-party share
    state_bid_2020 = vest["bid"].sum()
    state_total_2020 = vest["total_2p"].sum()
    biden_state_2p = state_bid_2020 / state_total_2020 if state_total_2020 > 0 else 0.5

    # 2024 Harris two-party share
    harris_state_2p = RESULTS_2024[abbr]

    # Uniform swing
    swing = harris_state_2p - biden_state_2p
    print(f"  [{abbr}] Biden 2020 two-party: {biden_state_2p:.3f} → Harris 2024: {harris_state_2p:.3f}  (swing {swing:+.3f})")

    # Per-precinct 2020 Biden two-party share
    vest["biden_pct"] = np.where(
        vest["total_2p"] > 0,
        vest["bid"] / vest["total_2p"],
        0.5,
    )
    # Adjusted Harris estimate (clamped 0-1)
    vest["harris_est"] = np.clip(vest["biden_pct"] + swing, 0.0, 1.0)

    print(f"  [{abbr}] Loading district map…")
    districts = gpd.read_file(gpkg)

    # Align CRS
    vest = vest.to_crs(districts.crs)

    # Centroid join
    cents = vest.copy()
    cents["geometry"] = vest.geometry.centroid

    joined = gpd.sjoin(
        cents[["harris_est", "total_2p", "geometry"]],
        districts[["district_id", "geometry"]],
        how="left",
        predicate="within",
    )

    result = {}
    d_count = 0
    r_count = 0

    for did, grp in joined.groupby("district_id"):
        # Weight Harris estimate by two-party vote total
        w = grp["total_2p"].values
        h = grp["harris_est"].values
        total_w = w.sum()
        harris_pct = float((h * w).sum() / total_w) if total_w > 0 else 0.5

        margin = round(abs(harris_pct - 0.5) * 200)
        lean = "D" if harris_pct >= 0.5 else "R"
        label = f"{'D' if lean=='D' else 'R'}+{margin}"

        if lean == "D":
            d_count += 1
        else:
            r_count += 1

        result[int(did)] = {
            "harris_pct": round(harris_pct, 4),
            "lean": lean,
            "margin": margin,
            "label": label,
        }

    print(f"  [{abbr}] D={d_count}  R={r_count}")
    return {"districts": result, "D": d_count, "R": r_count}


def main():
    states = sorted(RESULTS_2024.keys())
    output = {}
    total_d = 0
    total_r = 0

    for abbr in states:
        print(f"\n[{abbr}]")
        res = compute_state_lean(abbr)
        if res:
            output[abbr] = res
            total_d += res["D"]
            total_r += res["R"]

    # At-large states (k=1): single district = whole state, lean from RESULTS_2024 directly.
    AT_LARGE = {"AK": 0.436, "DE": 0.583, "ND": 0.321, "SD": 0.361, "VT": 0.676, "WY": 0.233}
    for abbr, harris_pct in AT_LARGE.items():
        margin = round(abs(harris_pct - 0.5) * 200)
        lean = "D" if harris_pct >= 0.5 else "R"
        output[abbr] = {
            "districts": {0: {"harris_pct": harris_pct, "lean": lean, "margin": margin,
                               "label": f"{'D' if lean=='D' else 'R'}+{margin}"}},
            "D": 1 if lean == "D" else 0,
            "R": 1 if lean == "R" else 0,
        }
        if lean == "D":
            total_d += 1
        else:
            total_r += 1

    output["totals"] = {"D": total_d, "R": total_r, "total": total_d + total_r}

    # Margin breakdowns — 2 / 5 / 8 percent
    margins = {}
    for thresh, key in [(2, "within_2pct"), (5, "within_5pct"), (8, "within_8pct")]:
        d_cnt, r_cnt, d_states, r_states = 0, 0, set(), set()
        for abbr, v in output.items():
            if abbr in ("totals", "margins") or "districts" not in v:
                continue
            for did, d in v["districts"].items():
                if d["margin"] <= thresh:
                    if d["lean"] == "D":
                        d_cnt += 1
                        d_states.add(abbr)
                    else:
                        r_cnt += 1
                        r_states.add(abbr)
        margins[key] = {
            "D": d_cnt, "R": r_cnt, "total": d_cnt + r_cnt,
            "D_states": sorted(d_states), "R_states": sorted(r_states),
        }
    output["margins"] = margins

    out_path = DATA_ROOT / "lean_2024.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print(f"TOTALS across {len(output)-2} states:")
    print(f"  Democrat-leaning  : {total_d} districts")
    print(f"  Republican-leaning: {total_r} districts")
    print(f"  Total             : {total_d + total_r} districts")
    for key, m in margins.items():
        print(f"  {key}: D={m['D']} R={m['R']} total={m['total']}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
