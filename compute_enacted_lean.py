"""
compute_enacted_lean.py

Post-hoc partisan lean for the enacted 118th Congress districts using the
same VEST 2020 + uniform 2024 swing method as compute_lean_2024.py.

Output: data/enacted_lean_2024.json  — same schema as lean_2024.json
"""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np

from compute_lean_2024 import RESULTS_2024, VEST_DIR

DATA_ROOT = Path("data")

# Map state abbr → enacted shapefile
def enacted_path(abbr: str) -> Path | None:
    d = DATA_ROOT / abbr.lower() / "current_districts"
    if not d.exists():
        return None
    shps = list(d.glob("tl_*.shp"))
    return shps[0] if shps else None


def vest_path(abbr: str) -> Path | None:
    p = VEST_DIR / abbr.lower() / f"{abbr.lower()}_2020.shp"
    return p if p.exists() else None


def compute_enacted_lean(abbr: str) -> dict | None:
    enacted_shp = enacted_path(abbr)
    if enacted_shp is None:
        print(f"  [{abbr}] No enacted shapefile found")
        return None

    vest_shp = vest_path(abbr)
    if vest_shp is None:
        print(f"  [{abbr}] No VEST data found")
        return None

    if abbr not in RESULTS_2024:
        print(f"  [{abbr}] No 2024 results data")
        return None

    print(f"  [{abbr}] Loading enacted districts…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enacted = gpd.read_file(enacted_shp)

    # Remove non-voting delegate seats (CD118FP == 'ZZ')
    id_col = "CD118FP"
    if id_col in enacted.columns:
        enacted = enacted[enacted[id_col] != "ZZ"].copy()
    else:
        print(f"  [{abbr}] No CD118FP column, skipping")
        return None

    print(f"  [{abbr}] Loading VEST 2020…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vest = gpd.read_file(vest_shp)

    if "G20PREDBID" not in vest.columns or "G20PRERTRU" not in vest.columns:
        print(f"  [{abbr}] Missing Biden/Trump columns")
        return None

    vest = vest[["G20PREDBID", "G20PRERTRU", "geometry"]].copy()
    vest["bid"] = vest["G20PREDBID"].fillna(0).astype(float)
    vest["tru"] = vest["G20PRERTRU"].fillna(0).astype(float)
    vest["total_2p"] = vest["bid"] + vest["tru"]

    state_bid_2020 = vest["bid"].sum()
    state_total_2020 = vest["total_2p"].sum()
    biden_state_2p = state_bid_2020 / state_total_2020 if state_total_2020 > 0 else 0.5
    harris_state_2p = RESULTS_2024[abbr]
    swing = harris_state_2p - biden_state_2p

    vest["biden_pct"] = np.where(vest["total_2p"] > 0, vest["bid"] / vest["total_2p"], 0.5)
    vest["harris_est"] = np.clip(vest["biden_pct"] + swing, 0.0, 1.0)

    # Reproject VEST to enacted CRS
    vest = vest.to_crs(enacted.crs)

    # Centroid join
    cents = vest.copy()
    cents["geometry"] = vest.geometry.centroid

    joined = gpd.sjoin(
        cents[["harris_est", "total_2p", "geometry"]],
        enacted[["CD118FP", "geometry"]],
        how="left",
        predicate="within",
    )

    result = {}
    d_count = 0
    r_count = 0

    for cd_fp, grp in joined.groupby("CD118FP"):
        try:
            did = int(cd_fp)
        except (ValueError, TypeError):
            continue
        w = grp["total_2p"].values
        h = grp["harris_est"].values
        total_w = w.sum()
        harris_pct = float((h * w).sum() / total_w) if total_w > 0 else 0.5

        margin = round(abs(harris_pct - 0.5) * 200)
        lean = "D" if harris_pct >= 0.5 else "R"
        label = f"{'D' if lean == 'D' else 'R'}+{margin}"

        if lean == "D":
            d_count += 1
        else:
            r_count += 1

        result[did] = {
            "harris_pct": round(harris_pct, 4),
            "lean": lean,
            "margin": margin,
            "label": label,
        }

    print(f"  [{abbr}] D={d_count}  R={r_count}  ({len(result)} districts)")
    return {"districts": result, "D": d_count, "R": r_count}


def main():
    # Include all states that have enacted shapefiles AND 2024 results
    states = sorted(
        abbr for abbr in RESULTS_2024
        if enacted_path(abbr) is not None
    )
    # Also handle legacy 'maryland' path
    if "MD" in RESULTS_2024 and enacted_path("MD") is None:
        md_leg = DATA_ROOT / "maryland" / "current_districts"
        if md_leg.exists():
            states = sorted(set(states) | {"MD"})

    output = {}
    total_d = 0
    total_r = 0

    for abbr in states:
        print(f"\n[{abbr}]")
        res = compute_enacted_lean(abbr)
        if res:
            output[abbr] = res
            total_d += res["D"]
            total_r += res["R"]

    output["totals"] = {"D": total_d, "R": total_r, "total": total_d + total_r}

    # Compute margin breakdowns
    for thresh, key in [(2, "within_2pct"), (5, "within_5pct"), (8, "within_8pct")]:
        d_cnt, r_cnt = 0, 0
        for abbr, v in output.items():
            if not isinstance(v, dict) or "districts" not in v:
                continue
            for did, d in v["districts"].items():
                if d["margin"] <= thresh:
                    if d["lean"] == "D":
                        d_cnt += 1
                    else:
                        r_cnt += 1
        output[f"margins_{key}"] = {"D": d_cnt, "R": r_cnt, "total": d_cnt + r_cnt}

    out_path = DATA_ROOT / "enacted_lean_2024.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print(f"ENACTED TOTALS across {len(output)-3} states:")
    print(f"  Democrat-leaning  : {total_d} districts")
    print(f"  Republican-leaning: {total_r} districts")
    print(f"  Total             : {total_d + total_r} districts")
    for key in ["margins_within_2pct", "margins_within_5pct", "margins_within_8pct"]:
        m = output.get(key, {})
        print(f"  {key}: D={m.get('D',0)} R={m.get('R',0)} total={m.get('total',0)}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
