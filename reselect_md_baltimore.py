"""
reselect_md_baltimore.py

Re-select Maryland maps from the 100k ensemble with a Baltimore City
constraint: Baltimore City may not be split across more than MAX_BALT_DISTRICTS
congressional districts.

Reads:  data/maryland/ensemble/  (100k-plan ensemble)
Writes: data/md/final/           (replaces 2k-based selection)
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from state_configs import STATES_BY_ABBR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_BALT_DISTRICTS = 2          # hard ceiling on Baltimore City district splits
PP_MIN_THRESHOLD   = 0.10
POP_DEV_MAX_THRESHOLD = 0.005
SAMPLE_SIZE = 5_000
RANDOM_SEED = 42

ENSEMBLE_DIR = Path("data/maryland/ensemble")
GRAPH_DIR    = Path("data/maryland/graph")
FINAL_DIR    = Path("data/md/final")
COUNTY_SHP   = Path("data/maryland/counties/tl_2020_us_county.shp")

BALT_COUNTY_FIPS = "510"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_plan(row_idx: int, plans_path: Path) -> dict:
    table = pq.read_table(plans_path)
    return {col: int(table[col][row_idx].as_py()) for col in table.column_names}


def _load_plans_subset(row_indices: list, plans_path: Path) -> pd.DataFrame:
    table = pq.read_table(plans_path)
    df = table.to_pandas()
    return df.iloc[row_indices].reset_index(drop=True)


def _dissolve_to_districts(plan_row: dict, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    node_ids = {int(k[1:]): v for k, v in plan_row.items()}
    gdf["district_id"] = gdf["node_id"].map(node_ids)
    dissolved = gdf.dissolve(by="district_id", aggfunc={"pop": "sum"}).reset_index()
    dissolved = dissolved.rename(columns={"pop": "population"})
    dissolved["district_id"] = dissolved["district_id"].astype(int)
    return dissolved[["district_id", "population", "geometry"]].sort_values("district_id")


def _compute_county_splits(plans_df: pd.DataFrame, gdf: gpd.GeoDataFrame) -> np.ndarray:
    county_array = gdf["county_fips"].values
    node_ids = [int(c[1:]) for c in plans_df.columns]
    node_to_pos = {nid: i for i, nid in enumerate(gdf["node_id"].values)}
    county_positions = np.array([node_to_pos.get(nid, -1) for nid in node_ids])
    valid_mask = county_positions >= 0
    plans_valid = plans_df.values[:, valid_mask]
    county_values = county_array[county_positions[valid_mask]]

    splits = np.zeros(len(plans_df), dtype=int)
    for i in tqdm(range(len(plans_df)), desc="    County splits"):
        df = pd.DataFrame({"county": county_values, "district": plans_valid[i]})
        splits[i] = df.drop_duplicates().shape[0] - df["county"].nunique()
    return splits


def _pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    costs = df[["pp_mean", "county_splits", "cut_edges"]].copy()
    costs["pp_mean"] = -costs["pp_mean"]
    n = len(costs)
    is_dominated = np.zeros(n, dtype=bool)
    cost_arr = costs.values
    for i in tqdm(range(n), desc="    Pareto check"):
        if is_dominated[i]:
            continue
        dominated_mask = (
            np.all(cost_arr <= cost_arr[i], axis=1) &
            np.any(cost_arr < cost_arr[i], axis=1)
        )
        dominated_mask[i] = False
        is_dominated[i] = dominated_mask.any()
    frontier = df[~is_dominated].copy()
    print(f"    Frontier size: {len(frontier)} / {n}")
    return frontier


def _write_map(plan_row, gdf, out_path, stats_path, metrics_row, label):
    dissolved = _dissolve_to_districts(plan_row, gdf)
    dissolved.to_file(out_path, driver="GPKG")
    stats = {
        "label": label,
        "output_file": str(out_path),
        "n_districts": int(len(dissolved)),
        "pp_min": metrics_row.get("pp_min"),
        "pp_mean": metrics_row.get("pp_mean"),
        "pp_max": metrics_row.get("pp_max"),
        "pop_dev_max": metrics_row.get("pop_dev_max"),
        "pop_dev_mean": metrics_row.get("pop_dev_mean"),
        "cut_edges": metrics_row.get("cut_edges"),
        "county_splits": metrics_row.get("county_splits"),
        "baltimore_city_districts": int(metrics_row.get("balt_districts", -1)),
        "district_populations": {
            f"district_{r['district_id']}": int(r["population"])
            for _, r in dissolved.iterrows()
        },
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    plans_path   = ENSEMBLE_DIR / "plans.parquet"
    metrics_path = ENSEMBLE_DIR / "metrics.parquet"
    gpkg_path    = GRAPH_DIR / "md_precincts_pop.gpkg"

    rng = np.random.default_rng(RANDOM_SEED)

    print("Loading precinct GeoDataFrame…")
    gdf = gpd.read_file(gpkg_path)
    if "node_id" not in gdf.columns:
        gdf["node_id"] = gdf.index

    # Baltimore City node IDs and their column names.
    balt_nodes = gdf[gdf["county_fips"] == BALT_COUNTY_FIPS]["node_id"].tolist()
    balt_cols  = [f"n{nid}" for nid in balt_nodes]
    print(f"Baltimore City: {len(balt_nodes)} precincts")

    print("Loading ensemble metrics…")
    metrics = pd.read_parquet(metrics_path)
    print(f"  Total plans: {len(metrics):,}")

    # Hard filter 1: standard geometry filters.
    mask = (metrics["pp_min"] >= PP_MIN_THRESHOLD) & (metrics["pop_dev_max"] <= POP_DEV_MAX_THRESHOLD)
    metrics = metrics[mask].copy()
    print(f"  After pp_min/pop_dev filter: {len(metrics):,} plans")

    # Hard filter 2: Baltimore City split constraint.
    print(f"Loading Baltimore City columns ({len(balt_cols)} cols) for {len(metrics):,} plans…")
    balt_plans = pq.read_table(plans_path, columns=balt_cols).to_pandas()
    balt_plans_filtered = balt_plans.iloc[metrics.index]

    # Count distinct districts per plan for Baltimore City nodes.
    balt_district_counts = balt_plans_filtered.nunique(axis=1).values
    metrics = metrics.copy()
    metrics["balt_districts"] = balt_district_counts

    before = len(metrics)
    metrics = metrics[metrics["balt_districts"] <= MAX_BALT_DISTRICTS]
    print(f"  After Baltimore City ≤{MAX_BALT_DISTRICTS} districts filter: {len(metrics):,} plans "
          f"(removed {before - len(metrics):,})")

    if len(metrics) == 0:
        # Try relaxing to 3
        print(f"  WARNING: no plans with ≤{MAX_BALT_DISTRICTS}. Relaxing to 3…")
        metrics_orig = pd.read_parquet(metrics_path)
        mask = (metrics_orig["pp_min"] >= PP_MIN_THRESHOLD) & (metrics_orig["pop_dev_max"] <= POP_DEV_MAX_THRESHOLD)
        metrics = metrics_orig[mask].copy()
        balt_plans_filtered2 = balt_plans.iloc[metrics.index]
        metrics["balt_districts"] = balt_plans_filtered2.nunique(axis=1).values
        metrics = metrics[metrics["balt_districts"] <= 3]
        print(f"  After ≤3 filter: {len(metrics):,} plans")

    print(f"\nDistribution of Baltimore City district splits in passing plans:")
    print(metrics["balt_districts"].value_counts().sort_index().to_string())

    # Sample for Pareto computation.
    n_sample = min(SAMPLE_SIZE, len(metrics))
    sample_pos = rng.choice(len(metrics), size=n_sample, replace=False)
    sample_meta = metrics.iloc[sample_pos].copy().reset_index()
    sample_meta["plan_idx"] = metrics.index[sample_pos].tolist()
    original_rows = metrics.index[sample_pos].tolist()

    print(f"\nLoading {n_sample:,} sampled plans for county-splits computation…")
    plans_sample = _load_plans_subset(original_rows, plans_path)

    print("Computing county splits…")
    splits = _compute_county_splits(plans_sample, gdf)
    sample_meta["county_splits"] = splits

    print("Computing Pareto frontier…")
    frontier = _pareto_frontier(sample_meta)
    frontier.to_csv(FINAL_DIR / "pareto_frontier.csv", index=False)

    # Select best plans.
    compact_idx = int(frontier.loc[frontier["pp_mean"].idxmax(), "plan_idx"])
    splits_sorted = frontier.sort_values(["county_splits", "pp_mean"], ascending=[True, False])
    fewest_idx = int(splits_sorted.iloc[0]["plan_idx"])

    print(f"\nSelected plans:")
    compact_meta = sample_meta[sample_meta["plan_idx"] == compact_idx].iloc[0].to_dict()
    fewest_meta  = sample_meta[sample_meta["plan_idx"] == fewest_idx].iloc[0].to_dict()
    print(f"  Best compact:       row {compact_idx}  "
          f"pp_mean={compact_meta['pp_mean']:.3f}  "
          f"county_splits={compact_meta['county_splits']:.0f}  "
          f"balt_districts={compact_meta['balt_districts']:.0f}")
    print(f"  Fewest splits:      row {fewest_idx}  "
          f"pp_mean={fewest_meta['pp_mean']:.3f}  "
          f"county_splits={fewest_meta['county_splits']:.0f}  "
          f"balt_districts={fewest_meta['balt_districts']:.0f}")

    print("\nWriting final maps…")
    compact_row = _load_plan(compact_idx, plans_path)
    fewest_row  = _load_plan(fewest_idx, plans_path)

    _write_map(compact_row, gdf,
               FINAL_DIR / "best_map_compact.gpkg",
               FINAL_DIR / "best_map_compact_stats.json",
               compact_meta, "best_map_compact")
    _write_map(fewest_row, gdf,
               FINAL_DIR / "best_map_fewest_splits.gpkg",
               FINAL_DIR / "best_map_fewest_splits_stats.json",
               fewest_meta, "best_map_fewest_splits")

    print(f"\nDone. Maps in {FINAL_DIR}")
    print(f"Baltimore City split constraint: ≤{MAX_BALT_DISTRICTS} districts")


if __name__ == "__main__":
    main()
