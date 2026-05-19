"""
pipeline/select.py

Generalized map selector for any US state.

This is a direct generalization of maryland_select_map.py.
Key changes:
    - Paths use cfg.abbr.lower()
    - K = cfg.k

Selection pipeline:
    1. Hard filter: pp_min >= 0.10 AND pop_dev_max <= 0.005
    2. Sample up to 5,000 plans from the filtered set.
    3. Compute county_splits for the sampled plans.
    4. Compute Pareto frontier: maximise pp_mean, minimise county_splits,
       minimise cut_edges.
    5. Extract two representative maps from the frontier.

STRICTLY GEOGRAPHY-ONLY — ZERO PARTISAN OR DEMOGRAPHIC DATA
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from state_configs import StateConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_SIZE = 5_000
RANDOM_SEED = 42
# PP_MIN_THRESHOLD = 0.10 was too strict for CA (K=52) and other large
# multi-district states whose coastal/mountain geography makes it geometrically
# impossible for every district to reach PP≥0.10.  Lowered to 0.05 — this
# still filters plans with severely non-compact districts while allowing valid
# results for geographically challenging states.
PP_MIN_THRESHOLD = 0.05
POP_DEV_MAX_THRESHOLD = 0.005

# Peninsula filter: a cut edge whose shared border is shorter than this
# threshold relative to the median cut-edge border indicates a narrow
# peninsula connector.  Plans where any cut edge is < 5% of the median
# cut-edge border are deprioritised in final selection.
PENINSULA_RATIO_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def _load_plan(row_idx: int, plans_path: Path) -> dict:
    """Load one plan (assignment vector) from plans.parquet by row index."""
    table = pq.read_table(plans_path)
    return {col: int(table[col][row_idx].as_py()) for col in table.column_names}


def _load_plans_subset(row_indices: list, plans_path: Path) -> pd.DataFrame:
    """Load a subset of plans by row indices."""
    table = pq.read_table(plans_path)
    df = table.to_pandas()
    return df.iloc[row_indices].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Hard filters
# ---------------------------------------------------------------------------


def _apply_hard_filters(metrics: pd.DataFrame) -> pd.DataFrame:
    print("  Applying hard filters…")
    print(f"    Before: {len(metrics):,} plans")
    filtered = metrics[
        (metrics["pp_min"] >= PP_MIN_THRESHOLD)
        & (metrics["pop_dev_max"] <= POP_DEV_MAX_THRESHOLD)
    ].copy()
    print(
        f"    After (pp_min>={PP_MIN_THRESHOLD},"
        f" pop_dev_max<={POP_DEV_MAX_THRESHOLD}): {len(filtered):,} plans"
    )
    if len(filtered) == 0:
        raise RuntimeError(
            "No plans passed the hard filters. "
            "Consider relaxing thresholds or increasing n_steps."
        )
    return filtered


# ---------------------------------------------------------------------------
# County split computation
# ---------------------------------------------------------------------------


def _compute_county_splits(
    plans_df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    node_col_prefix: str = "n",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-plan county split metrics.

    Returns
    -------
    splits              : total county splits (sum over counties of districts_in_county − 1)
    max_county_districts: maximum number of districts any single county is split into

    The second metric penalises plans that concentrate fragmentation on one
    county/city — e.g. a single dense urban county carved into 4 districts
    would score 4, whereas the same total splits spread across 4 rural counties
    would score 2. Both metrics are minimised in the Pareto search.
    """
    county_array = gdf["county_fips"].values
    node_ids = [int(c[len(node_col_prefix):]) for c in plans_df.columns]
    node_to_pos = {nid: i for i, nid in enumerate(gdf["node_id"].values)}
    county_positions = np.array([node_to_pos.get(nid, -1) for nid in node_ids])
    valid_mask = county_positions >= 0

    plans_matrix = plans_df.values
    county_values = county_array[county_positions[valid_mask]]
    plans_valid = plans_matrix[:, valid_mask]

    splits = np.zeros(len(plans_df), dtype=int)
    max_county_districts = np.zeros(len(plans_df), dtype=int)

    for i in tqdm(range(len(plans_df)), desc="    County splits"):
        assignment_row = plans_valid[i]
        df = pd.DataFrame({"county": county_values, "district": assignment_row})
        per_county = df.groupby("county")["district"].nunique()
        splits[i] = int((per_county - 1).sum())
        max_county_districts[i] = int(per_county.max())

    return splits, max_county_districts


# ---------------------------------------------------------------------------
# Cut-border metrics (peninsula detection)
# ---------------------------------------------------------------------------


def _compute_cut_border_metrics(
    plans_df: pd.DataFrame,
    G: nx.Graph,
    node_col_prefix: str = "n",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-plan cut-border metrics using graph edge border lengths.

    These metrics are computed from the plans data at selection time, so they
    work for both old ensembles (metrics.parquet without cut_border_m) and new
    ones (which include it in metrics.parquet as well).

    Returns
    -------
    cut_border_m     : float64 [n_plans]
        Total length (m) of all inter-district precinct boundaries.
        Lower is better — reflects clean, geographically natural splits.
    min_cut_border_m : float64 [n_plans]
        Length (m) of the shortest single inter-district boundary.
        Very low values (< 5% of median) indicate a narrow peninsula connector
        that appears as an isolated pocket at map scale.
    """
    col_node_ids = [int(c[len(node_col_prefix):]) for c in plans_df.columns]
    nid_to_col: dict[int, int] = {nid: i for i, nid in enumerate(col_node_ids)}

    eu_list, ev_list, border_list = [], [], []
    for u, v, data in G.edges(data=True):
        ci = nid_to_col.get(int(u), -1)
        cj = nid_to_col.get(int(v), -1)
        if ci >= 0 and cj >= 0:
            eu_list.append(ci)
            ev_list.append(cj)
            border_list.append(float(data.get("border_len_m", 0.0)))

    n_plans = len(plans_df)
    if not eu_list:
        return np.zeros(n_plans, dtype=np.float64), np.zeros(n_plans, dtype=np.float64)

    eu_arr     = np.array(eu_list, dtype=np.int32)
    ev_arr     = np.array(ev_list, dtype=np.int32)
    border_arr = np.array(border_list, dtype=np.float64)

    plans_matrix = plans_df.values  # (n_plans, n_cols)

    # Vectorised across all plans at once: (n_plans, n_edges)
    u_asgn = plans_matrix[:, eu_arr]
    v_asgn = plans_matrix[:, ev_arr]
    is_cut  = u_asgn != v_asgn  # True where the edge is a district boundary

    # Total cut-border length per plan
    cut_border_m = (is_cut.astype(np.float64) * border_arr[np.newaxis, :]).sum(axis=1)

    # Minimum cut-border length per plan (peninsula indicator)
    BIG = 1e12
    cut_or_big = np.where(is_cut, border_arr[np.newaxis, :], BIG)
    raw_min = cut_or_big.min(axis=1)
    min_cut_border_m = np.where(raw_min >= BIG, 0.0, raw_min)

    return cut_border_m, min_cut_border_m


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------


def _pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pareto non-dominated set.
    Objectives: maximise pp_mean, minimise county_splits, minimise cut_edges,
                minimise max_county_districts, minimise cut_border_m.

    max_county_districts is the worst-case fragmentation of any single county.
    cut_border_m penalises plans with long, ragged inter-district boundaries;
    a clean geographic split tends to follow natural/administrative lines and
    has a lower total boundary length than a patchwork plan.
    """
    print("  Computing Pareto frontier…")
    objectives = ["pp_mean", "county_splits", "cut_edges",
                  "max_county_districts", "cut_border_m"]
    # Only include cut_border_m if it's present (backward compat with old ensembles
    # that ran before the metric was added — select.py always computes it now).
    avail = [o for o in objectives if o in df.columns]
    costs = df[avail].copy()
    costs["pp_mean"] = -costs["pp_mean"]  # convert to minimisation

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
    print(f"    Frontier size: {len(frontier)} / {n} sampled plans")
    print(f"    Objectives used: {avail}")
    return frontier


# ---------------------------------------------------------------------------
# Select representative plans
# ---------------------------------------------------------------------------


def _select_best_plans(frontier: pd.DataFrame) -> tuple[int, int]:
    """
    Return (compact_idx, splits_idx) — original row indices in plans.parquet.

    Selection tiers (applied in order):
    1. Prefer plans with no peninsula connectors (min_cut_border_m >= 5% of
       median cut-border, per PENINSULA_RATIO_THRESHOLD).  If no such plan
       exists in the frontier, fall back to the full frontier.
    2. Among preferred plans, take those with the lowest max_county_districts
       (prevents concentrating fragmentation on one dense urban county).
    3. Optimise the primary objective within that group.
    """
    work = frontier.copy()

    # ── Tier 1: peninsula filter ─────────────────────────────────────────────
    # Compare min_cut_border_m to the mean single-edge length per plan
    # (= cut_border_m / cut_edges).  A plan where the shortest cut edge is
    # < PENINSULA_RATIO_THRESHOLD × average cut edge has a suspicious narrow
    # peninsula connector and is deprioritised.
    if ("min_cut_border_m" in work.columns and "cut_border_m" in work.columns
            and "cut_edges" in work.columns):
        mean_edge = work["cut_border_m"] / work["cut_edges"].clip(lower=1)
        ratio = work["min_cut_border_m"] / mean_edge.clip(lower=1.0)
        non_peninsula = work[ratio >= PENINSULA_RATIO_THRESHOLD]
        if len(non_peninsula) > 0:
            work = non_peninsula
            print(f"    Peninsula filter: {len(work)} / {len(frontier)} plans retained "
                  f"(min/mean edge ratio ≥ {PENINSULA_RATIO_THRESHOLD})")
        else:
            print(f"    Peninsula filter: no non-peninsula plans found — "
                  f"using full frontier ({len(frontier)} plans)")

    # ── Tier 2: county fragmentation ────────────────────────────────────────
    best_max_cd = work["max_county_districts"].min()
    best_cd_plans = work[work["max_county_districts"] == best_max_cd]

    # ── Tier 3: primary objectives ───────────────────────────────────────────
    compact_pos = best_cd_plans["pp_mean"].idxmax()
    compact_idx = work.loc[compact_pos, "plan_idx"]

    splits_sorted = best_cd_plans.sort_values(
        ["county_splits", "pp_mean"], ascending=[True, False]
    )
    splits_idx = splits_sorted.iloc[0]["plan_idx"]

    return int(compact_idx), int(splits_idx)


# ---------------------------------------------------------------------------
# Dissolve to district polygons
# ---------------------------------------------------------------------------


def _dissolve_to_districts(
    plan_row: dict,
    gdf: gpd.GeoDataFrame,
    node_col_prefix: str = "n",
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    node_ids = {int(k[len(node_col_prefix):]): v for k, v in plan_row.items()}
    gdf["district_id"] = gdf["node_id"].map(node_ids)
    dissolved = gdf.dissolve(by="district_id", aggfunc={"pop": "sum"}).reset_index()
    dissolved = dissolved.rename(columns={"pop": "population"})
    dissolved["district_id"] = dissolved["district_id"].astype(int)
    dissolved = dissolved[["district_id", "population", "geometry"]].sort_values(
        "district_id"
    )
    return dissolved


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------


def _write_map(
    plan_row: dict,
    gdf: gpd.GeoDataFrame,
    out_path: Path,
    stats_path: Path,
    metrics_row: dict,
    label: str,
) -> None:
    dissolved = _dissolve_to_districts(plan_row, gdf)
    dissolved.to_file(out_path, driver="GPKG")
    print(f"  Saved {label}: {out_path}")

    pop_values = dissolved["population"].values
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
        "cut_border_m": metrics_row.get("cut_border_m"),
        "min_cut_border_m": metrics_row.get("min_cut_border_m"),
        "county_splits": metrics_row.get("county_splits"),
        "max_county_districts": metrics_row.get("max_county_districts"),
        "district_populations": {
            f"district_{row['district_id']}": int(row["population"])
            for _, row in dissolved.iterrows()
        },
    }
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  Saved stats: {stats_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_maps(cfg: StateConfig, data_root: Path) -> dict:
    """
    Select Pareto-optimal maps from the ensemble for one state.

    Parameters
    ----------
    cfg       : StateConfig for the target state.
    data_root : Root data directory (e.g. Path("data")).

    Outputs (written to data/{abbr}/final/):
        best_map_compact.gpkg
        best_map_fewest_splits.gpkg
        best_map_compact_stats.json
        best_map_fewest_splits_stats.json
        pareto_frontier.csv
        report.json

    Returns
    -------
    dict  — report dictionary (also written to report.json).
    """
    abbr = cfg.abbr.lower()
    ensemble_dir = data_root / abbr / "ensemble"
    graph_dir = data_root / abbr / "graph"
    final_dir = data_root / abbr / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    plans_path = ensemble_dir / "plans.parquet"
    metrics_path = ensemble_dir / "metrics.parquet"
    gpickle_path = graph_dir / f"{abbr}_precinct_dual_graph.gpickle"
    gpkg_path = graph_dir / f"{abbr}_precincts_pop.gpkg"

    print(f"\n[{cfg.abbr}] Selecting maps for {cfg.name}")

    # Consolidate any leftover chunks from a previously interrupted sampling run.
    from pipeline.sample import consolidate_chunks
    consolidate_chunks(ensemble_dir)

    rng = np.random.default_rng(RANDOM_SEED)

    # Load graph.
    print("  Loading graph…")
    with open(gpickle_path, "rb") as fh:
        G = pickle.load(fh)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load GeoDataFrame.
    print("  Loading precinct GeoDataFrame…")
    gdf = gpd.read_file(gpkg_path)
    if "node_id" not in gdf.columns:
        gdf["node_id"] = gdf.index
    print(f"  {len(gdf)} precincts")

    # Load metrics and apply hard filters.
    print("  Loading ensemble metrics…")
    metrics = pd.read_parquet(metrics_path)
    total_plans = len(metrics)
    print(f"  metrics.parquet: {total_plans:,} rows")
    metrics = _apply_hard_filters(metrics)

    # Sample plans for county_splits computation.
    n_sample = min(SAMPLE_SIZE, len(metrics))
    sample_indices_in_filtered = rng.choice(len(metrics), size=n_sample, replace=False)
    sample_df_meta = (
        metrics.iloc[sample_indices_in_filtered].copy().reset_index()
    )
    sample_df_meta["plan_idx"] = sample_indices_in_filtered
    original_row_numbers = metrics.index[sample_indices_in_filtered].tolist()

    print(f"  Loading {n_sample:,} sampled plans from plans.parquet…")
    plans_sample = _load_plans_subset(original_row_numbers, plans_path)

    print("  Computing county splits…")
    splits, max_county_districts = _compute_county_splits(plans_sample, gdf)
    sample_df_meta["county_splits"] = splits
    sample_df_meta["max_county_districts"] = max_county_districts
    print(f"    max_county_districts distribution:\n"
          f"      {pd.Series(max_county_districts).value_counts().sort_index().to_dict()}")

    print("  Computing cut-border metrics (peninsula detection)…")
    cut_border_m, min_cut_border_m = _compute_cut_border_metrics(plans_sample, G)
    sample_df_meta["cut_border_m"]     = cut_border_m
    sample_df_meta["min_cut_border_m"] = min_cut_border_m
    print(f"    cut_border_m    : min={cut_border_m.min():.0f} m  "
          f"median={np.median(cut_border_m):.0f} m  "
          f"max={cut_border_m.max():.0f} m")
    print(f"    min_cut_border_m: min={min_cut_border_m.min():.1f} m  "
          f"median={np.median(min_cut_border_m):.1f} m  "
          f"max={min_cut_border_m.max():.0f} m")

    # Pareto frontier.
    frontier = _pareto_frontier(sample_df_meta)
    frontier_csv_path = final_dir / "pareto_frontier.csv"
    frontier.to_csv(frontier_csv_path, index=False)
    print(f"  Saved Pareto frontier: {frontier_csv_path}")

    # Select best plans.
    compact_orig_idx, splits_orig_idx = _select_best_plans(frontier)
    print(f"  Selected plans:")
    print(f"    best_compact       -> plans.parquet row {compact_orig_idx}")
    print(f"    best_fewest_splits -> plans.parquet row {splits_orig_idx}")

    # Load full assignment rows.
    compact_plan_row = _load_plan(compact_orig_idx, plans_path)
    splits_plan_row = _load_plan(splits_orig_idx, plans_path)

    compact_metrics = (
        sample_df_meta[sample_df_meta["plan_idx"] == compact_orig_idx].iloc[0].to_dict()
    )
    splits_metrics = (
        sample_df_meta[sample_df_meta["plan_idx"] == splits_orig_idx].iloc[0].to_dict()
    )

    # Write maps.
    print("  Writing final maps…")
    _write_map(
        compact_plan_row, gdf,
        out_path=final_dir / "best_map_compact.gpkg",
        stats_path=final_dir / "best_map_compact_stats.json",
        metrics_row=compact_metrics,
        label="best_map_compact",
    )
    _write_map(
        splits_plan_row, gdf,
        out_path=final_dir / "best_map_fewest_splits.gpkg",
        stats_path=final_dir / "best_map_fewest_splits_stats.json",
        metrics_row=splits_metrics,
        label="best_map_fewest_splits",
    )

    # Write report.
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "Weighted ReCom MCMC (DeFord, Duchin & Solomon 2021)",
        "system": f"{cfg.name} Blind Redistricting (generalized pipeline)",
        "state": cfg.name,
        "abbr": cfg.abbr,
        "fips": cfg.fips,
        "k": cfg.k,
        "partisan_data_used": False,
        "demographic_data_used": False,
        "electoral_data_used": False,
        "inputs": {
            "graph": str(gpickle_path),
            "plans": str(plans_path),
            "metrics": str(metrics_path),
        },
        "ensemble": {
            "total_plans": int(total_plans),
            "plans_passing_hard_filters": int(len(metrics)),
            "plans_sampled_for_pareto": int(n_sample),
            "pareto_frontier_size": int(len(frontier)),
        },
        "hard_filters": {
            "pp_min_threshold": PP_MIN_THRESHOLD,
            "pop_dev_max_threshold": POP_DEV_MAX_THRESHOLD,
        },
        "pareto_objectives": [
            "maximise pp_mean",
            "minimise county_splits",
            "minimise cut_edges",
            "minimise max_county_districts",
            "minimise cut_border_m",
        ],
        "peninsula_filter": {
            "enabled": True,
            "description": (
                "Plans where min_cut_border_m / median(cut_border_m) < "
                f"{PENINSULA_RATIO_THRESHOLD} are deprioritised as potential "
                "peninsula connectors (narrow artificial boundaries)."
            ),
            "ratio_threshold": PENINSULA_RATIO_THRESHOLD,
        },
        "best_map_compact": {
            "plan_row": compact_orig_idx,
            "pp_mean": compact_metrics.get("pp_mean"),
            "county_splits": int(compact_metrics.get("county_splits", -1)),
            "cut_edges": int(compact_metrics.get("cut_edges", -1)),
            "cut_border_m": compact_metrics.get("cut_border_m"),
            "min_cut_border_m": compact_metrics.get("min_cut_border_m"),
        },
        "best_map_fewest_splits": {
            "plan_row": splits_orig_idx,
            "pp_mean": splits_metrics.get("pp_mean"),
            "county_splits": int(splits_metrics.get("county_splits", -1)),
            "cut_edges": int(splits_metrics.get("cut_edges", -1)),
            "cut_border_m": splits_metrics.get("cut_border_m"),
            "min_cut_border_m": splits_metrics.get("min_cut_border_m"),
        },
        "outputs": {
            "best_map_compact": str(final_dir / "best_map_compact.gpkg"),
            "best_map_fewest_splits": str(final_dir / "best_map_fewest_splits.gpkg"),
            "pareto_frontier": str(frontier_csv_path),
        },
        "note": (
            "Maps selected on geographic criteria only. "
            "A post-hoc VRA audit may be applied externally. "
            "The algorithm never saw partisan, racial, or demographic data."
        ),
    }
    report_path = final_dir / "report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Report saved: {report_path}")

    print(f"[{cfg.abbr}] Map selection complete.")
    return report
