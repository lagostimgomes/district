"""
pipeline/build_graph.py

Generalized weighted precinct dual graph builder for any US state.

This is a direct generalization of maryland_build_graph.py.
Key changes from the Maryland-specific version:
    - FIPS filter and paths driven by StateConfig instead of hardcoded "24".
    - County FIPS list built dynamically from the national county shapefile.
    - skip_water=True path omits the water edge-weight divisor for speed.
    - Output paths: data/{abbr}/graph/{abbr}_precinct_dual_graph.gpickle
                    data/{abbr}/graph/{abbr}_precincts_pop.gpkg

Algorithm (Section 3.2 of the technical plan):
    1. Load VTD precincts; reproject to cfg.crs.
    2. Aggregate population from census block shapefile (POP20) via
       centroid-in-polygon spatial join.
    3. Centroid-in-polygon spatial joins for county, place, cousub admin units.
    4. Build adjacency graph: edges where precincts share >= 1 m border.
    5. Compute edge weights (admin-unit multipliers + road divisor).
    6. Validate connectivity; save graph + GeoPackage.

Edge weight formula (skip_water=True default):
    base = 1.0
    × W_SAME_COUNTY (10) if both precincts in same county
    × W_SAME_PLACE  (5)  if both precincts in same incorporated place
    × W_SAME_COUSUB (3)  if both precincts in same county subdivision
    ÷ W_ROAD_DIV    (2)  if a primary road crosses the shared border (50 m buffer)

STRICTLY GEOGRAPHY-ONLY — ZERO PARTISAN OR DEMOGRAPHIC DATA
"""

import json
import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.strtree import STRtree
from tqdm import tqdm

from state_configs import StateConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_BORDER_M = 50.0      # minimum shared border (metres) to register an edge
# Raised from 1.0 → 50.0 (CS professor recommendation):
# Near-miss polygon corners can share < 10 m due to digitization noise;
# these spurious edges create thin peninsula connectors in K=2 states.
# 50 m eliminates noise while retaining all real precinct boundaries.

# Edge weight multipliers.
W_SAME_COUNTY = 10.0
W_SAME_PLACE = 5.0
W_SAME_COUSUB = 3.0
W_WATER_DIV = 5.0        # ÷ divisor when water crosses shared border
W_ROAD_DIV = 2.0         # ÷ divisor when a primary road crosses shared border

WATER_BUFFER_M = 100.0
ROAD_BUFFER_M = 50.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_shapefile(directory: Path, pattern: str = "*.shp") -> Path:
    matches = list(directory.rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No shapefile matching '{pattern}' found under {directory}"
        )
    return matches[0]


def _shared_border_length(geom_a, geom_b) -> float:
    """Return the length (metres) of the shared boundary between two geometries."""
    try:
        intersection = geom_a.boundary.intersection(geom_b.boundary)
        return intersection.length
    except Exception:
        return 0.0


def _build_road_tree(roads_shp: Path, crs: str):
    """Return (STRtree, geoms_list) for primary road geometries."""
    gdf = gpd.read_file(roads_shp).to_crs(crs)
    if gdf.empty:
        return None, []
    geoms = gdf.geometry.values.tolist()
    return STRtree(geoms), geoms


def _build_water_tree(water_area_dir: Path, water_linear_dir: Path, crs: str):
    """
    Load all water geometries (per-county subdirs) and return STRtree + list.
    Returns (None, []) if no water files are found (skip_water path).
    """
    all_geoms = []
    for d in [water_area_dir, water_linear_dir]:
        if not d.exists():
            continue
        shp_files = list(d.rglob("*.shp"))
        for shp in shp_files:
            try:
                g = gpd.read_file(shp).to_crs(crs)
                if not g.empty:
                    all_geoms.extend(g.geometry.values.tolist())
            except Exception as exc:
                print(f"  WARNING: could not read {shp}: {exc}")
    if not all_geoms:
        return None, []
    return STRtree(all_geoms), all_geoms


# ---------------------------------------------------------------------------
# Step 1 — Load VTDs
# ---------------------------------------------------------------------------


def _load_vtds(cfg: StateConfig, data_root: Path) -> gpd.GeoDataFrame:
    """
    Load atomic units for this state: VTD precincts if available, census
    tracts otherwise (for states like CA/HI/MO/OR that don't publish VTDs).
    """
    state_dir = data_root / cfg.abbr.lower()
    vtd_dir   = state_dir / "vtd_precincts"
    tract_dir = state_dir / "census_tracts"

    # Pick source: prefer VTDs; fall back to tracts.
    try:
        shp = _find_shapefile(vtd_dir)
        source = "VTD precincts"
    except FileNotFoundError:
        try:
            shp = _find_shapefile(tract_dir)
            source = "census tracts (VTD fallback)"
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No shapefile found under {vtd_dir} or {tract_dir}. "
                "Run download_state first."
            )

    print(f"  Loading {source} from {shp.parent}…")
    gdf = gpd.read_file(shp)
    print(f"    {len(gdf)} units loaded (CRS: {gdf.crs})")
    gdf = gdf.to_crs(cfg.crs)
    print(f"    Reprojected to {cfg.crs}")

    geoid_col = next(
        (c for c in gdf.columns if c.upper() in ("GEOID20", "GEOID")), None
    )
    if geoid_col is None:
        raise ValueError(f"Cannot find GEOID column in {shp.name}.")
    gdf = gdf.rename(columns={geoid_col: "GEOID20"})
    gdf = gdf[["GEOID20", "geometry"]].copy()
    gdf = gdf.reset_index(drop=True)
    gdf["node_id"] = gdf.index
    return gdf


# ---------------------------------------------------------------------------
# Step 2 — Join population from census blocks
# ---------------------------------------------------------------------------


def _join_pop_from_blocks(
    gdf: gpd.GeoDataFrame,
    cfg: StateConfig,
    data_root: Path,
) -> gpd.GeoDataFrame:
    """
    Aggregate census block POP20 to VTD precincts via centroid-within-polygon
    spatial join.  The tabblock20 shapefile includes POP20 from the 2020
    redistricting file.  Only total population is used.
    """
    blocks_dir = data_root / cfg.abbr.lower() / "census_blocks"
    shp = _find_shapefile(blocks_dir)
    print(f"  Loading census blocks: {shp}")
    blocks = gpd.read_file(shp, columns=["GEOID20", "POP20", "geometry"])
    blocks = blocks.to_crs(cfg.crs)
    blocks["pop_int"] = (
        pd.to_numeric(blocks["POP20"], errors="coerce").fillna(0).astype(int)
    )
    print(f"  {len(blocks):,} blocks; total pop: {blocks['pop_int'].sum():,}")

    print("  Computing block centroids for spatial join…")
    block_centroids = blocks.copy()
    block_centroids["geometry"] = blocks.geometry.centroid

    print("  Joining block centroids to VTD polygons…")
    joined = gpd.sjoin(
        block_centroids[["GEOID20", "pop_int", "geometry"]],
        gdf[["node_id", "geometry"]],
        how="left",
        predicate="within",
    )

    pop_by_node = joined.groupby("node_id")["pop_int"].sum().reset_index()
    pop_by_node.columns = ["node_id", "pop"]

    gdf = gdf.merge(pop_by_node, on="node_id", how="left")
    gdf["pop"] = gdf["pop"].fillna(0).astype(int)

    matched = (gdf["pop"] > 0).sum()
    print(f"  Population aggregated for {matched}/{len(gdf)} precincts")
    print(f"  Total population: {gdf['pop'].sum():,}")
    return gdf


# ---------------------------------------------------------------------------
# Step 3 — Admin unit spatial joins (centroid-in-polygon)
# ---------------------------------------------------------------------------


def _join_admin_units(
    gdf: gpd.GeoDataFrame,
    cfg: StateConfig,
    data_root: Path,
    county_shp: Path,
) -> gpd.GeoDataFrame:
    """
    Add county_fips, place_fips, cousub_fips columns to gdf via
    centroid-in-polygon spatial joins.
    """
    print("  Computing precinct centroids for admin unit joins…")
    centroids = gdf.copy()
    centroids["geometry"] = gdf.geometry.centroid

    def _join(layer_path_or_dir, fips_col_candidates, out_col, state_filter=None):
        """Join one admin layer. layer_path_or_dir may be a .shp path or directory."""
        if isinstance(layer_path_or_dir, Path) and layer_path_or_dir.is_dir():
            try:
                shp = _find_shapefile(layer_path_or_dir)
            except FileNotFoundError:
                print(f"    WARNING: no shapefile in {layer_path_or_dir}; {out_col}=None")
                gdf[out_col] = None
                return
        else:
            shp = layer_path_or_dir

        try:
            layer = gpd.read_file(shp).to_crs(cfg.crs)
        except Exception as exc:
            print(f"    WARNING: could not read {shp}: {exc}; {out_col}=None")
            gdf[out_col] = None
            return

        if state_filter:
            state_col = next(
                (
                    c for c in layer.columns
                    if c.upper() in ("STATEFP20", "STATEFP", "STATE_FIPS")
                ),
                None,
            )
            if state_col:
                layer = layer[layer[state_col] == state_filter].copy()
                print(f"    Filtered to state {state_filter}: {len(layer)} features")

        fips_col = next(
            (c for c in layer.columns if c.upper() in [x.upper() for x in fips_col_candidates]),
            None,
        )
        if fips_col is None:
            print(
                f"    WARNING: could not find FIPS column in {shp.name}; {out_col}=None"
            )
            gdf[out_col] = None
            return

        layer = layer[[fips_col, "geometry"]].rename(columns={fips_col: out_col})
        joined = gpd.sjoin(
            centroids[["node_id", "geometry"]],
            layer,
            how="left",
            predicate="within",
        )
        joined = joined.drop_duplicates(subset="node_id", keep="first")
        gdf[out_col] = gdf["node_id"].map(joined.set_index("node_id")[out_col])
        matched = gdf[out_col].notna().sum()
        print(f"    {out_col}: {matched}/{len(gdf)} precincts matched")

    state_dir = data_root / cfg.abbr.lower()

    print("    Joining county FIPS…")
    _join(
        county_shp,
        ["GEOID20", "GEOID", "COUNTYFP20", "COUNTYFP"],
        "county_fips",
        state_filter=cfg.fips,
    )

    print("    Joining place FIPS…")
    _join(
        state_dir / "places",
        ["PLACEFP20", "PLACEFP", "GEOID20", "GEOID"],
        "place_fips",
    )

    print("    Joining county subdivision FIPS…")
    _join(
        state_dir / "county_subdivisions",
        ["COUSUBFP20", "COUSUBFP", "GEOID20", "GEOID"],
        "cousub_fips",
    )

    return gdf


# ---------------------------------------------------------------------------
# Step 4 — Build adjacency graph
# ---------------------------------------------------------------------------


def _build_adjacency_graph(gdf: gpd.GeoDataFrame) -> nx.Graph:
    print(f"  Building adjacency graph for {len(gdf)} precincts…")
    G = nx.Graph()

    for _, row in gdf.iterrows():
        G.add_node(
            row["node_id"],
            geoid=row["GEOID20"],
            pop=int(row["pop"]),
            county_fips=row.get("county_fips"),
            place_fips=row.get("place_fips"),
            cousub_fips=row.get("cousub_fips"),
            area_m2=float(row.geometry.area),
        )

    print("    Building spatial index…")
    geoms = list(gdf.geometry)
    tree = STRtree(geoms)

    print("    Detecting shared borders (>= 1 m)…")
    edge_count = 0
    for i, geom_i in enumerate(tqdm(geoms, desc="    Adjacency")):
        candidates = tree.query(geom_i)
        for j in candidates:
            if j <= i:
                continue
            geom_j = geoms[j]
            if geom_i.touches(geom_j) or geom_i.intersects(geom_j):
                border_len = _shared_border_length(geom_i, geom_j)
                if border_len >= MIN_BORDER_M:
                    G.add_edge(i, j, border_len_m=round(border_len, 2), weight=1.0)
                    edge_count += 1

    print(f"    Added {edge_count} edges")
    return G


# ---------------------------------------------------------------------------
# Step 5 — Compute edge weights
# ---------------------------------------------------------------------------


def _compute_edge_weights(
    G: nx.Graph,
    gdf: gpd.GeoDataFrame,
    water_tree,
    water_geoms: list,
    road_tree,
    road_geoms: list,
    skip_water: bool,
) -> nx.Graph:
    """Compute geographic edge weights using STRtree spatial indices."""
    print("  Computing geographic edge weights…")

    node_attrs = {}
    for _, row in gdf.iterrows():
        node_attrs[row["node_id"]] = {
            "county_fips": row.get("county_fips"),
            "place_fips": row.get("place_fips"),
            "cousub_fips": row.get("cousub_fips"),
            "geometry": row.geometry,
        }

    def _hits_tree(buffered, tree, geoms):
        if tree is None or buffered.is_empty:
            return False
        candidates = tree.query(buffered)
        return any(geoms[i].intersects(buffered) for i in candidates)

    for u, v, data in tqdm(G.edges(data=True), desc="    Weighting", total=G.number_of_edges()):
        weight = 1.0
        a = node_attrs[u]
        b = node_attrs[v]

        # Admin-unit multipliers.
        if a["county_fips"] and a["county_fips"] == b["county_fips"]:
            weight *= W_SAME_COUNTY
        if a["place_fips"] and a["place_fips"] == b["place_fips"]:
            weight *= W_SAME_PLACE
        if a["cousub_fips"] and a["cousub_fips"] == b["cousub_fips"]:
            weight *= W_SAME_COUSUB

        # Road divisor.
        try:
            shared_border = a["geometry"].boundary.intersection(b["geometry"].boundary)
        except Exception:
            shared_border = a["geometry"].boundary

        if not shared_border.is_empty:
            if not skip_water and water_tree is not None:
                water_buf = shared_border.buffer(WATER_BUFFER_M)
                if _hits_tree(water_buf, water_tree, water_geoms):
                    weight /= W_WATER_DIV

            road_buf = shared_border.buffer(ROAD_BUFFER_M)
            if _hits_tree(road_buf, road_tree, road_geoms):
                weight /= W_ROAD_DIV

        G[u][v]["weight"] = round(weight, 4)

    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    if weights:
        print(
            f"    Weight distribution: min={min(weights):.4f}, "
            f"max={max(weights):.4f}, mean={np.mean(weights):.4f}"
        )
    return G


# ---------------------------------------------------------------------------
# Step 6 — Validate and save
# ---------------------------------------------------------------------------


def _validate_and_save(
    G: nx.Graph,
    gdf: gpd.GeoDataFrame,
    cfg: StateConfig,
    data_root: Path,
    skip_water: bool,
) -> None:
    abbr = cfg.abbr.lower()
    graph_dir = data_root / abbr / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    print("  Validating graph…")

    components = list(nx.connected_components(G))
    if len(components) > 1:
        print(
            f"  WARNING: Graph has {len(components)} connected components. "
            "Largest component will be used. Inspect isolated nodes."
        )
        largest = max(components, key=len)
        G = G.subgraph(largest).copy()

    isolated = [n for n in G.nodes if G.degree(n) == 0]
    if isolated:
        print(f"  WARNING: {len(isolated)} isolated nodes found.")

    total_pop = sum(G.nodes[n]["pop"] for n in G.nodes)
    print(f"  Total population: {total_pop:,}")

    zero_pop = [n for n in G.nodes if G.nodes[n]["pop"] == 0]
    if zero_pop:
        print(f"  WARNING: {len(zero_pop)} zero-population precincts.")

    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    if not all(w > 0 for w in weights):
        print("  WARNING: non-positive edge weights detected.")

    # Save graph.
    gpickle_path = graph_dir / f"{abbr}_precinct_dual_graph.gpickle"
    with open(gpickle_path, "wb") as fh:
        pickle.dump(G, fh)
    print(f"  Saved graph: {gpickle_path}")

    # Save GeoPackage (filter gdf to nodes that remain after largest-component extraction).
    node_set = set(G.nodes)
    gdf_out = gdf[gdf["node_id"].isin(node_set)].copy()
    gpkg_path = graph_dir / f"{abbr}_precincts_pop.gpkg"
    gdf_out.to_file(gpkg_path, driver="GPKG")
    print(f"  Saved GeoPackage: {gpkg_path}")

    # Save summary JSON.
    summary = {
        "state": cfg.name,
        "abbr": cfg.abbr,
        "fips": cfg.fips,
        "k": cfg.k,
        "crs": cfg.crs,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "is_connected": nx.is_connected(G),
        "total_population": total_pop,
        "zero_pop_nodes": len(zero_pop),
        "weight_min": round(min(weights), 4) if weights else None,
        "weight_max": round(max(weights), 4) if weights else None,
        "weight_mean": round(float(np.mean(weights)), 4) if weights else None,
        "skip_water": skip_water,
        "weight_params": {
            "same_county": W_SAME_COUNTY,
            "same_place": W_SAME_PLACE,
            "same_cousub": W_SAME_COUSUB,
            "water_divisor": None if skip_water else W_WATER_DIV,
            "road_divisor": W_ROAD_DIV,
            "water_buffer_m": WATER_BUFFER_M,
            "road_buffer_m": ROAD_BUFFER_M,
        },
        "min_border_m": MIN_BORDER_M,
    }
    summary_path = graph_dir / f"{abbr}_precinct_dual_graph_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved summary: {summary_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_graph(
    cfg: StateConfig,
    data_root: Path,
    county_shp: Path,
    roads_shp: Path,
    skip_water: bool = True,
) -> nx.Graph:
    """
    Build a weighted precinct dual graph for the given state.

    Parameters
    ----------
    cfg        : StateConfig for the target state.
    data_root  : Root data directory (e.g. Path("data")).
    county_shp : Path to the national county shapefile
                 (e.g. data/maryland/counties/tl_2020_us_county.shp).
    roads_shp  : Path to the national primary roads shapefile
                 (e.g. data/maryland/roads/tl_2020_us_primaryroads.shp).
    skip_water : If True (default), omit water-based edge weight divisor.

    Returns
    -------
    nx.Graph  (also persisted to data/{abbr}/graph/{abbr}_precinct_dual_graph.gpickle)
    """
    print(f"\n[{cfg.abbr}] Building precinct dual graph for {cfg.name}")

    # Step 1: load VTDs.
    gdf = _load_vtds(cfg, data_root)

    # Step 2: join population.
    print("  Joining population from census blocks…")
    gdf = _join_pop_from_blocks(gdf, cfg, data_root)

    # Step 3: admin unit joins.
    print("  Joining admin units…")
    gdf = _join_admin_units(gdf, cfg, data_root, county_shp)

    # Step 4: build adjacency graph.
    G = _build_adjacency_graph(gdf)

    # Step 5: edge weights.
    print("  Loading road geometries…")
    road_tree, road_geoms = _build_road_tree(roads_shp, cfg.crs)
    print(f"    Road index: {len(road_geoms)} features")

    water_tree, water_geoms = None, []
    if not skip_water:
        abbr = cfg.abbr.lower()
        water_area_dir = data_root / abbr / "water_area"
        water_linear_dir = data_root / abbr / "water_linear"
        print("  Loading water geometries…")
        water_tree, water_geoms = _build_water_tree(water_area_dir, water_linear_dir, cfg.crs)
        print(f"    Water index: {len(water_geoms)} features")

    G = _compute_edge_weights(G, gdf, water_tree, water_geoms, road_tree, road_geoms, skip_water)

    # Step 6: validate and save.
    _validate_and_save(G, gdf, cfg, data_root, skip_water)

    print(f"[{cfg.abbr}] Graph build complete.")
    return G
