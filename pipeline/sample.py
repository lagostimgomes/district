"""
pipeline/sample.py

Generalized weighted ReCom MCMC sampler for any US state.

This is a direct generalization of maryland_sample_maps.py.
Key changes:
    - K = cfg.k (number of districts from StateConfig)
    - Paths use cfg.abbr.lower()
    - CHECKPOINT_EVERY = n_steps (single checkpoint at end)
    - RANDOM_SEED = 42
    - BETA = 1.0
    - POP_TOL = 0.005

Algorithm: Weighted ReCom (Recombination) MCMC
    DeFord, Duchin & Solomon (2021). Harvard Data Science Review 3(1).

STRICTLY GEOGRAPHY-ONLY — ZERO PARTISAN OR DEMOGRAPHIC DATA
"""

import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from state_configs import StateConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA = 2.0          # geographic weight strength; prob ∝ 1/w^β (0=uniform, higher=stricter county adherence)
RANDOM_SEED = 42    # reproducibility
POP_TOL = 0.005     # ±0.5% population tolerance
BURN_IN = 500       # warm-up steps before recording


# ---------------------------------------------------------------------------
# Geometry / metric helpers (identical logic to maryland_sample_maps.py)
# ---------------------------------------------------------------------------


def polsby_popper(area: float, perimeter: float) -> float:
    """Polsby-Popper compactness score.  Range [0, 1]; 1 = perfect circle."""
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


def build_geometry_cache(
    G: nx.Graph, gdf: gpd.GeoDataFrame
) -> tuple[dict, dict, dict]:
    """
    Precompute per-node and per-edge geometry quantities for O(N+E) Polsby-Popper
    computation without any geometry dissolve.

    Returns
    -------
    node_area   {node_id: area_m2}
    node_perim  {node_id: boundary_length_m}
    edge_border {(u, v): shared_border_len_m}
    """
    node_area: dict[int, float] = {}
    node_perim: dict[int, float] = {}
    geom_by_node = gdf.set_index("node_id")["geometry"].to_dict()

    for nid, geom in geom_by_node.items():
        node_area[nid] = geom.area
        node_perim[nid] = geom.boundary.length

    edge_border: dict[tuple, float] = {}
    for u, v, data in G.edges(data=True):
        edge_border[(u, v)] = data.get("border_len_m", 0.0)
        edge_border[(v, u)] = data.get("border_len_m", 0.0)

    return node_area, node_perim, edge_border


def compute_metrics_fast(
    assignment: dict,
    G: nx.Graph,
    node_area: dict,
    node_perim: dict,
    edge_border: dict,
    ideal_pop: float,
    pop_per_node: dict,
) -> dict:
    """O(N+E) metric computation — no geometry dissolve."""
    dist_area: dict[int, float] = {}
    dist_perim: dict[int, float] = {}
    dist_pop: dict[int, int] = {}

    for n, d in assignment.items():
        dist_area[d] = dist_area.get(d, 0.0) + node_area.get(n, 0.0)
        dist_perim[d] = dist_perim.get(d, 0.0) + node_perim.get(n, 0.0)
        dist_pop[d] = dist_pop.get(d, 0) + pop_per_node.get(n, 0)

    # Subtract 2× shared border for every intra-district edge.
    for u, v in G.edges():
        if assignment[u] == assignment[v]:
            d = assignment[u]
            dist_perim[d] -= 2.0 * edge_border.get((u, v), 0.0)

    pp_scores = []
    pop_deviations = []
    for d in sorted(dist_area):
        pp = polsby_popper(dist_area[d], dist_perim[d])
        pp_scores.append(pp)
        dev = abs(dist_pop[d] - ideal_pop) / ideal_pop
        pop_deviations.append(dev)

    cut_edges = sum(1 for u, v in G.edges() if assignment[u] != assignment[v])

    return {
        "pp_min": float(min(pp_scores)),
        "pp_mean": float(np.mean(pp_scores)),
        "pp_max": float(max(pp_scores)),
        "pop_dev_max": float(max(pop_deviations)),
        "pop_dev_mean": float(np.mean(pop_deviations)),
        "cut_edges": int(cut_edges),
    }


# ---------------------------------------------------------------------------
# Weighted random spanning tree (Wilson's algorithm)
# ---------------------------------------------------------------------------


def _weighted_random_spanning_tree(
    subgraph: nx.Graph, beta: float, rng: np.random.Generator
) -> nx.Graph:
    """
    Sample a random spanning tree of subgraph.
    Edge transition probability ∝ 1/weight so high-weight (community-preserving)
    edges are traversed less often and less likely to be the chosen cut.
    """
    nodes = list(subgraph.nodes())
    if len(nodes) <= 1:
        return subgraph.copy()

    tree = nx.Graph()
    tree.add_nodes_from(nodes)

    in_tree = {nodes[0]}
    path: dict = {}

    for start in nodes:
        if start in in_tree:
            continue
        current = start
        walk_pos: dict = {current: 0}
        walk_order = [current]

        while current not in in_tree:
            neighbors = list(subgraph.neighbors(current))
            if not neighbors:
                break
            weights_raw = np.array(
                [subgraph[current][nb].get("weight", 1.0) for nb in neighbors],
                dtype=float,
            )
            # β amplifies the weight contrast exponentially: prob ∝ 1/w^β.
            # β=1 → raw weights; β=2 → county-boundary edges 100× more likely
            # to be cut than intra-county edges (since W_SAME_COUNTY=10).
            probs = 1.0 / (weights_raw ** beta)
            probs /= probs.sum()
            next_node = rng.choice(neighbors, p=probs)

            if next_node in walk_pos:
                idx = walk_pos[next_node]
                for erased in walk_order[idx + 1:]:
                    del walk_pos[erased]
                    if erased in path:
                        del path[erased]
                walk_order = walk_order[: idx + 1]
            else:
                path[current] = next_node
                walk_pos[next_node] = len(walk_order)
                walk_order.append(next_node)
            current = next_node

        current = start
        while current not in in_tree:
            next_node = path[current]
            tree.add_edge(current, next_node)
            in_tree.add(current)
            current = next_node

    return tree


# ---------------------------------------------------------------------------
# ReCom step
# ---------------------------------------------------------------------------


def recom_step(
    assignment: dict,
    G: nx.Graph,
    pop_per_node: dict,
    ideal_pop: float,
    rng: np.random.Generator,
    beta: float,
    k: int,
) -> dict:
    """
    One ReCom step.  Returns updated assignment dict, or the same dict if
    no valid split was found (rejection step).
    """
    district_nodes: dict[int, list] = {d: [] for d in range(k)}
    for node, dist in assignment.items():
        district_nodes[dist].append(node)

    boundary_edges = [
        (u, v) for u, v in G.edges() if assignment[u] != assignment[v]
    ]
    if not boundary_edges:
        return assignment

    u, v = boundary_edges[rng.integers(len(boundary_edges))]
    dist_a = assignment[u]
    dist_b = assignment[v]

    nodes_ab = district_nodes[dist_a] + district_nodes[dist_b]
    subgraph = G.subgraph(nodes_ab).copy()

    tree = _weighted_random_spanning_tree(subgraph, beta, rng)

    valid_cuts = []
    low = ideal_pop * (1 - POP_TOL)
    high = ideal_pop * (1 + POP_TOL)

    for edge in list(tree.edges()):
        tree.remove_edge(*edge)
        components = list(nx.connected_components(tree))
        if len(components) == 2:
            pop_0 = sum(pop_per_node[n] for n in components[0])
            pop_1 = sum(pop_per_node[n] for n in components[1])
            if low <= pop_0 <= high and low <= pop_1 <= high:
                valid_cuts.append((edge, components[0], components[1]))
        tree.add_edge(*edge)

    if not valid_cuts:
        return assignment  # Rejection step.

    chosen_idx = rng.integers(len(valid_cuts))
    _, part_a, part_b = valid_cuts[chosen_idx]

    new_assignment = dict(assignment)
    for n in part_a:
        new_assignment[n] = dist_a
    for n in part_b:
        new_assignment[n] = dist_b

    return new_assignment


# ---------------------------------------------------------------------------
# Initial partition
# ---------------------------------------------------------------------------


def initial_partition(
    G: nx.Graph,
    pop_per_node: dict,
    ideal_pop: float,
    rng: np.random.Generator,
    k: int,
) -> dict:
    """
    Produce a valid initial partition using GerryChain's recursive_tree_part.

    Uses a relaxed tolerance for the seed partition (4× the MCMC tolerance)
    so that large high-K states (CA K=52, TX K=38) don't spend hours searching
    for a perfect starting point.  The MCMC chain and hard filters enforce the
    strict POP_TOL on every recorded plan.
    """
    from functools import partial
    from gerrychain.tree import recursive_tree_part, bipartition_tree
    from gerrychain import Graph as GCGraph

    # Relaxed epsilon for the seed only: 4× the sampling tolerance.
    # The chain mixes toward balanced plans within the first few hundred steps.
    SEED_EPSILON = min(POP_TOL * 4, 0.02)

    print(f"  Generating initial partition via GerryChain recursive_tree_part "
          f"(seed ε={SEED_EPSILON:.1%})…")
    nx.set_node_attributes(G, pop_per_node, name="population")
    gc_graph = GCGraph.from_networkx(G)

    # max_attempts=None removes the attempt ceiling — necessary for large states
    # (CA K=52, TX K=38) where the default 10k limit is routinely exceeded.
    assignment = recursive_tree_part(
        gc_graph,
        parts=range(k),
        pop_target=ideal_pop,
        pop_col="population",
        epsilon=SEED_EPSILON,
        method=partial(bipartition_tree, max_attempts=None),
    )
    return {int(kk): int(vv) for kk, vv in assignment.items()}


# ---------------------------------------------------------------------------
# Chunk-based parquet writers + checkpoint/resume
# ---------------------------------------------------------------------------

FLUSH_EVERY = 1_000   # rows per chunk file; keeps peak RAM at O(FLUSH_EVERY × nodes)


def _make_plans_schema(node_list: list) -> pa.Schema:
    return pa.schema([pa.field(f"n{n}", pa.int8()) for n in node_list])


def _make_metrics_schema() -> pa.Schema:
    return pa.schema([
        pa.field("step",         pa.int32()),
        pa.field("pp_min",       pa.float32()),
        pa.field("pp_mean",      pa.float32()),
        pa.field("pp_max",       pa.float32()),
        pa.field("pop_dev_max",  pa.float32()),
        pa.field("pop_dev_mean", pa.float32()),
        pa.field("cut_edges",    pa.int32()),
    ])


def _write_chunk(batch: list[dict], schema: pa.Schema,
                 chunk_dir: Path, chunk_id: int, prefix: str) -> None:
    """Write one batch as a numbered chunk file."""
    if not batch:
        return
    cols = {}
    for field in schema:
        key = field.name
        cols[key] = pa.array([r[key] for r in batch], type=field.type)
    path = chunk_dir / f"{prefix}_{chunk_id:05d}.parquet"
    pq.write_table(pa.table(cols, schema=schema), path, compression="snappy")


def _save_checkpoint(ensemble_dir: Path, assignment: dict,
                     rng: np.random.Generator, steps_done: int,
                     chunk_id: int) -> None:
    """Persist RNG state + current assignment so a killed run can resume."""
    ckpt = {
        "assignment": assignment,
        "rng_state":  rng.bit_generator.state,
        "steps_done": steps_done,
        "chunk_id":   chunk_id,
    }
    tmp = ensemble_dir / "checkpoint.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    tmp.replace(ensemble_dir / "checkpoint.pkl")   # atomic replace


def _load_checkpoint(ensemble_dir: Path) -> dict | None:
    path = ensemble_dir / "checkpoint.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def consolidate_chunks(ensemble_dir: Path) -> None:
    """
    Merge all chunk files into plans.parquet + metrics.parquet and remove chunks.
    Safe to call even if consolidation was previously interrupted.
    """
    chunk_dir = ensemble_dir / "chunks"
    if not chunk_dir.exists():
        return

    for prefix, out_name in [("plans", "plans.parquet"),
                              ("metrics", "metrics.parquet")]:
        chunks = sorted(chunk_dir.glob(f"{prefix}_*.parquet"))
        if not chunks:
            continue
        out = ensemble_dir / out_name
        tables = [pq.read_table(c) for c in chunks]
        pq.write_table(pa.concat_tables(tables), out, compression="snappy")
        print(f"  Consolidated {len(chunks)} {prefix} chunks → {out}")

    # Remove chunks only after both outputs are written.
    import shutil
    shutil.rmtree(chunk_dir)
    ckpt = ensemble_dir / "checkpoint.pkl"
    if ckpt.exists():
        ckpt.unlink()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sampling(
    cfg: StateConfig,
    data_root: Path,
    n_steps: int = 2000,
) -> None:
    """
    Run the weighted ReCom MCMC sampler for one state.

    Parameters
    ----------
    cfg       : StateConfig for the target state.
    data_root : Root data directory (e.g. Path("data")).
    n_steps   : Number of MCMC steps to record after burn-in.

    Outputs (written to data/{abbr}/ensemble/):
        plans.parquet   — assignment vectors
        metrics.parquet — per-step metrics (pp_min, pp_mean, pp_max,
                          pop_dev_max, pop_dev_mean, cut_edges)
    """
    abbr = cfg.abbr.lower()
    graph_dir = data_root / abbr / "graph"
    ensemble_dir = data_root / abbr / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    graph_path = graph_dir / f"{abbr}_precinct_dual_graph.gpickle"
    gpkg_path = graph_dir / f"{abbr}_precincts_pop.gpkg"

    k = cfg.k

    print(f"\n[{cfg.abbr}] Sampling {n_steps:,} plans "
          f"(K={k}, β={BETA}, seed={RANDOM_SEED})")

    # Load graph.
    print("  Loading graph…")
    with open(graph_path, "rb") as fh:
        G = pickle.load(fh)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load GeoDataFrame and build geometry cache.
    print("  Loading GeoDataFrame and building geometry cache…")
    gdf = gpd.read_file(gpkg_path)
    if "node_id" not in gdf.columns:
        gdf["node_id"] = gdf.index
    node_area, node_perim, edge_border = build_geometry_cache(G, gdf)
    print(f"  Geometry cache: {len(node_area)} nodes, {len(edge_border)//2} edges")

    pop_per_node = {n: G.nodes[n]["pop"] for n in G.nodes()}
    total_pop = sum(pop_per_node.values())
    ideal_pop = total_pop / k
    print(f"  Total pop: {total_pop:,}  |  Ideal district: {ideal_pop:,.1f}")

    # ── Checkpoint resume ──────────────────────────────────────────────────
    chunk_dir = ensemble_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    ckpt = _load_checkpoint(ensemble_dir)
    if ckpt is not None:
        assignment  = ckpt["assignment"]
        rng         = np.random.default_rng()
        rng.bit_generator.state = ckpt["rng_state"]
        start_step  = ckpt["steps_done"]
        chunk_id    = ckpt["chunk_id"] + 1
        print(f"  Resuming from step {start_step:,}  (chunk {chunk_id})")
    else:
        rng        = np.random.default_rng(RANDOM_SEED)
        assignment = initial_partition(G, pop_per_node, ideal_pop, rng, k)
        print(f"  Running {BURN_IN} burn-in steps…")
        for _ in range(BURN_IN):
            assignment = recom_step(
                assignment, G, pop_per_node, ideal_pop, rng, BETA, k
            )
        print("  Burn-in complete.")
        start_step = 0
        chunk_id   = 0

    # ── Main sampling loop ─────────────────────────────────────────────────
    node_list      = sorted(G.nodes())
    plans_schema   = _make_plans_schema(node_list)
    metrics_schema = _make_metrics_schema()
    plans_buf:   list[dict] = []
    metrics_buf: list[dict] = []

    remaining = n_steps - start_step
    print(f"  Running {remaining:,} MCMC steps "
          f"(chunk every {FLUSH_EVERY:,}, resumable)…")

    for step in tqdm(range(start_step, n_steps),
                     desc=f"  [{cfg.abbr}] Sampling",
                     initial=start_step, total=n_steps):
        assignment = recom_step(
            assignment, G, pop_per_node, ideal_pop, rng, BETA, k
        )

        plans_buf.append({f"n{n}": assignment[n] for n in node_list})
        m = compute_metrics_fast(
            assignment, G, node_area, node_perim, edge_border,
            ideal_pop, pop_per_node,
        )
        m["step"] = step
        metrics_buf.append(m)

        if len(plans_buf) >= FLUSH_EVERY:
            _write_chunk(plans_buf,   plans_schema,   chunk_dir, chunk_id, "plans")
            _write_chunk(metrics_buf, metrics_schema, chunk_dir, chunk_id, "metrics")
            _save_checkpoint(ensemble_dir, assignment, rng, step + 1, chunk_id)
            tqdm.write(f"  [{cfg.abbr}] chunk {chunk_id:05d} saved  "
                       f"(step {step+1:,}/{n_steps:,})")
            chunk_id += 1
            plans_buf.clear()
            metrics_buf.clear()

    # Final partial chunk.
    if plans_buf:
        _write_chunk(plans_buf,   plans_schema,   chunk_dir, chunk_id, "plans")
        _write_chunk(metrics_buf, metrics_schema, chunk_dir, chunk_id, "metrics")

    # Consolidate chunks → plans.parquet + metrics.parquet, remove chunks.
    print(f"  Consolidating chunks…")
    consolidate_chunks(ensemble_dir)

    print(f"[{cfg.abbr}] Sampling complete.")
    print(f"  Plans  : {ensemble_dir / 'plans.parquet'}")
    print(f"  Metrics: {ensemble_dir / 'metrics.parquet'}")
