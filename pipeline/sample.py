"""
pipeline/sample.py  (v2 — memory-efficient)

Weighted ReCom MCMC sampler for any US state.

Memory design
-------------
The v1 sampler accumulated 3–4 GB of RSS for large states (CA, NY) because
Python's allocator never returns freed pages to the OS, and every step
allocated: a NetworkX subgraph copy, a Python dict for assignment,
lists for boundary edges and valid cuts.

v2 fixes this by:

1. CompactGraph (CSR numpy arrays) replaces NetworkX in the hot loop.
   NY (14 190 nodes, 39 092 edges): ~3 MB vs ~300 MB for nx.Graph.

2. numpy int8 assignment array, modified in-place — no per-step dict copy.

3. Vectorised O(N+E) Polsby-Popper via np.bincount — no geometry dissolve.

4. O(M) valid-cut: root the spanning tree, compute subtree populations
   bottom-up, find valid cut nodes in one pass (vs. O(M²) in v1).

5. del G; del gdf; gc.collect() before the hot loop.  Pages stay in the
   Python allocator pool on macOS but are reused rather than causing growth.

6. Preallocated int8 plans_matrix[FLUSH_EVERY, N] buffer — never resized.

7. Atomic chunk writes: write to .tmp then os.replace().

8. Checkpoint saved as {node_id: district} dict — backward compatible with
   v1 checkpoints.

Peak RSS estimate (CA K=52, 24 k nodes, 50 k edges):
  CompactGraph + geometry arrays:  ~6 MB
  plans_matrix buffer:             ~24 MB
  Wilson's walk (per-step Python): ~50 MB working set
  GDF/G residual (allocator pool): ~250 MB
  Total: well under 500 MB (hard limit = 10 GB).

Algorithm: Weighted ReCom (Recombination) MCMC
    DeFord, Duchin & Solomon (2021). Harvard Data Science Review 3(1).

STRICTLY GEOGRAPHY-ONLY — ZERO PARTISAN OR DEMOGRAPHIC DATA
"""

from __future__ import annotations

import gc
import pickle
import platform
import resource
from collections import deque
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from state_configs import StateConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA         = 2.0    # prob ∝ 1/w^β; β=2 → cross-county cut 100× more likely
# K=2 states (WV, ME, NH, ID, MT, RI, HI) use a lower beta; see run_sampling().
BETA_K2      = 0.5    # β=0.5 for two-district states (statistician recommendation)
RANDOM_SEED  = 42
POP_TOL      = 0.005  # ±0.5 % population tolerance for MCMC and hard filter
# SEED_EPSILON uses a relaxed 2 % tolerance so that GerryChain's recursive
# bipartition can quickly find the initial partition even for K=52.  The
# initial partition may therefore contain districts up to 2 % off-ideal.
# A repair burn-in phase (REPAIR_BURN_IN steps at POP_TOL_REPAIR) is run
# immediately after loading the initial partition to bring all districts
# within POP_TOL before recording begins.  Without this repair pass, the
# chain gets permanently stuck: any district with |pop_dev| > POP_TOL has
# combined pop with any neighbor = ideal×2 ± (>1 %), which contains no
# valid balanced cut at POP_TOL = 0.5 %.
SEED_EPSILON    = 0.02    # starting relaxed epsilon for GerryChain partition
# NOTE: initial_partition() may double SEED_EPSILON up to 4× if needed.
# Whichever epsilon is achieved, the repair burn-in uses it as POP_TOL_REPAIR.
REPAIR_BURN_IN  = 2_000   # repair steps (relaxed tolerance) before normal burn-in
BURN_IN         = 500     # warm-up steps at POP_TOL before recording begins
FLUSH_EVERY     = 1_000   # plans per chunk file
MEM_LIMIT_GB = 10.0   # log a warning if RSS exceeds this


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------


def _rss_gb() -> float:
    """Current process RSS in gigabytes."""
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kilobytes
        return rss / 1_073_741_824 if platform.system() == "Darwin" else rss / 1_048_576
    except Exception:
        return float("nan")


def _log_rss(label: str) -> None:
    rss = _rss_gb()
    flag = "  ⚠  EXCEEDS MEM_LIMIT_GB!" if rss > MEM_LIMIT_GB else ""
    print(f"  [RAM] {label}: {rss:.2f} GB{flag}")


# ---------------------------------------------------------------------------
# CompactGraph — CSR numpy representation of the precinct dual graph
# ---------------------------------------------------------------------------


class CompactGraph:
    """
    CSR (Compressed Sparse Row) representation of the precinct dual graph.

    Replaces NetworkX in the hot MCMC loop.  Memory usage:
      NY (14 190 nodes, 39 092 edges): ~3 MB vs ~300 MB for nx.Graph
      CA (24 000 nodes, 50 000 edges): ~4 MB vs ~400 MB for nx.Graph

    Attributes
    ----------
    N            : int
    node_list    : list[int]   node_list[i] = original NetworkX node id
    node_to_idx  : dict[int, int]
    pop          : int64  [N]
    adj_ptr      : int32  [N+1]   CSR row pointers
    adj_dst      : int32  [2E]    neighbour indices (each undirected edge stored twice)
    adj_wgt      : float32 [2E]
    edges        : int32  [E, 2]  one entry per undirected edge (u_idx, v_idx),
                                  same order as G.edges()
    """

    __slots__ = ("N", "node_list", "node_to_idx", "pop",
                 "adj_ptr", "adj_dst", "adj_wgt", "edges")

    def __init__(self, G: nx.Graph) -> None:
        nodes = sorted(G.nodes())
        N = len(nodes)
        self.N = N
        self.node_list: list[int] = nodes
        self.node_to_idx: dict[int, int] = {n: i for i, n in enumerate(nodes)}

        self.pop = np.array(
            [G.nodes[n].get("pop", 0) for n in nodes], dtype=np.int64
        )

        # ── Build CSR adjacency ───────────────────────────────────────────────
        deg = np.array([G.degree(n) for n in nodes], dtype=np.int32)
        self.adj_ptr = np.zeros(N + 1, dtype=np.int32)
        np.cumsum(deg, out=self.adj_ptr[1:])

        total = int(self.adj_ptr[-1])
        self.adj_dst = np.empty(total, dtype=np.int32)
        self.adj_wgt = np.empty(total, dtype=np.float32)

        fill = np.zeros(N, dtype=np.int32)
        for u, v, data in G.edges(data=True):
            ui = self.node_to_idx[u]
            vi = self.node_to_idx[v]
            w = float(data.get("weight", 1.0))
            # Store u → v
            p = int(self.adj_ptr[ui]) + int(fill[ui])
            self.adj_dst[p] = vi
            self.adj_wgt[p] = w
            fill[ui] += 1
            # Store v → u
            p = int(self.adj_ptr[vi]) + int(fill[vi])
            self.adj_dst[p] = ui
            self.adj_wgt[p] = w
            fill[vi] += 1

        # ── Edge list aligned with G.edges() ─────────────────────────────────
        self.edges = np.array(
            [(self.node_to_idx[u], self.node_to_idx[v]) for u, v in G.edges()],
            dtype=np.int32,
        )  # shape (E, 2)

    def neighbors(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (dst_indices, weights) for node i."""
        s = int(self.adj_ptr[i])
        e = int(self.adj_ptr[i + 1])
        return self.adj_dst[s:e], self.adj_wgt[s:e]


# ---------------------------------------------------------------------------
# Geometry cache — numpy arrays for O(N+E) Polsby-Popper
# ---------------------------------------------------------------------------


def build_geometry_cache(
    G: nx.Graph,
    gdf: gpd.GeoDataFrame,
    node_list: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract per-node area/perimeter and per-edge shared-border-length into
    numpy arrays aligned with node_list and G.edges() respectively.

    Returns
    -------
    area_arr   : float64 [N]  precinct area (m²)
    perim_arr  : float64 [N]  precinct boundary length (m)
    border_arr : float64 [E]  shared border length; index e corresponds to
                              the e-th edge in G.edges()
    """
    N = len(node_list)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    area_arr  = np.zeros(N, dtype=np.float64)
    perim_arr = np.zeros(N, dtype=np.float64)

    geom_by_node: dict = gdf.set_index("node_id")["geometry"].to_dict()
    for nid, geom in geom_by_node.items():
        i = node_to_idx.get(int(nid))
        if i is not None and geom is not None:
            area_arr[i]  = geom.area
            perim_arr[i] = geom.boundary.length

    E = G.number_of_edges()
    border_arr = np.zeros(E, dtype=np.float64)
    for e, (_, _, data) in enumerate(G.edges(data=True)):
        border_arr[e] = data.get("border_len_m", 0.0)

    return area_arr, perim_arr, border_arr


# ---------------------------------------------------------------------------
# Vectorised O(N+E) metrics — no geometry dissolve
# ---------------------------------------------------------------------------


def compute_metrics_fast(
    assignment: np.ndarray,
    cg: CompactGraph,
    area_arr: np.ndarray,
    perim_arr: np.ndarray,
    border_arr: np.ndarray,
    ideal_pop: float,
    k: int,
) -> dict:
    """
    Compute per-plan metrics in O(N+E) using np.bincount.

    Polsby-Popper is computed without dissolving district geometries:
      PP(d) = 4π · Σ_i area[i] / (Σ_i perim[i] − 2 · Σ_{intra-edges} border)²

    The −2·border correction removes shared borders that are interior to the
    district from the district perimeter.
    """
    a = assignment.astype(np.intp)   # np.bincount requires intp

    dist_area  = np.bincount(a, weights=area_arr,             minlength=k)
    dist_perim = np.bincount(a, weights=perim_arr,            minlength=k)
    dist_pop   = np.bincount(a, weights=cg.pop.astype(float), minlength=k)

    eu = cg.edges[:, 0]
    ev = cg.edges[:, 1]
    same = a[eu] == a[ev]

    # Subtract 2× shared border for every intra-district edge
    np.subtract.at(dist_perim, a[eu[same]], 2.0 * border_arr[same])

    pp = np.where(
        dist_perim > 0,
        np.clip(4.0 * np.pi * dist_area / dist_perim ** 2, 0.0, 1.0),
        0.0,
    )
    dev = np.abs(dist_pop - ideal_pop) / ideal_pop

    cut_mask = ~same
    return {
        "pp_min":         float(pp.min()),
        "pp_mean":        float(pp.mean()),
        "pp_max":         float(pp.max()),
        "pop_dev_max":    float(dev.max()),
        "pop_dev_mean":   float(dev.mean()),
        "cut_edges":      int(np.sum(cut_mask)),
        # Total length (m) of all inter-district precinct boundaries.
        # Low values indicate clean, geographically natural splits;
        # high values indicate ragged, patchwork boundaries.
        "cut_border_m":   float(border_arr[cut_mask].sum()),
    }


# ---------------------------------------------------------------------------
# Wilson's Loop-Erased Random Walk — spanning tree sampling
# ---------------------------------------------------------------------------


def _wilson_lerw_compact(
    merged_indices: np.ndarray,
    cg: CompactGraph,
    beta: float,
    rng: np.random.Generator,
) -> tuple[dict[int, int], int]:
    """
    Sample a random spanning tree over the merged subgraph using Wilson's
    loop-erased random walk (Wilson 1996).

    Transition probability ∝ 1/w^β so high-weight (intra-community) edges
    are traversed less often and less likely to appear in the spanning tree,
    increasing the probability of cross-community cuts.

    Parameters
    ----------
    merged_indices : int32 [M]  CSR indices of nodes in the merged subgraph

    Returns
    -------
    parent : dict {child_idx: parent_idx}  (root has no entry)
    root   : int  root node index
    """
    merged_set = set(merged_indices.tolist())

    # Precompute normalised transition probabilities once per node.
    # Restricting to merged_set ensures the walk stays inside the subgraph.
    local_adj: dict[int, tuple[list[int], np.ndarray]] = {}
    for ni in merged_indices.tolist():
        dst, wgt = cg.neighbors(ni)
        mask = np.fromiter(
            (int(d) in merged_set for d in dst), dtype=bool, count=len(dst)
        )
        if not mask.any():
            local_adj[ni] = ([], np.empty(0, dtype=np.float64))
            continue
        nbrs = [int(d) for d in dst[mask]]
        probs = 1.0 / wgt[mask].astype(np.float64) ** beta
        probs /= probs.sum()
        local_adj[ni] = (nbrs, probs)

    root = int(merged_indices[0])
    parent: dict[int, int] = {}
    in_tree: set[int] = {root}

    for start in merged_indices.tolist():
        if start in in_tree:
            continue

        current = start
        walk_pos: dict[int, int] = {current: 0}
        walk_order: list[int] = [current]

        while current not in in_tree:
            nbrs, probs = local_adj[current]
            if not nbrs:
                break  # isolated node — should not occur in a valid planar graph
            chosen_pos = int(rng.choice(len(nbrs), p=probs))
            next_node = nbrs[chosen_pos]

            if next_node in walk_pos:
                # Loop erasure: truncate the walk at the re-visited node
                idx = walk_pos[next_node]
                for erased in walk_order[idx + 1:]:
                    del walk_pos[erased]
                    parent.pop(erased, None)
                walk_order = walk_order[: idx + 1]
            else:
                parent[current] = next_node
                walk_pos[next_node] = len(walk_order)
                walk_order.append(next_node)
            current = next_node

        # Commit the loop-erased path to the tree
        current = start
        while current not in in_tree:
            in_tree.add(current)
            current = parent[current]

    return parent, root


# ---------------------------------------------------------------------------
# O(M) valid-cut finder
# ---------------------------------------------------------------------------


def _find_valid_cut(
    parent: dict[int, int],
    root: int,
    merged_indices: np.ndarray,
    cg: CompactGraph,
    ideal_pop: float,
    pop_tol: float,
    rng: np.random.Generator,
) -> set[int] | None:
    """
    Find population-balanced spanning-tree cuts in O(M).

    Algorithm
    ---------
    1. Build children dict from parent (invert the parent pointers).
    2. BFS from root → topological ordering.
    3. Bottom-up pass: subtree_pop[v] = pop[v] + Σ subtree_pop[child].
    4. A non-root node v is a valid cut iff:
         low ≤ subtree_pop[v] ≤ high  AND
         low ≤ total_pop − subtree_pop[v] ≤ high
    5. Pick one valid node uniformly at random.
    6. BFS from the chosen node (following children only) → component 1.

    Returns
    -------
    set of node indices forming component 1 (the cut subtree), or None if
    no valid cut exists (rejection step).
    """
    # Build children list (O(M))
    children: dict[int, list[int]] = {int(ni): [] for ni in merged_indices}
    for child, par in parent.items():
        if par in children:
            children[par].append(child)

    # BFS topological order from root
    order: list[int] = []
    visited: set[int] = {root}
    q: deque[int] = deque([root])
    while q:
        node = q.popleft()
        order.append(node)
        for child in children.get(node, []):
            if child not in visited:
                visited.add(child)
                q.append(child)

    # Bottom-up subtree population accumulation
    subtree_pop: dict[int, int] = {int(ni): int(cg.pop[ni]) for ni in merged_indices}
    for node in reversed(order):
        for child in children.get(node, []):
            subtree_pop[node] += subtree_pop[child]

    total_pop = subtree_pop[root]
    low  = ideal_pop * (1.0 - pop_tol)
    high = ideal_pop * (1.0 + pop_tol)

    valid_nodes = [
        node for node in merged_indices.tolist()
        if node != root
        and low <= subtree_pop[node] <= high
        and low <= (total_pop - subtree_pop[node]) <= high
    ]

    if not valid_nodes:
        return None

    cut_node = valid_nodes[int(rng.integers(len(valid_nodes)))]

    # BFS from cut_node following children → component 1
    comp1: set[int] = set()
    q2: deque[int] = deque([cut_node])
    while q2:
        node = q2.popleft()
        comp1.add(node)
        for child in children.get(node, []):
            q2.append(child)

    return comp1


# ---------------------------------------------------------------------------
# ReCom step — numpy, in-place
# ---------------------------------------------------------------------------


def recom_step(
    assignment: np.ndarray,
    cg: CompactGraph,
    ideal_pop: float,
    rng: np.random.Generator,
    beta: float,
    k: int,
    pop_tol: float = POP_TOL,
) -> bool:
    """
    One weighted ReCom step.  Modifies assignment in-place.

    pop_tol controls the balance requirement for valid cuts.  Use POP_TOL
    (0.5 %) during normal sampling; use POP_TOL_REPAIR (2 %) during the
    repair burn-in to fix any stuck districts from the initial partition.

    Returns True (accepted) or False (rejected — no valid cut found).
    """
    eu = cg.edges[:, 0]
    ev = cg.edges[:, 1]

    # Vectorised boundary-edge detection
    is_boundary = assignment[eu] != assignment[ev]
    if not is_boundary.any():
        return False

    boundary_idxs = np.where(is_boundary)[0]
    chosen_edge   = int(boundary_idxs[rng.integers(len(boundary_idxs))])
    u = int(cg.edges[chosen_edge, 0])
    v = int(cg.edges[chosen_edge, 1])

    dist_a = int(assignment[u])
    dist_b = int(assignment[v])

    # All nodes belonging to the two selected districts
    merged_mask    = (assignment == dist_a) | (assignment == dist_b)
    merged_indices = np.where(merged_mask)[0].astype(np.int32)

    # Sample random spanning tree via Wilson's LERW
    parent, root = _wilson_lerw_compact(merged_indices, cg, beta, rng)

    # Find a population-balanced cut
    comp1 = _find_valid_cut(parent, root, merged_indices, cg, ideal_pop, pop_tol, rng)
    if comp1 is None:
        return False  # Rejection step — assignment unchanged

    # Apply the split in-place
    for node in merged_indices.tolist():
        assignment[node] = np.int8(dist_a if node in comp1 else dist_b)

    return True


# ---------------------------------------------------------------------------
# Initial partition
# ---------------------------------------------------------------------------


def initial_partition(
    G: nx.Graph,
    pop_per_node: dict,
    ideal_pop: float,
    rng: np.random.Generator,
    k: int,
) -> dict[int, int]:
    """
    Produce a valid initial partition using GerryChain's recursive_tree_part.

    Strategy: try epsilon=SEED_EPSILON (2 %).  If GerryChain fails to balance
    any bipartition within MAX_BIPARTITION_ATTEMPTS attempts, double epsilon
    and retry the full partition.  Repeat up to MAX_EPSILON_DOUBLINGS times.
    Districts may start up to epsilon_final off-ideal; the repair burn-in in
    run_sampling runs at POP_TOL_REPAIR (= final epsilon) to fix them.

    The `max_attempts` cap (rather than None) is critical: without it,
    bipartition_tree loops forever on geographically constrained sub-regions
    (e.g. coastal California narrow peninsulas) where no balanced cut exists.
    """
    from functools import partial

    from gerrychain import Graph as GCGraph
    from gerrychain.tree import bipartition_tree, recursive_tree_part

    MAX_BIPARTITION_ATTEMPTS = 5_000   # per-bipartition attempt ceiling
    MAX_EPSILON_DOUBLINGS    = 4       # up to SEED_EPSILON × 2^4 = 32 % max

    nx.set_node_attributes(G, pop_per_node, name="population")
    gc_graph = GCGraph.from_networkx(G)

    epsilon = SEED_EPSILON
    for attempt in range(MAX_EPSILON_DOUBLINGS + 1):
        print(f"  Generating initial partition via GerryChain "
              f"(ε={epsilon:.1%}, attempt {attempt + 1})…")
        try:
            assignment = recursive_tree_part(
                gc_graph,
                parts=range(k),
                pop_target=ideal_pop,
                pop_col="population",
                epsilon=epsilon,
                method=partial(bipartition_tree,
                               max_attempts=MAX_BIPARTITION_ATTEMPTS),
            )
            print(f"  Initial partition found at ε={epsilon:.1%}.")
            return (
                {int(kk): int(vv) for kk, vv in assignment.items()},
                epsilon,
            )
        except Exception as exc:
            print(f"  ε={epsilon:.1%} failed ({type(exc).__name__}: {exc}). "
                  f"Retrying with doubled epsilon…")
            epsilon = min(epsilon * 2.0, 0.50)  # cap at 50 %

    raise RuntimeError(
        f"Could not generate initial partition for K={k} after "
        f"{MAX_EPSILON_DOUBLINGS + 1} attempts (final ε={epsilon:.1%})."
    )


# ---------------------------------------------------------------------------
# Chunk-based parquet writers
# ---------------------------------------------------------------------------


def _make_metrics_schema() -> pa.Schema:
    return pa.schema([
        pa.field("step",          pa.int32()),
        pa.field("pp_min",        pa.float32()),
        pa.field("pp_mean",       pa.float32()),
        pa.field("pp_max",        pa.float32()),
        pa.field("pop_dev_max",   pa.float32()),
        pa.field("pop_dev_mean",  pa.float32()),
        pa.field("cut_edges",     pa.int32()),
        pa.field("cut_border_m",  pa.float32()),
    ])


def _write_chunk_atomic(
    plans_matrix: np.ndarray,
    metrics_buf: list[dict],
    count: int,
    node_list: list[int],
    chunk_dir: Path,
    chunk_id: int,
) -> None:
    """
    Write `count` rows from plans_matrix and metrics_buf as numbered chunk
    files using atomic tmp → rename to prevent partial writes.
    """
    metrics_schema = _make_metrics_schema()

    # Plans chunk
    cols_p = {
        f"n{node_list[j]}": pa.array(plans_matrix[:count, j], type=pa.int8())
        for j in range(len(node_list))
    }
    p_path = chunk_dir / f"plans_{chunk_id:05d}.parquet"
    p_tmp  = p_path.with_suffix(".tmp")
    pq.write_table(pa.table(cols_p), p_tmp, compression="snappy")
    p_tmp.replace(p_path)

    # Metrics chunk
    cols_m = {
        f.name: pa.array([r[f.name] for r in metrics_buf[:count]], type=f.type)
        for f in metrics_schema
    }
    m_path = chunk_dir / f"metrics_{chunk_id:05d}.parquet"
    m_tmp  = m_path.with_suffix(".tmp")
    pq.write_table(pa.table(cols_m, schema=metrics_schema), m_tmp, compression="snappy")
    m_tmp.replace(m_path)


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------


def _save_checkpoint(
    ensemble_dir: Path,
    assignment: np.ndarray,
    node_list: list[int],
    rng: np.random.Generator,
    steps_done: int,
    chunk_id: int,
) -> None:
    """
    Atomically save RNG state + current assignment.

    Assignment is stored as {node_id: district} for backward compatibility
    with v1 checkpoints and with select.py which loads plans by node_id.
    """
    assignment_dict = {int(node_list[i]): int(assignment[i]) for i in range(len(node_list))}
    ckpt = {
        "assignment": assignment_dict,
        "rng_state":  rng.bit_generator.state,
        "steps_done": steps_done,
        "chunk_id":   chunk_id,
    }
    tmp = ensemble_dir / "checkpoint.pkl.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(ckpt, fh, protocol=4)
    tmp.replace(ensemble_dir / "checkpoint.pkl")  # atomic rename


def _load_checkpoint(ensemble_dir: Path) -> dict | None:
    path = ensemble_dir / "checkpoint.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def consolidate_chunks(ensemble_dir: Path) -> None:
    """
    Merge all chunk files into plans.parquet + metrics.parquet, then remove
    the chunk directory and checkpoint file.

    Safe to call even if a previous consolidation was interrupted mid-run.
    """
    chunk_dir = ensemble_dir / "chunks"
    if not chunk_dir.exists():
        return

    for prefix, out_name in [("plans", "plans.parquet"),
                              ("metrics", "metrics.parquet")]:
        chunks = sorted(chunk_dir.glob(f"{prefix}_*.parquet"))
        if not chunks:
            continue
        out    = ensemble_dir / out_name
        tables = [pq.read_table(c) for c in chunks]
        pq.write_table(pa.concat_tables(tables), out, compression="snappy")
        print(f"  Consolidated {len(chunks)} {prefix} chunks → {out}")

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
    n_steps: int = 2_000,
) -> None:
    """
    Run the weighted ReCom MCMC sampler for one state.

    Execution sequence
    ------------------
    1. Load G (NetworkX) and gdf (GeoDataFrame).
    2. Extract geometry → numpy arrays; build CompactGraph.
    3. If fresh run: call initial_partition(G, …) → numpy assignment.
    4. If resuming:  load checkpoint → numpy assignment + RNG state.
    5. del G; del gdf; gc.collect()  ← frees most memory before hot loop.
    6. Repair burn-in (fresh runs only): REPAIR_BURN_IN steps at
       POP_TOL_REPAIR to fix any districts > POP_TOL from the initial
       partition (SEED_EPSILON=2 % can create up to 2 %-off districts).
    7. Normal burn-in (fresh runs only): BURN_IN steps at POP_TOL.
    8. Main sampling loop — all operations on numpy arrays.
       Flush FLUSH_EVERY rows atomically; save checkpoint after each flush.
    9. Consolidate chunks → plans.parquet + metrics.parquet.

    Outputs (written to data/{abbr}/ensemble/):
        plans.parquet   — int8 assignment vectors (columns n{node_id})
        metrics.parquet — per-step metrics (pp_min/mean/max, pop_dev, cut_edges)
    """
    abbr         = cfg.abbr.lower()
    graph_dir    = data_root / abbr / "graph"
    ensemble_dir = data_root / abbr / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir    = ensemble_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    graph_path = graph_dir / f"{abbr}_precinct_dual_graph.gpickle"
    gpkg_path  = graph_dir / f"{abbr}_precincts_pop.gpkg"
    k = cfg.k

    # Adaptive beta: for K=2 states, β=2.0 creates a 100:1 preference for
    # cross-county edges that over-constrains the two-district problem and
    # produces peninsula connectors.  β=0.5 creates only a 1.4:1 preference,
    # allowing the MCMC to explore a broader set of geographically natural cuts.
    beta = BETA_K2 if k <= 2 else BETA

    print(f"\n[{cfg.abbr}] Sampling {n_steps:,} plans "
          f"(K={k}, β={beta}, seed={RANDOM_SEED})")
    _log_rss("start")

    # ── Load NetworkX graph ────────────────────────────────────────────────────
    print("  Loading graph…")
    with open(graph_path, "rb") as fh:
        G: nx.Graph = pickle.load(fh)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    _log_rss("after G load")

    # Population statistics
    pop_per_node = {n: int(G.nodes[n].get("pop", 0)) for n in G.nodes()}
    total_pop  = sum(pop_per_node.values())
    ideal_pop  = total_pop / k
    print(f"  Total pop: {total_pop:,}  |  Ideal district: {ideal_pop:,.1f}")

    # ── Check for existing checkpoint ──────────────────────────────────────────
    ckpt     = _load_checkpoint(ensemble_dir)
    resuming = ckpt is not None

    # ── Load GeoDataFrame for geometry cache only ──────────────────────────────
    print("  Loading precinct GeoDataFrame…")
    gdf = gpd.read_file(gpkg_path)
    if "node_id" not in gdf.columns:
        gdf["node_id"] = gdf.index
    print(f"  {len(gdf)} precincts")
    _log_rss("after gdf load")

    # ── Build numpy geometry cache ─────────────────────────────────────────────
    print("  Building geometry cache (numpy arrays)…")
    node_list = sorted(G.nodes())
    area_arr, perim_arr, border_arr = build_geometry_cache(G, gdf, node_list)
    print(f"  Geometry: {len(area_arr)} nodes, {len(border_arr)} edges")

    # ── Build CompactGraph (CSR) ───────────────────────────────────────────────
    print("  Building CompactGraph (CSR)…")
    cg = CompactGraph(G)
    _log_rss("after CompactGraph")

    # ── Initial partition — must happen before del G ───────────────────────────
    if not resuming:
        rng                      = np.random.default_rng(RANDOM_SEED)
        assign_dict, seed_eps    = initial_partition(G, pop_per_node, ideal_pop, rng, k)
        assignment               = np.array(
            [np.int8(assign_dict[node_list[i]]) for i in range(cg.N)],
            dtype=np.int8,
        )
        start_step = 0
        chunk_id   = 0
    else:
        rng = np.random.default_rng()
        rng.bit_generator.state = ckpt["rng_state"]
        assignment = np.array(
            [np.int8(ckpt["assignment"][node_list[i]]) for i in range(cg.N)],
            dtype=np.int8,
        )
        start_step = ckpt["steps_done"]
        chunk_id   = ckpt["chunk_id"] + 1
        print(f"  Resuming from step {start_step:,}  (next chunk {chunk_id})")

    # ── Free large Python objects before the hot loop ─────────────────────────
    # Pages stay in Python's allocator pool on macOS but are reused for small
    # working-set allocations rather than causing the RSS to keep growing.
    print("  Releasing NetworkX graph and GeoDataFrame…")
    del gdf
    del G
    gc.collect()
    _log_rss("after del G+gdf")

    # ── Targeted repair (fresh runs only) ────────────────────────────────────
    # The initial partition uses SEED_EPSILON (2 %, possibly doubled).  Any
    # district with |pop_dev| > POP_TOL is "stuck": the ReCom MCMC at
    # POP_TOL=0.5 % will never find a valid cut involving it because:
    #   combined = (1 + d_x + d_y) × 2 × ideal ≥ 2×ideal×(1 + d_x)
    # exceeds 2×ideal×(1+POP_TOL) when d_x > POP_TOL — so no half can be
    # within ±POP_TOL of ideal.
    #
    # Targeted repair: identify the most imbalanced district, select a
    # random boundary edge to a neighbor, compute the exact minimum tolerance
    # required for a valid cut of those two districts (= |combined − 2×ideal|
    # / (2×ideal) + small buffer), and run ReCom with that tolerance.
    # Each targeted step is guaranteed to find a valid cut; we repeat until
    # all districts are within POP_TOL (usually < K×10 steps).
    if not resuming:
        pop_arr   = cg.pop.astype(float)
        eu        = cg.edges[:, 0]
        ev        = cg.edges[:, 1]

        def _current_max_dev() -> tuple[float, int]:
            dp  = np.bincount(assignment.astype(np.intp), weights=pop_arr, minlength=k)
            dev = np.abs(dp - ideal_pop) / ideal_pop
            return float(dev.max()), int(np.argmax(dev)), dp

        max_dev, bad_dist, dist_pop = _current_max_dev()
        if max_dev > POP_TOL:
            print(f"  Repair needed: pop_dev_max = {max_dev:.4%}  "
                  f"(worst district = {bad_dist})")
            repair_iters = 0
            MAX_REPAIR = REPAIR_BURN_IN * 10   # safety ceiling
            while max_dev > POP_TOL and repair_iters < MAX_REPAIR:
                # Find boundary edges involving the worst district.
                bad_mask = (
                    ((assignment[eu] == bad_dist) & (assignment[ev] != bad_dist)) |
                    ((assignment[ev] == bad_dist) & (assignment[eu] != bad_dist))
                )
                bad_edges = np.where(bad_mask)[0]
                if len(bad_edges) == 0:
                    # Isolated district (shouldn't happen in valid graph)
                    break

                # Pick a random boundary edge and get the neighbour district.
                e       = int(bad_edges[rng.integers(len(bad_edges))])
                u, v    = int(cg.edges[e, 0]), int(cg.edges[e, 1])
                dist_a  = int(assignment[u])
                dist_b  = int(assignment[v])

                # Compute the minimum tolerance that guarantees a valid cut.
                # For combined = (d_a + d_b) * ideal, valid halves must each
                # be in [ideal*(1−t), ideal*(1+t)].  This requires:
                #   t ≥ |combined/(2*ideal) − 1|
                combined    = float(dist_pop[dist_a] + dist_pop[dist_b])
                min_tol     = abs(combined / (2.0 * ideal_pop) - 1.0)
                repair_tol  = max(min_tol * 1.05 + 1e-4, POP_TOL)  # 5 % buffer

                # Merge and recut with the computed tolerance.
                merged_mask    = (assignment == dist_a) | (assignment == dist_b)
                merged_indices = np.where(merged_mask)[0].astype(np.int32)
                parent, root   = _wilson_lerw_compact(merged_indices, cg, beta, rng)
                comp1          = _find_valid_cut(
                    parent, root, merged_indices, cg, ideal_pop, repair_tol, rng
                )
                if comp1 is not None:
                    for node in merged_indices.tolist():
                        assignment[node] = np.int8(dist_a if node in comp1 else dist_b)

                repair_iters += 1
                max_dev, bad_dist, dist_pop = _current_max_dev()

            print(f"  Repair done in {repair_iters} steps: "
                  f"pop_dev_max = {max_dev:.4%}")
            if max_dev > POP_TOL:
                print(f"  ⚠  pop_dev_max {max_dev:.4%} still exceeds POP_TOL "
                      f"{POP_TOL:.4%} after {repair_iters} repair steps.")
        else:
            print(f"  pop_dev_max = {max_dev:.4%} — no repair needed.")

    # ── Normal burn-in (fresh runs only) ──────────────────────────────────────
    if not resuming:
        print(f"  Running {BURN_IN} normal burn-in steps (POP_TOL={POP_TOL:.1%})…")
        for _ in range(BURN_IN):
            recom_step(assignment, cg, ideal_pop, rng, beta, k)
        print("  Burn-in complete.")
        _log_rss("after burn-in")

    # ── Main sampling loop ─────────────────────────────────────────────────────
    # Preallocated int8 plans buffer — never resized during sampling.
    plans_matrix = np.empty((FLUSH_EVERY, cg.N), dtype=np.int8)
    metrics_buf: list[dict] = []
    buf_pos = 0

    remaining = n_steps - start_step
    print(f"  Running {remaining:,} MCMC steps "
          f"(flush every {FLUSH_EVERY:,} steps, resumable)…")

    for step in tqdm(
        range(start_step, n_steps),
        desc=f"  [{cfg.abbr}] Sampling",
        initial=start_step,
        total=n_steps,
    ):
        recom_step(assignment, cg, ideal_pop, rng, beta, k)

        # Record current assignment and metrics into the buffer
        plans_matrix[buf_pos] = assignment
        m = compute_metrics_fast(
            assignment, cg, area_arr, perim_arr, border_arr, ideal_pop, k
        )
        m["step"] = step
        metrics_buf.append(m)
        buf_pos += 1

        # Flush when buffer is full
        if buf_pos >= FLUSH_EVERY:
            _write_chunk_atomic(
                plans_matrix, metrics_buf, buf_pos, node_list, chunk_dir, chunk_id
            )
            _save_checkpoint(
                ensemble_dir, assignment, node_list, rng, step + 1, chunk_id
            )
            tqdm.write(
                f"  [{cfg.abbr}] chunk {chunk_id:05d} saved  "
                f"(step {step + 1:,}/{n_steps:,})  RAM {_rss_gb():.2f} GB"
            )
            chunk_id += 1
            buf_pos = 0
            metrics_buf.clear()

    # Flush the final partial buffer
    if buf_pos > 0:
        _write_chunk_atomic(
            plans_matrix, metrics_buf, buf_pos, node_list, chunk_dir, chunk_id
        )

    # ── Consolidate chunks → final parquet files ───────────────────────────────
    print("  Consolidating chunks…")
    consolidate_chunks(ensemble_dir)

    _log_rss("sampling complete")
    print(f"[{cfg.abbr}] Sampling complete.")
    print(f"  Plans  : {ensemble_dir / 'plans.parquet'}")
    print(f"  Metrics: {ensemble_dir / 'metrics.parquet'}")
