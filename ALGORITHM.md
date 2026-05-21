# Algorithm and Mathematical Reference

## Blind Redistricting — Technical Specification

**Version:** Initial release  
**Scope:** Congressional district map generation for all 50 US states  
**Guarantee:** Zero partisan, racial, or demographic data used at any stage of map drawing

---

## Table of Contents

1. [Design Principle: What "Blind" Means](#1-design-principle-what-blind-means)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Data Sources and Preprocessing](#3-data-sources-and-preprocessing)
4. [Precinct Dual Graph Construction](#4-precinct-dual-graph-construction)
5. [Edge Weight Formula](#5-edge-weight-formula)
6. [The MCMC Sampler: Weighted ReCom](#6-the-mcmc-sampler-weighted-recom)
7. [Random Spanning Tree: Wilson's Algorithm](#7-random-spanning-tree-wilsons-algorithm)
8. [Population Balance and Hard Feasibility](#8-population-balance-and-hard-feasibility)
9. [Metrics Computed Per Plan](#9-metrics-computed-per-plan)
10. [Hard Filters](#10-hard-filters)
11. [Pareto Frontier Selection](#11-pareto-frontier-selection)
12. [Selecting Two Representative Plans](#12-selecting-two-representative-plans)
13. [MCMC Step Count Scaling](#13-mcmc-step-count-scaling)
14. [Burn-In and Chain Initialization](#14-burn-in-and-chain-initialization)
15. [Reproducibility and Determinism](#15-reproducibility-and-determinism)
16. [Post-Hoc Partisan Lean (Strictly Separate)](#16-post-hoc-partisan-lean-strictly-separate)
17. [What the Algorithm Cannot Do](#17-what-the-algorithm-cannot-do)
18. [Parameter Summary](#18-parameter-summary)
19. [References](#19-references)

---

## 1. Design Principle: What "Blind" Means

The redistricting algorithm operates in strict isolation from any data that could encode partisan preference, racial composition, or electoral history. Specifically:

**Never loaded, never used, never proxied:**
- Party registration by precinct
- Past election results (presidential, congressional, state, or local)
- Racial or ethnic composition
- Income or socioeconomic indicators
- Age or voter turnout

**Used exclusively:**
- Census block total population (2020 Redistricting File, PL 94-171) — *headcount only*
- Precinct boundary geometry (Census TIGER 2020 VTDs)
- County, place, and county-subdivision boundary geometry (Census TIGER 2020)
- Primary road geometry (Census TIGER 2020)

The word "blind" is substantive, not rhetorical. Party registration and race data exist in the same TIGER/census download package; this pipeline downloads only the files listed above and never opens the others.

Post-hoc partisan lean (Section 16) is computed *after* maps are finalized, for transparency only. It influences no selection decision.

---

## 2. Pipeline Overview

The pipeline runs four sequential stages for each state:

```
Stage 1: download_state    → fetch TIGER 2020 shapefiles from Census FTP
Stage 2: build_graph       → construct weighted precinct dual graph
Stage 3: run_sampling      → run weighted ReCom MCMC, record ensemble
Stage 4: select_maps       → apply filters, compute Pareto frontier, extract maps
```

Each stage is deterministic given the same input data and random seed. Outputs are reproducible.

---

## 3. Data Sources and Preprocessing

### 3.1 Atomic geographic units

The smallest geographic unit (precinct) is the **Voting Tabulation District (VTD)** from Census TIGER 2020. For states that do not publish VTDs (California, Hawaii, Missouri, Oregon), census tracts serve as a geometric substitute.

VTDs are the official smallest units used in redistricting nationwide. They are drawn before election results are known for any given cycle and are independent of partisan data.

### 3.2 Population data

Population per precinct is derived from **2020 Census Redistricting Data (PL 94-171)**, delivered via the TIGER tabblock20 shapefile which includes the `POP20` field — total resident population, no demographic breakdown.

Population is aggregated from census blocks to precincts via **centroid-in-polygon spatial join**: each block's centroid is tested for containment in a precinct polygon, and the block's total population is added to that precinct. Blocks whose centroid falls outside all precinct polygons are discarded (typically <0.1% of population, caused by minor topological gaps in boundary data).

### 3.3 Administrative unit geography

Three administrative layers are joined to precincts by centroid-in-polygon:

| Layer | Source | Field retained | Role |
|-------|--------|----------------|------|
| County | TIGER 2020 national county shapefile | GEOID | Edge weight ×10 |
| Incorporated place | TIGER 2020 per-state places | PLACEFP | Edge weight ×5 |
| County subdivision | TIGER 2020 per-state cousub | COUSUBFP | Edge weight ×3 |

These layers encode *administrative community structure*, not demographics. A county boundary is a boundary regardless of who lives on either side.

### 3.4 Primary roads

Primary road centerlines (TIGER 2020 `tl_2020_us_primaryroads.shp`) are used as a weight divisor, reflecting the geographic reality that major roads often serve as natural district boundaries. Roads are identified by proximity to the shared border between precincts (50 m buffer), not by classification of the precincts they pass through.

---

## 4. Precinct Dual Graph Construction

### 4.1 Definition

The **precinct dual graph** G = (V, E) is an undirected weighted graph where:

- **V** = one node per precinct, attributed with `{pop, county_fips, place_fips, cousub_fips, area_m²}`
- **E** = one edge per pair of precincts sharing a border of at least 50 metres

### 4.2 Adjacency detection

For each pair of precincts (i, j) identified as candidate neighbors via a spatial R-tree index (Shapely `STRtree`), the shared border length is computed as:

```
border_len(i, j) = length(boundary(i) ∩ boundary(j))
```

An edge is added to G if and only if `border_len(i, j) ≥ MIN_BORDER_M = 50 m`.

This 50 m threshold serves two purposes:
1. **Eliminates digitization noise.** TIGER polygon boundaries often share a few vertices at corners where precincts nearly (but do not truly) touch. These produce phantom edges with `border_len < 10 m` that have no geographic meaning.
2. **Prevents peninsula connectors.** In K=2 states (e.g. West Virginia), a very short shared border can become the sole connection between two precinct clusters, creating a narrow "bridge" that appears as an isolated pocket on the rendered map. Raising the threshold from the original 1 m to 50 m eliminates these without affecting legitimate adjacencies (typical precinct shared borders are hundreds to thousands of metres).

### 4.3 Connectivity guarantee

After construction, the graph's connected components are checked. If more than one component exists (rare — caused by island precincts or topological gaps in TIGER data), only the largest component is retained and a warning is logged. The largest component always contains ≥ 99.9% of the state population.

### 4.4 Graph sizes

States range from ~180 nodes (Wyoming, K=1, skipped) to ~14,191 nodes (New York). California has ~9,129 nodes. Graphs are stored as Python `networkx.Graph` objects serialised with `pickle`.

---

## 5. Edge Weight Formula

Each edge (u, v) receives a scalar weight that encodes the *strength of the community bond* between the two precincts. Higher weight = stronger bond = less likely to be severed by a district boundary.

### 5.1 Formula

```
w(u, v) = base
         × W_SAME_COUNTY   if county_fips(u) = county_fips(v)
         × W_SAME_PLACE     if place_fips(u) = place_fips(v)  [both non-null]
         × W_SAME_COUSUB    if cousub_fips(u) = cousub_fips(v) [both non-null]
         ÷ W_ROAD_DIV       if a primary road intersects the 50 m buffer
                             around border(u, v)
```

### 5.2 Parameter values

| Parameter | Symbol | Value | Rationale |
|-----------|--------|-------|-----------|
| Base weight | base | 1.0 | Neutral starting point |
| Same county | W_SAME_COUNTY | 10.0 | Counties are the primary redistricting boundary |
| Same place | W_SAME_PLACE | 5.0 | Incorporated cities/towns should stay whole |
| Same cousub | W_SAME_COUSUB | 3.0 | Township/MCD integrity at sub-county level |
| Road divisor | W_ROAD_DIV | 2.0 | Roads naturally divide communities |

### 5.3 Weight range

The theoretical maximum weight is `1.0 × 10 × 5 × 3 = 150.0` (same county, same city, same township, no road). The minimum is `1.0 / 2.0 = 0.5` (cross-county, cross-city, cross-township, road crosses border). In practice the mean weight is approximately 10–15 for most states.

### 5.4 Why these multipliers are not demographic

All three multipliers depend on which *administrative polygon* a precinct centroid falls within — a purely geometric test. The county boundary is the same regardless of which party dominates it. An incorporated-place boundary is the same regardless of the racial composition of its residents.

A precinct near a city boundary gets the same treatment whether that city voted 80% Democrat or 80% Republican. The weight formula has no knowledge of either.

### 5.5 Beta amplification

During spanning tree sampling, edge weights are raised to the power β (the BETA parameter) before computing transition probabilities:

```
p(traverse edge e) ∝ 1 / w(e)^β
```

With β = 2 and W_SAME_COUNTY = 10:

```
p(cross-county edge)   ∝ 1 / 1²   = 1.000
p(intra-county edge)   ∝ 1 / 10²  = 0.010
```

A cross-county edge is therefore **100× more likely to be traversed** (and thus selected as the cut) than an intra-county edge. β = 0 would produce a fully uniform random spanning tree with no geographic bias.

### 5.6 Adaptive beta for K=2 states

β = 2.0 is appropriate for multi-district states where county structure provides a meaningful guide to natural splits. However, for **K=2 states** (WV, ME, NH, ID, MT, RI, HI), this strong bias causes the MCMC to over-correct:

- With K=2, only a single internal boundary exists. The chain must thread a cut through the entire state.
- At β=2, intra-county edges are penalised 100×, forcing cuts to cross county lines even when the most natural geographic division would cut through a single county.
- The result is a chain that concentrates on a narrow set of plans, producing narrow "peninsula" connectors between precinct clusters.

For K≤2 states, **β is set to 0.5**, giving only a 1.4× preference for cross-county cuts:

```
p(cross-county edge)   ∝ 1 / 1^0.5   = 1.000
p(intra-county edge)   ∝ 1 / 10^0.5  = 0.316
```

This allows the MCMC to explore a broader set of geographically natural splits — including plans that cut through a single county where that county spans the geographic midpoint of the state. West Virginia's β=0.5 run reduced county splits from 5 to 3 and eliminated all peninsula connectors (see [STATE_NOTES.md](STATE_NOTES.md)).

---

## 6. The MCMC Sampler: Weighted ReCom

### 6.1 Background

ReCom (Recombination) is a Markov chain Monte Carlo method for sampling redistricting plans, introduced by DeFord, Duchin & Solomon (2021). It is the standard algorithm used by academic redistricting researchers and has been applied in expert testimony in redistricting litigation.

This implementation uses **Weighted ReCom** — a variant where the random spanning tree step is biased by edge weights to favour administratively coherent splits.

### 6.2 State space

The state space Ω is the set of all valid K-partitions of the precinct graph such that:
1. Every part (district) induces a connected subgraph of G
2. Every part's total population P_d satisfies |P_d − P̄| / P̄ ≤ ε = 0.005

where P̄ = total_population / K is the ideal district population.

### 6.3 One MCMC step

Given current partition π, one ReCom step produces a new partition π' as follows:

**Step 1 — Select a boundary edge:**
Sample uniformly at random from all edges (u, v) ∈ E such that district(u) ≠ district(v). Call the two districts D_a and D_b.

**Step 2 — Form the merged region:**
Let R = D_a ∪ D_b (the set of precincts in either district). Construct the induced subgraph G[R].

**Step 3 — Sample a random spanning tree:**
Draw a spanning tree T of G[R] using the weighted version of Wilson's loop-erased random walk (Section 7). Edge transition probabilities are proportional to 1/w^β.

**Step 4 — Find valid cuts:**
For each edge e ∈ T, temporarily remove e and check whether the resulting two connected components C₁, C₂ satisfy the population balance constraint:

```
P̄(1 − ε) ≤ pop(C₁) ≤ P̄(1 + ε)
P̄(1 − ε) ≤ pop(C₂) ≤ P̄(1 + ε)
```

Collect all valid cuts into set Λ(T).

**Step 5 — Select or reject:**
- If Λ(T) = ∅: **rejection step** — return π unchanged. The chain stays at the current state.
- If Λ(T) ≠ ∅: sample one cut uniformly at random from Λ(T). Assign nodes in C₁ to D_a and nodes in C₂ to D_b. Return new partition π'.

Every district other than D_a and D_b is unchanged.

### 6.4 Why rejection sampling is correct

The rejection step is not a flaw — it is essential to maintaining the correct stationary distribution. A step that always accepts would oversample plans with many valid cuts and undersample those with few. By rejecting when no valid cut exists, the chain preserves detailed balance with respect to the target distribution.

### 6.5 Stationary distribution

The unweighted ReCom chain (β = 0) has a stationary distribution that is approximately uniform over valid plans. The weighted chain (β > 0) tilts the distribution toward plans that respect administrative boundaries — an intentional and disclosed design choice. The key guarantee is that **every valid plan in Ω has positive probability of being reached** in a finite number of steps, ensuring the chain is irreducible and aperiodic (ergodic).

---

## 7. Random Spanning Tree: Wilson's Algorithm

The spanning tree in each ReCom step is sampled via **Wilson's Loop-Erased Random Walk** (Wilson 1996), a classical algorithm that samples uniformly from the set of spanning trees of a graph — or, with weighted transition probabilities, from the weighted distribution T ∝ ∏_{e ∈ T} w(e)^{-β}.

### 7.1 Algorithm

```
Input: connected graph H, weight function w, exponent β
Output: spanning tree T of H

1. Pick any node r as root. Mark r as "in tree."
2. For each node v not yet in tree:
   a. Start a random walk from v.
   b. At current node u, choose next node n with probability:
        p(u → n) = (1/w(u,n)^β) / Σ_{neighbors m of u} (1/w(u,m)^β)
   c. If n is already visited in this walk (loop formed):
        erase the loop (remove all nodes after n from the walk path)
   d. If n is already in tree:
        add the path from v to n to T; mark all path nodes as "in tree"
3. Return T
```

### 7.2 Correctness

Wilson (1996) proved that this algorithm samples spanning trees with probability proportional to the product of their edge weights under the original graph's weights — and with edge-flip to 1/w^β, it samples proportionally to ∏_{e ∈ T} (1/w(e)^β). This is exactly the distribution we want: trees that avoid high-weight (community-preserving) edges are more likely, making it more probable that the eventual cut falls along a community boundary.

### 7.3 Loop erasure preserves the distribution

The loop erasure step (step 2c) is not an approximation — it is mathematically equivalent to running an independent walk from the current node. The resulting path is a loop-erased random walk, whose distribution is identical to that of a simple random walk conditioned on visiting nodes in the same order without looping.

### 7.4 Implementation detail

The implementation stores the walk as an ordered list with an index dictionary for O(1) loop detection. When a loop is detected at node n (already in the walk at position idx), all nodes at positions idx+1 and beyond are erased:

```python
if next_node in walk_pos:
    idx = walk_pos[next_node]
    for erased in walk_order[idx + 1:]:
        del walk_pos[erased]
        del path[erased]
    walk_order = walk_order[:idx + 1]
```

This achieves O(N) expected time per tree for random regular graphs (Aldous 1990, Broder 1989), though worst-case time is O(N²) for adversarial graphs. In practice, precinct graphs have short mixing times.

---

## 8. Population Balance and Hard Feasibility

### 8.1 Ideal population

```
P̄ = Σ_{v ∈ V} pop(v)  /  K
```

where pop(v) is the 2020 census total population of precinct v and K is the number of congressional seats allocated to the state under the 118th Congress apportionment.

### 8.2 Tolerance during sampling

Each ReCom step enforces:

```
|pop(D_d) − P̄| / P̄  ≤  POP_TOL = 0.005    ∀ d ∈ {1, …, K}
```

This ±0.5% deviation is the standard threshold used in academic redistricting work and is substantially tighter than the ±5–10% tolerances used in state-level legislative redistricting.

### 8.3 Seed partition tolerance

The initial partition (before any MCMC steps) uses a relaxed tolerance of:

```
SEED_EPSILON = min(4 × POP_TOL, 0.02) = 2%
```

This only applies to the starting point. The chain's burn-in period rapidly drives all district populations to within the strict 0.5% tolerance; any plan recorded in the ensemble that exceeds 0.5% deviation is removed by the hard filter in Stage 4.

The 4× relaxation is necessary for large states (California, K=52; Texas, K=38) where the GerryChain `recursive_tree_part` algorithm's default 10,000-attempt ceiling is routinely exceeded at strict tolerance, causing the initialization to never complete.

### 8.4 Hard filter confirmation

The `pop_dev_max` metric (maximum deviation across all districts) is recorded for every sampled plan. Plans with `pop_dev_max > 0.005` are removed in the hard-filter step before Pareto selection. This eliminates any seed-epsilon contamination from the final maps.

---

## 9. Metrics Computed Per Plan

For every recorded plan, seven metrics are computed via an O(N + E) algorithm — no geometry dissolve, no spatial join.

### 9.1 Polsby-Popper compactness

The Polsby-Popper score for district d is:

```
PP(d) = 4π · Area(d) / Perimeter(d)²
```

Range: (0, 1]. A circle scores 1.0; elongated or fragmented shapes score near 0.

**O(N + E) computation without dissolve:**

Area and perimeter are precomputed per precinct from the GeoDataFrame geometry cache. For each district d:

```
Area(d)      = Σ_{v ∈ D_d} area(v)
Perimeter(d) = Σ_{v ∈ D_d} boundary_len(v)
               − 2 · Σ_{(u,v) ∈ E, u∈D_d, v∈D_d} border_len(u,v)
```

The second term corrects for shared internal borders: when two precincts in the same district share a border, that border contributes to both precincts' individual perimeters but is interior to the district and must be subtracted twice.

**Three PP statistics are recorded:**

| Metric | Definition |
|--------|-----------|
| `pp_min` | min_{d} PP(d) — worst-case district |
| `pp_mean` | arithmetic mean of PP across all K districts |
| `pp_max` | max_{d} PP(d) — best-case district |

### 9.2 Population deviation

```
dev(d) = |pop(D_d) − P̄| / P̄
```

Recorded as `pop_dev_max` = max_{d} dev(d) and `pop_dev_mean` = mean_{d} dev(d).

### 9.3 Cut edges

```
cut_edges = |{(u,v) ∈ E : district(u) ≠ district(v)}|
```

Cut edges count the total number of precinct-pair adjacencies that cross a district boundary. Minimising cut edges is equivalent to maximising within-district connectivity — a measure of compactness independent of shape.

### 9.4 Cut border length

```
cut_border_m(π) = Σ_{(u,v) ∈ E, district(u) ≠ district(v)} border_len(u, v)
```

This is the total length (metres) of all precinct-pair shared borders that cross a district boundary in plan π. It measures how "ragged" the district boundary is:

- A plan whose boundaries follow natural geographic lines (river valleys, ridgelines) typically crosses fewer and longer precinct borders — lower `cut_border_m` relative to the number of cut edges.
- A patchwork plan with many short, jagged boundaries accumulates a high `cut_border_m`.

`cut_border_m` is used as a fifth Pareto objective (see Section 11). It is also used to detect **peninsula connectors**: an edge whose `border_len` is much shorter than the average cut-border length may represent a narrow artificial connection between two precinct clusters rather than a real geographic boundary (see Section 12.3).

### 9.5 County splits (computed at selection time)

County splits are computed for sampled plans at Stage 4, not during sampling, to avoid the per-step overhead.

For each county c and plan π:

```
districts_in_county(c, π) = |{district(v) : county(v) = c, v ∈ V}|
splits(c, π)              = max(0, districts_in_county(c, π) − 1)
```

Total splits for plan π:
```
county_splits(π) = Σ_c splits(c, π)
```

Maximum districts per county:
```
max_county_districts(π) = max_c districts_in_county(c, π)
```

`max_county_districts` was added specifically to prevent the Pareto selection from concentrating fragmentation on one dense urban county (e.g. splitting a single large city into 4 districts while appearing well-behaved on total splits). A plan with one county split into 4 and three counties each split once has `county_splits = 6` and `max_county_districts = 4`; a plan with six counties each split twice also has `county_splits = 6` but `max_county_districts = 2`. The second plan is preferable.

---

## 10. Hard Filters

Before Pareto selection, two filters eliminate plans that fail minimum quality thresholds:

| Filter | Threshold | What it enforces |
|--------|-----------|-----------------|
| `pp_min ≥ 0.10` | 10% Polsby-Popper | No district may be so elongated or gerrymandered-looking that its worst shape is below the 10th-percentile circle |
| `pop_dev_max ≤ 0.005` | ±0.5% | Every district must be within 0.5% of ideal population |

These filters are applied uniformly to all states; no state-specific thresholds exist. Plans failing either filter are discarded before any selection, regardless of how many plans remain.

If zero plans pass the hard filters, the pipeline raises a `RuntimeError` rather than returning a potentially invalid map.

---

## 11. Pareto Frontier Selection

### 11.1 Objectives

From the filtered ensemble, plans are scored on five objectives simultaneously:

| Objective | Direction | Meaning |
|-----------|-----------|---------|
| `pp_mean` | Maximise | Higher Polsby-Popper mean = more compact overall |
| `county_splits` | Minimise | Fewer county splits = more community-intact |
| `cut_edges` | Minimise | Fewer cut edges = more internally connected districts |
| `max_county_districts` | Minimise | Worst-case county fragmentation |
| `cut_border_m` | Minimise | Total cross-district boundary length (raggedness) |

No weights, no aggregation, no trade-off coefficients. The Pareto frontier is the set of plans not dominated on all five dimensions simultaneously.

### 11.2 Dominance

Plan A **dominates** plan B if and only if:

```
pp_mean(A) ≥ pp_mean(B)              AND
county_splits(A) ≤ county_splits(B)        AND
cut_edges(A) ≤ cut_edges(B)               AND
max_county_districts(A) ≤ max_county_districts(B)  AND
cut_border_m(A) ≤ cut_border_m(B)
```

with at least one strict inequality.

### 11.3 Algorithm

```python
# Convert to pure minimisation: negate pp_mean
costs[:, 0] = -pp_mean

for i in range(n):
    dominated[i] = any plan j ≠ i such that:
        all(costs[j] ≤ costs[i]) AND any(costs[j] < costs[i])

frontier = {plans where not dominated[i]}
```

This is an O(n²) scan over the sampled plans. With n ≤ 5,000 sampled plans, runtime is under 30 seconds.

### 11.4 Sampling for Pareto

Up to 5,000 plans are drawn uniformly at random (without replacement) from the filtered ensemble for county-split computation and Pareto analysis. When the filtered ensemble is smaller than 5,000, all plans are used.

The sampling uses `numpy.random.default_rng(RANDOM_SEED).choice()` — seeded, reproducible, and unbiased with respect to any plan characteristic.

### 11.5 Why five objectives and not one

Using a single objective (e.g. maximum compactness) would maximise that one criterion while potentially producing plans with extreme county fragmentation or poor population balance. The Pareto frontier respects the inherent tension between objectives without assigning a subjective weighting to any one of them.

Importantly, no partisan or demographic criterion appears in any of the five objectives. The Pareto frontier is blind to all political outcomes.

---

## 12. Selecting Two Representative Plans

From the Pareto frontier, two plans are extracted:

### 12.1 Best compact

Selection rule (applied in order):
1. Find the minimum value of `max_county_districts` across all frontier plans.
2. Restrict to the subset of plans achieving that minimum.
3. Within this subset, select the plan with the highest `pp_mean`.

### 12.2 Fewest county splits

Selection rule (applied in order):
1. Same minimum-`max_county_districts` restriction as above.
2. Within this subset, sort by `county_splits` ascending, then `pp_mean` descending.
3. Select the first plan.

### 12.3 Peninsula filter

Before the county-fragmentation tier, a **peninsula filter** is applied to reject plans with anomalously short cut-edge borders:

```
mean_cut_border(π) = cut_border_m(π) / cut_edges(π)

peninsula_ratio(π) = min_{e cut} border_len(e) / mean_cut_border(π)
```

Plans with `peninsula_ratio < PENINSULA_RATIO_THRESHOLD = 0.05` are deprioritised (the full frontier is used only if no plan passes this threshold).

**What this detects:** If a plan's shortest inter-district border is less than 5% of the average inter-district border, one precinct-pair adjacency is doing disproportionate work connecting two otherwise-separate precinct clusters. On a rendered map, this appears as an isolated district "pocket" connected to the rest of its district by a narrow corridor. These plans can score well on Polsby-Popper (the dissolved district polygon may look compact) while visually appearing fragmented.

This is especially relevant for K=2 states where a single cut must span the entire state.

### 12.4 Why filter by max_county_districts first

This two-stage selection prevents a pathological case: a plan that achieves marginally higher compactness by concentrating all county fragmentation on a single dense urban county. Without this filter, a plan with Baltimore City in 4 districts could appear on the Pareto frontier if it had slightly better PP mean than a plan with Baltimore City in 2 districts, because total county splits might be identical. The `max_county_districts` pre-filter eliminates this class of plans before the primary objective comparison.

---

## 13. MCMC Step Count Scaling

### 13.1 Scaling rule

The number of recorded MCMC steps scales linearly with the number of districts:

```
steps(state) = max(BASE_STEPS, round(BASE_STEPS × K / K_MD))
```

where BASE_STEPS = 2,000 and K_MD = 8 (Maryland's K).

| State | K | Steps |
|-------|---|-------|
| Maryland (baseline) | 8 | 2,000 |
| Tennessee | 9 | 2,250 |
| Virginia | 11 | 2,750 |
| Ohio | 15 | 3,750 |
| New York | 26 | 6,500 (100k override) |
| California | 52 | 13,000 (100k override) |

Large states (CA, NY, TX, FL) are run with an explicit 100,000-step override, reflecting that larger K requires more chain mixing to explore the plan space adequately.

### 13.2 Theoretical justification

The mixing time of ReCom scales at least linearly with K: each step modifies exactly 2 of K districts, so covering the full plan space requires O(K) steps per effective sample. The linear scaling rule is a conservative lower bound, not a guarantee of full mixing. For states with K ≤ 17 at 2,000 steps, the chain is expected to explore the space adequately for selection purposes; the Pareto diversity and hard filters provide additional quality control.

---

## 14. Burn-In and Chain Initialization

### 14.1 Initial partition

The starting partition is generated by GerryChain's `recursive_tree_part`, which recursively bipartitions the graph using random spanning trees. It produces a valid connected K-partition with population balance within `SEED_EPSILON = 2%`.

`max_attempts=None` is passed to the bipartition subroutine, removing the default attempt ceiling that causes timeouts for large K.

### 14.2 Burn-in

500 ReCom steps are run before recording begins. These steps are not saved. Their purpose is to move the chain away from the structured initial partition (which has systematic biases from the recursive construction algorithm) toward the bulk of the stationary distribution.

Plans recorded after burn-in are used for all downstream analysis. The burn-in count (500) was chosen conservatively; empirically, ReCom mixes fast relative to other redistricting chains (DeFord et al. 2021, Section 5).

### 14.3 Checkpoint resume

After every 1,000 recorded steps, the pipeline atomically saves:
- Current district assignment (full node-to-district map)
- Complete NumPy RNG state (PCG64 bit generator state vector)
- Step count and chunk ID

The save is atomic (write to `.tmp`, then `os.replace()`) so a kill signal mid-write leaves the previous checkpoint intact. On restart, the pipeline resumes from the last checkpoint without re-running burn-in.

---

## 15. Reproducibility and Determinism

### 15.1 Seeds

All random number generation uses `numpy.random.default_rng(RANDOM_SEED)` with `RANDOM_SEED = 42`. The PCG64 generator is seeded before the initial partition, burn-in, and sampling loop.

### 15.2 Full reproducibility conditions

A run is exactly reproducible if:
1. The same TIGER 2020 shapefiles are used (fixed release, not updated)
2. The same `RANDOM_SEED = 42` is used
3. No checkpoint resume is triggered (checkpoint resumes restore the exact RNG state, but floating-point timing of OS preemptions can cause different resume points)
4. The same Python, NumPy, and Shapely versions are used

The output parquet files and final GeoPackages are fully reproducible under these conditions.

### 15.3 What is NOT random

- Edge weight formula (deterministic from geography)
- Population joins (deterministic spatial join)
- Hard filter thresholds (fixed constants)
- Pareto dominance check (deterministic)
- Plan selection from frontier (deterministic, no randomness)

---

## 16. Post-Hoc Partisan Lean (Strictly Separate)

### 16.1 What it is

After maps are finalized, a partisan lean label is computed for display purposes only using **VEST 2020 precinct-level election results** (Voting and Election Science Team, Harvard Dataverse).

### 16.2 Metric

```
Biden%(district d) = Σ_{v ∈ D_d} G20PREDBID(v)
                    ─────────────────────────────────────────────────
                    Σ_{v ∈ D_d} (G20PREDBID(v) + G20PRERTRU(v))
```

where `G20PREDBID` and `G20PRERTRU` are the 2020 presidential vote totals (Biden and Trump respectively) for each VEST precinct. Precincts are joined to districts by centroid-within-polygon to avoid double-counting precincts that straddle a district boundary.

Margin label: `D+{round(|Biden% − 50%| × 200)}` if Biden% ≥ 50%, else `R+{...}`.

### 16.3 Strict isolation guarantees

The function `district_lean()` is defined in `pipeline/lean.py`, which is not imported by `pipeline/sample.py` or `pipeline/select.py`. There is no code path by which the VEST data can influence the MCMC sampler, the hard filters, the Pareto frontier, or the plan selection. A code review of the import graph confirms:

```
pipeline/sample.py  → imports: numpy, networkx, geopandas, pyarrow (NO lean)
pipeline/select.py  → imports: numpy, geopandas, pandas, pyarrow (NO lean)
pipeline/build_graph.py → imports: geopandas, networkx, shapely (NO lean)
```

### 16.4 Why show it at all

Displaying post-hoc lean answers the obvious question — "what would these districts produce politically?" — without allowing that question to influence the answer. Transparency about outcomes is compatible with, and indeed strengthens, the claim of geographic-only design.

---

## 17. What the Algorithm Cannot Do

### 17.1 Cannot gerrymander for a party

The sampler has no representation of votes, registration, or party. It cannot construct a plan that concentrates opposition voters or cracks a majority-minority coalition. The only signal available to it is the shape and adjacency of census geography.

### 17.2 Cannot deliberately target or avoid majority-minority districts

The algorithm draws no distinction between a precinct that voted 90% for one party and one that voted 10%. It does not know whether two adjacent precincts share a racial or linguistic community. A majority-minority district may or may not appear in a sampled plan — this is a consequence of the geographic distribution of communities, not a design choice.

**Important limitation:** The algorithm provides no affirmative guarantee of VRA compliance. Majority-minority districts may or may not appear in the output maps. Post-hoc VRA audit by a qualified attorney is necessary before any map produced by this pipeline is used for official purposes.

### 17.3 Cannot guarantee optimality

The Pareto frontier is a sample from the stationary distribution, not an exhaustive enumeration. The two selected plans are representative, not provably optimal across all possible plans for the state. Different random seeds or longer chains may produce modestly different results.

### 17.4 Cannot replace a legal redistricting process

Maps produced by this pipeline are illustrative. Real congressional redistricting requires legislative action, public comment periods, court review, and compliance with federal and state law. This pipeline produces candidate maps for analytical comparison, not certified legal district boundaries.

---

## 17b. Map Rendering and VTD Coverage

### 17b.1 What is rendered

District maps are rendered directly from the GeoPackage produced by the selection step. Each district polygon is the dissolved union of all precincts assigned to that district. Rendering uses `geopandas.GeoDataFrame.plot()` with per-district colours, optionally overlaid with county boundary lines.

### 17b.2 VTD coverage gap

VTD (Voting Tabulation District) shapefiles do not always tile the full state geography. Three states have material gaps:

| State | VTD coverage | Gap source |
|-------|-------------|------------|
| Hawaii | 10.4% | Pacific Ocean within county boundaries |
| Maine | 14.6% | Gulf of Maine within county boundaries |
| West Virginia | 72.3% | Federal land (national forests, wilderness areas) |

All other 41 multi-district states are at ≥92.9% coverage; 38 are at exactly 100%.

For Hawaii and Maine the gap is ocean — the TIGER county boundaries extend to territorial waters, so the county geometry is much larger than the land area. VTDs only exist on inhabited land. On a rendered map these gaps are correctly perceived as water; no intervention is needed.

For West Virginia the gap is **inland federal land**. Areas such as the Monongahela National Forest (900,000+ acres) and adjacent wilderness tracts have no VTD assignment. When only the precinct-derived district polygons are drawn, these interior areas render as the dark background colour, producing visible "holes" in what should be a solid two-colour map.

### 17b.3 State-outline base layer

`render_all_state_comparisons.py` addresses the coverage gap by drawing a dissolved state outline as a filled base layer before drawing district polygons:

```python
from shapely.ops import unary_union
state_outline = gpd.GeoDataFrame(
    geometry=[unary_union(state_counties.geometry)], crs=crs
)
state_outline.plot(ax=ax, color="#2a3140", edgecolor="none")
# … then draw district polygons on top
```

The base layer colour (`#2a3140`) is a neutral gray that sits between the dark background and the district colours, so unpopulated gaps are clearly within the state footprint but visually distinct from any district. This fix is applied uniformly to all 44 states; it has no visible effect on states with 100% VTD coverage.

### 17b.4 Enacted-map placeholder districts

The 118th Congress enacted shapefiles (TIGER `tl_*.shp`) include a `CD118FP = 'ZZ'` placeholder record for the non-voting delegate districts of Washington DC, Puerto Rico, and the US territories. These records have string district codes and no valid geometry for comparison purposes. The renderer filters them before display:

```python
if "CD118FP" in gdf.columns:
    gdf = gdf[gdf["CD118FP"].str.match(r"^\d+$", na=False)].copy()
```

Without this filter, states such as Connecticut, Illinois, and New Hampshire (which share TIGER files with adjacent territories) raise a `ValueError` when the code attempts `int('ZZ')` to assign a colour.

---

## 18. Parameter Summary

| Parameter | Value | Location | Effect |
|-----------|-------|----------|--------|
| `BETA` | 2.0 | `sample.py` | Exponential weight amplification for K>2 states; 100× county-crossing preference |
| `BETA_K2` | 0.5 | `sample.py` | Adaptive β for K≤2 states; 1.4× county-crossing preference (prevents peninsula connectors) |
| `RANDOM_SEED` | 42 | `sample.py` | PCG64 RNG seed |
| `POP_TOL` | 0.005 | `sample.py` | ±0.5% population balance tolerance |
| `BURN_IN` | 500 | `sample.py` | Warm-up steps before recording |
| `FLUSH_EVERY` | 1,000 | `sample.py` | Chunk size for streaming writes |
| `SEED_EPSILON` | 0.02 | `sample.py` | Initial partition tolerance (2%) |
| `MIN_BORDER_M` | 50.0 | `build_graph.py` | Minimum shared border (m) to register adjacency; eliminates digitization noise and prevents peninsula connectors |
| `W_SAME_COUNTY` | 10.0 | `build_graph.py` | County-crossing weight multiplier |
| `W_SAME_PLACE` | 5.0 | `build_graph.py` | City-crossing weight multiplier |
| `W_SAME_COUSUB` | 3.0 | `build_graph.py` | Township-crossing weight multiplier |
| `W_ROAD_DIV` | 2.0 | `build_graph.py` | Primary road weight divisor |
| `ROAD_BUFFER_M` | 50.0 | `build_graph.py` | Buffer for road proximity check |
| `PP_MIN_THRESHOLD` | 0.05 | `select.py` | Hard filter: minimum PP for worst district |
| `POP_DEV_MAX_THRESHOLD` | 0.005 | `select.py` | Hard filter: maximum population deviation |
| `PENINSULA_RATIO_THRESHOLD` | 0.05 | `select.py` | Minimum ratio of shortest to mean cut-border length; plans below are deprioritised |
| `SAMPLE_SIZE` | 5,000 | `select.py` | Plans sampled for Pareto computation |
| `BASE_STEPS` | 2,000 | `run_all_states.py` | Maryland baseline step count |
| `K_MD_BASELINE` | 8 | `run_all_states.py` | Maryland K for step scaling |

---

## 19. References

**Algorithm:**

- DeFord, D., Duchin, M., & Solomon, J. (2021). Recombination: A family of Markov chains for redistricting. *Harvard Data Science Review*, 3(1). https://doi.org/10.1162/99608f92.eb30390f

**Spanning tree sampling:**

- Wilson, D. B. (1996). Generating random spanning trees more quickly than the cover time. *Proceedings of the 28th Annual ACM Symposium on Theory of Computing*, 296–303.
- Aldous, D. (1990). The random walk construction of uniform spanning trees and uniform labelled trees. *SIAM Journal on Discrete Mathematics*, 3(4), 450–465.
- Broder, A. (1989). Generating random spanning trees. *Proceedings of the 30th Annual Symposium on Foundations of Computer Science*, 442–447.

**Compactness:**

- Polsby, D. D., & Popper, R. D. (1991). The third criterion: Compactness as a procedural safeguard against partisan gerrymandering. *Yale Law & Policy Review*, 9(2), 301–353.

**Population data:**

- US Census Bureau. (2021). 2020 Census Redistricting Data (Public Law 94-171) Summary File. https://www.census.gov/programs-surveys/decennial-census/about/rdo/summary-files.html

**Geographic data:**

- US Census Bureau. (2020). TIGER/Line Shapefiles. https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

**Partisan lean (post-hoc only):**

- Voting and Election Science Team. (2021). 2020 Precinct-Level Election Results. Harvard Dataverse. https://doi.org/10.7910/DVN/K7760H

**Implementation:**

- Metric Geometry and Gerrymandering Group. GerryChain (v0.3.1+). https://github.com/mggg/GerryChain
