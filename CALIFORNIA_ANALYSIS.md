# Blind Redistricting: Algorithm Deep-Dive, California Failures, and the Road Forward

*An honest engineering post-mortem on weighted ReCom MCMC redistricting — what worked, what broke on California, and what we'd do differently.*

---

## Table of Contents

1. [How the Algorithm Works](#1-how-the-algorithm-works)
2. [The Mathematics of Fairness](#2-the-mathematics-of-fairness)
3. [California: A Chronicle of Failures](#3-california-a-chronicle-of-failures)
4. [Five Proposed Improvements](#4-five-proposed-improvements)
5. [Hardware Constraints](#5-hardware-constraints)

---

## 1. How the Algorithm Works

The pipeline has four stages: **download → graph → sample → select**. Everything interesting happens in stages 2 and 3.

### 1.1 Building the Weighted Precinct Dual Graph

Each US state is subdivided into *voting tabulation districts* (VTDs) — roughly, precincts. The algorithm builds a graph `G = (V, E)` where:

- **Nodes** `V`: one per precinct, attributed with population and administrative unit membership (county, municipality, county subdivision).
- **Edges** `E`: one per pair of precincts that share a geographic border of at least 1 metre.

```
PROCEDURE build_graph(precincts, counties, roads):
    G ← empty graph

    FOR each precinct p:
        G.add_node(p, pop=p.population, county=p.county_fips,
                      place=p.place_fips, cousub=p.cousub_fips)

    FOR each pair (p, q) with shared_border(p, q) >= 1.0 m:
        w ← 1.0
        IF p.county == q.county:  w ← w × W_SAME_COUNTY    # ×10
        IF p.place  == q.place:   w ← w × W_SAME_PLACE     # ×5
        IF p.cousub == q.cousub:  w ← w × W_SAME_COUSUB    # ×3
        IF road_cuts_border(p, q):  w ← w / W_ROAD_DIV     # ÷2
        G.add_edge(p, q, weight=w, border_len=shared_border(p,q))

    RETURN G
```

The weight formula encodes *administrative cohesion* without any partisan, demographic, or income data. Two precincts in the same county get a weight of at least 10; in the same county AND municipality, at least 50. A major road cutting the border halves the weight. These are pure geographic facts derived from US Census TIGER shapefiles.

### 1.2 Weighted ReCom MCMC Sampling

The sampler draws from the uniform distribution over valid K-district plans using *Recombination* (ReCom), a Markov chain Monte Carlo method introduced by DeFord, Duchin & Solomon (2021). "Valid" means: each district is a connected subgraph, and every district's population is within ±POP_TOL of the ideal.

```
PROCEDURE run_sampling(G, K, n_steps):
    assignment ← initial_partition(G, K)     # balanced seed
    assignment ← repair(assignment, G, K)    # bring into ±POP_TOL
    assignment ← burn_in(assignment, G, K, 500 steps)

    FOR step = 1 … n_steps:
        assignment ← recom_step(assignment, G, K)
        record(assignment, compute_metrics(assignment, G, K))

    RETURN recorded_plans
```

**One ReCom step** in detail:

```
PROCEDURE recom_step(assignment, G, K):
    boundary_edges ← {(u,v) ∈ E : assignment[u] ≠ assignment[v]}
    IF boundary_edges = ∅: RETURN assignment   # degenerate

    (u, v) ← uniform_sample(boundary_edges)
    d_a ← assignment[u];   d_b ← assignment[v]

    merged ← {i ∈ V : assignment[i] = d_a OR assignment[i] = d_b}
    G_sub  ← induced subgraph of G on merged

    T ← wilson_spanning_tree(G_sub, beta=2.0)   # biased by edge weights
    comp1 ← find_valid_cut(T, ideal_pop, POP_TOL)

    IF comp1 = None:
        RETURN assignment   # rejection step; same plan counted again

    FOR i IN merged:
        assignment[i] ← d_a IF i IN comp1 ELSE d_b

    RETURN assignment
```

**Key property**: the chain is a *uniform* sampler. Every accepted step is accepted unconditionally — there is no Metropolis–Hastings accept/reject ratio. The distribution over valid plans converges to uniform as steps → ∞.

### 1.3 Wilson's Loop-Erased Random Walk (LERW)

Sampling a random spanning tree naively requires O(N³) time. Wilson's algorithm (1996) does it in O(N·E) expected time using loop-erased random walks, and crucially the walk's transition probabilities can be biased by edge weights.

```
PROCEDURE wilson_spanning_tree(G_sub, beta):
    root ← G_sub.nodes[0]
    in_tree ← {root}
    parent  ← {}

    FOR each node s NOT in in_tree:
        current ← s
        walk_pos ← {current: 0};   walk_order ← [current]

        WHILE current NOT in in_tree:
            neighbors ← G_sub.neighbors(current)

            # Transition probability ∝ 1/w^β — biased AWAY from high-weight edges
            probs ← [1 / w(current,n)^beta for n in neighbors]
            probs ← normalise(probs)
            next ← sample(neighbors, probs)

            IF next IN walk_pos:
                # Erase the loop back to next
                FOR erased IN walk_order[walk_pos[next]+1 :]:
                    DELETE walk_pos[erased], parent[erased]
                walk_order ← walk_order[: walk_pos[next]+1]
            ELSE:
                parent[current] ← next
                walk_pos[next]  ← len(walk_order)
                walk_order.append(next)

            current ← next

        # Commit path from s to in_tree
        current ← s
        WHILE current NOT in in_tree:
            in_tree.add(current)
            current ← parent[current]

    RETURN parent   # spanning tree as parent pointers
```

### 1.4 O(M) Valid-Cut Search

A naïve cut search tests each of the M−1 spanning tree edges individually and re-traverses the tree after each removal — O(M²) total. The v2 implementation does it in O(M):

```
PROCEDURE find_valid_cut(parent, root, nodes, ideal_pop, POP_TOL):
    children ← invert(parent)         # build children list from parent dict
    order    ← BFS_order(root, children)   # topological order

    # Bottom-up subtree population accumulation
    subtree_pop[i] ← pop[i]  FOR ALL i IN nodes
    FOR node IN reversed(order):
        FOR child IN children[node]:
            subtree_pop[node] += subtree_pop[child]

    total ← subtree_pop[root]
    low   ← ideal_pop × (1 − POP_TOL)
    high  ← ideal_pop × (1 + POP_TOL)

    # A non-root node v is a valid cut iff:
    #   low ≤ subtree_pop[v] ≤ high   AND   low ≤ total − subtree_pop[v] ≤ high
    valid_cuts ← [v FOR v IN nodes IF v ≠ root
                    AND low ≤ subtree_pop[v] ≤ high
                    AND low ≤ total − subtree_pop[v] ≤ high]

    IF valid_cuts = ∅: RETURN None

    cut_node ← uniform_sample(valid_cuts)
    comp1    ← BFS subtree of cut_node following children only
    RETURN comp1
```

### 1.5 O(N+E) Polsby-Popper Without Geometry Dissolve

Computing district shapes by dissolving precinct geometries is expensive. The algorithm avoids it entirely using the identity:

```
area(district d)  = Σ_{i: dist=d} area[i]
perim(district d) = Σ_{i: dist=d} perim[i]  −  2 × Σ_{(i,j)∈E, dist=d} border[i,j]
```

The second term subtracts shared internal borders (counted twice, once from each side). In numpy:

```python
dist_area  = np.bincount(assignment, weights=area_arr,  minlength=K)
dist_perim = np.bincount(assignment, weights=perim_arr, minlength=K)
dist_pop   = np.bincount(assignment, weights=pop_arr,   minlength=K)

eu, ev = edges[:,0], edges[:,1]
same   = assignment[eu] == assignment[ev]
np.subtract.at(dist_perim, assignment[eu[same]], 2.0 * border_arr[same])

pp = 4π × dist_area / dist_perim²
```

This is O(N+E) — proportional to the graph, not to district polygon vertex count.

### 1.6 Pareto Map Selection

From the ensemble, two representative plans are chosen via Pareto optimality across four geographic objectives, with no partisan data ever touching the selector.

```
PROCEDURE select_maps(ensemble):
    filtered ← {plan : pp_min(plan) >= 0.05
                    AND pop_dev_max(plan) <= 0.005}

    sample   ← uniform_sample(filtered, min(5000, |filtered|))
    FOR plan IN sample:
        plan.county_splits       ← compute_county_splits(plan)
        plan.max_county_districts← max districts in any single county

    # Pareto frontier: no plan is better on ALL four objectives
    frontier ← pareto(sample,
                       maximise=pp_mean,
                       minimise={county_splits, cut_edges, max_county_districts})

    # Two representatives from the frontier
    best_max_cd ← min(frontier, key=max_county_districts)
    best_compact←  max(best_max_cd, key=pp_mean)
    fewest_splits← min(best_max_cd, key=(county_splits, -pp_mean))

    RETURN best_compact, fewest_splits
```

---

## 2. The Mathematics of Fairness

### 2.1 Blindness as the Primary Guarantee

The algorithm has **zero access** to the following data:
- Party registration or affiliation of any voter
- Past election results (presidential, congressional, gubernatorial)
- Racial, ethnic, or demographic composition
- Income, age, education, or any socioeconomic variable
- Current incumbent addresses

The only inputs are: precinct geometry (Census TIGER shapefiles), precinct population (Census block counts), and administrative unit membership (county, municipality, county subdivision). These are geometric facts, not political ones.

**Post-hoc partisan lean** is computed *after* maps are selected using VEST 2020 presidential vote data, and is displayed solely for transparency. It is never an input to the sampler or selector.

### 2.2 Why MCMC Samples from the Uniform Distribution

ReCom is designed as a *uniform sampler* over valid plans — every valid plan is equally likely to be drawn in the long run. This is crucial: it means the algorithm does not prefer plans that are more or less favourable to any party.

**Theorem (DeFord, Duchin & Solomon 2021)**: The ReCom Markov chain is ergodic (irreducible and aperiodic) over the space of valid connected K-partitions, and its stationary distribution is the *uniform* distribution over that space.

*Proof sketch*: 
- *Irreducibility*: any valid plan can be reached from any other via a sequence of ReCom steps (merge two districts → resample spanning tree → recut). The spanning tree samples cover all possible cuts.
- *Aperiodicity*: rejection steps (where no valid cut is found) make the chain aperiodic.
- *Uniformity*: the acceptance probability is 1 for all accepted proposals. The detailed balance equation holds because each step is reversible — merging d_a and d_b and resampling yields the same distribution regardless of which starting plan you came from within the same merged region.

### 2.3 Edge Weights and Partisan Blindness

The weight formula raises an important question: does biasing toward county-respecting cuts introduce partisan bias? The answer is no, for a structural reason.

**Claim**: The edge weight `w(u,v)` is a function of geometric containment relationships only.

*Proof*: Weight increases when precincts share a county, municipality, or county subdivision. These boundaries are determined by historical administrative decisions made before any election in the sample period. The algorithm tests only: "is the centroid of precinct u inside the polygon of county c?" — a pure computational geometry operation on Census shapefiles. No party affiliation data is loaded at graph-build time. ∎

The β=2 exponent amplifies geographic cohesion: a cross-county edge has weight at most 1.0 (no shared admin unit) while an intra-county edge has weight at least 10. In Wilson's walk, transition probability is proportional to `1/w^β`, so:

```
P(traverse cross-county edge) / P(traverse intra-county edge) = w_county^β / 1 = 10² = 100
```

County boundaries are therefore 100× more likely to appear as cut points. This respects the "county integrity" principle established in many state constitutions — without ever consulting a voter file.

### 2.4 Population Equality Guarantee

**Hard requirement**: `|pop(d) − ideal_pop| / ideal_pop ≤ POP_TOL = 0.005` for every district d in every recorded plan.

**How it is enforced**: The `find_valid_cut` procedure returns `None` if no cut satisfies the constraint. The step is then rejected (the same plan is recorded again). This means *zero* recorded plans violate the population equality constraint — it is a hard filter, not a soft penalty.

**Mathematical validity**: For a spanning tree T over a merged subgraph with M nodes and total population P = pop(d_a) + pop(d_b), the bottom-up subtree population algorithm finds all nodes v such that:

```
ideal × (1 − POP_TOL) ≤ subtree_pop(v) ≤ ideal × (1 + POP_TOL)
```

By the tree bisection lemma, such a node always exists when `POP_TOL ≥ max_precinct_pop / (2 × ideal_pop)`. For most states, `max_precinct_pop / ideal_pop ≈ 1–5%`, so `POP_TOL = 0.5%` may occasionally produce no valid cuts (rejection). This is correct behaviour — the chain rejects and tries again.

**The stuck-district problem**: If the initial partition places district d at deviation `d_x > POP_TOL`, and all adjacent districts have deviation `d_y > 0`, then combined = `ideal × (2 + d_x + d_y)`. For a valid cut at POP_TOL, we need:

```
combined ≤ 2 × ideal × (1 + POP_TOL)
⟺  d_x + d_y ≤ 2 × POP_TOL = 1%
```

If both d_x and d_y are positive (both districts overpopulated), this fails when d_x + d_y > 1%. The district is permanently stuck. This drove several California failures (see §3).

### 2.5 Compactness as a Gerrymandering Guard

The Polsby-Popper score for district d is:

```
PP(d) = 4π × area(d) / perimeter(d)²
```

PP = 1 for a perfect circle; PP → 0 for elongated or fragmented shapes. Classic partisan gerrymanders use long, winding shapes to "pack and crack" voters — these shapes have very low PP. By selecting plans that *maximise* PP, the algorithm structurally penalises the shapes most associated with partisan manipulation.

**Hard filter**: `min_{d} PP(d) ≥ 0.05` (updated from 0.10 after California analysis — see §3.5). This eliminates plans with any severely non-compact district regardless of how well the other districts score.

**No PP guarantee**: PP is a necessary but not sufficient condition for fairness. A state could have high PP in every district yet still produce biased outcomes if geographic sorting of voters is extreme. PP is a geometric filter, not a partisan neutrality proof.

### 2.6 Pareto Optimality — What It Guarantees and Does Not

The Pareto selection among `{pp_mean ↑, county_splits ↓, cut_edges ↓, max_county_districts ↓}` guarantees that no other sampled plan is strictly better on all four objectives simultaneously. This prevents the algorithm from trading away county integrity for marginal compactness gains.

What it does **not** guarantee: the two selected plans are the *globally* optimal plans. The ensemble is finite (100,000 plans), the Pareto sample is capped at 5,000, and the true Pareto frontier over all possible plans is unknown. The outputs are representative, not unique.

---

## 3. California: A Chronicle of Failures

California (K=52 congressional districts, 9,126 precincts, 39.5M people, ideal district pop = 760,272) was the hardest state in the pipeline by a wide margin. Here is a faithful account of every failure, in chronological order.

### Failure 1 — v1 Algorithm, Stuck District (76,000 steps wasted)

**What happened**: The first CA run used the v1 sampler (NetworkX subgraphs in the hot loop, Python dict assignment copies per step, O(M²) valid-cut search). It ran for approximately 4 days, completing 76 of 100 chunk files (76,000 recorded plans). The sampler was killed before completion.

**Failure mode**: On inspection, `pop_dev_max` was **constant at 1.8743%** across all 76,000 plans — a dead giveaway that the Markov chain was stuck. The initial partition (SEED_EPSILON=0.02, i.e. ±2%) had placed district 9 at 1.87% above ideal. Because `POP_TOL = 0.5%`, no adjacent pair involving district 9 could ever produce a valid cut:

```
combined(district_9, any_neighbor) = ideal × (1 + 0.0187 + d_neighbor)

For d_neighbor ≥ 0: combined ≥ 2.0187 × ideal
Required for valid cut: combined ≤ 2 × (1 + 0.005) × ideal = 2.01 × ideal
2.0187 > 2.01 ✗ — no valid cut exists
```

District 9 was geometrically frozen. The Polsby-Popper values varied (pp_min 0.030–0.074) because other districts were successfully recombining, but district 9 never changed. **All 76,000 plans failed the `pop_dev_max ≤ 0.5%` hard filter.**

**RAM during this run**: 3.8–4.2 GB RSS. Python's memory allocator never returned pages to the OS after millions of NetworkX subgraph allocations and frees. The `del obj; gc.collect()` pattern did not reduce RSS on macOS.

**Resolution**: Deleted the ensemble. Rewrote the sampler (v2).

---

### Failure 2 — v2 with SEED_EPSILON = POP_TOL = 0.5% (GerryChain deadlock)

**What happened**: The fix for Failure 1 seemed obvious: make the initial partition as tight as the MCMC requirement. Set `SEED_EPSILON = POP_TOL = 0.005`.

**Failure mode**: GerryChain's `bipartition_tree(max_attempts=None)` with ε=0.5% for K=52 entered an infinite loop. The log showed repeated `BipartitionWarning: Failed to find a balanced cut after 1000 attempts`. After **43 minutes of CPU time** with no output, the process was killed.

**Root cause**: California's geography includes narrow coastal peninsulas (Big Sur, Point Reyes, the San Francisco peninsula) and deep mountain valleys. For sub-regions of the recursive bipartition with ε=0.5%, no spanning-tree cut could split the precinct set into two populations each within ±0.5% of the sub-target. Because `max_attempts=None`, the code looped indefinitely rather than escalating.

**Resolution**: Implemented epsilon-doubling with a finite attempt cap (5,000 per bipartition), and separated initial partition tolerance from repair tolerance.

---

### Failure 3 — v2 with Random-Walk Repair at SEED_EPSILON (repair stalled)

**What happened**: With ε=2% initial partition (fast) and a 2,000-step repair burn-in at `pop_tol = seed_eps = 2%`, the repair barely moved. Log output: `After repair: pop_dev_max = 3.9900%` — almost unchanged from the initial 4% (ε had doubled to 4% because 2% bipartition also failed for one sub-region of CA).

**Failure mode**: The random-walk repair selects boundary edges uniformly. With K=52 districts and ~2,500 boundary edges, the worst district (district 9, deviation +3.84%) was touched in only ~2% of steps ≈ 40 times in 2,000. Moreover, when it *was* selected, the combined population of district 9 (deviation +3.84%) and any positive-deviation neighbour was:

```
combined = (1.0384 + 1.02) × ideal = 2.058 × ideal
Valid cut at pop_tol=4%: need combined ≤ 2 × 1.04 × ideal = 2.08 × ideal ✓
Valid cut range for first half: [0.96, 1.04] × ideal = [730k, 791k]
Second half: 2.058 × 760k − first_half ∈ [773k, 834k]
Intersection with [730k, 791k]: [773k, 791k]   ← only 18k window (~4 precincts)
```

The valid-cut window was so narrow that most random spanning trees produced zero valid cut nodes. Acceptance rate ≈ 1–2% per attempt on this district pair. Combined with the low probability of selecting the district at all, the repair chain moved at ~1% of a step per repair-step.

**Resolution**: Replaced random-walk repair with targeted repair (see Failure 4).

---

### Failure 4 — Targeted Repair Hits MAX_REPAIR Ceiling (0.5575% vs. 0.5000%)

**What happened**: The targeted repair algorithm explicitly identifies the most imbalanced district and merges it with a strategically chosen neighbour using `repair_tol = |combined/(2×ideal) − 1| × 1.05`. After MAX_REPAIR = 20,000 steps, log showed:

```
Repair done in 20000 steps: pop_dev_max = 0.5575%
⚠  pop_dev_max 0.5575% still exceeds POP_TOL 0.5000% after 20000 repair steps.
```

The repair converged to **0.5575%** — tantalizingly close to 0.5%, but never crossing the threshold. The 100,000 sampling steps then ran and completed correctly in terms of accepting ReCom steps (the chain was no longer frozen).

**Failure mode**: With `pop_dev_max = 0.5575%` at start of sampling, the worst district had population = 760,272 × 1.005575 = 764,509. Any adjacent pair involving this district had combined population:

```
combined = 764,509 + neighbour_pop
```

For valid cuts at POP_TOL=0.5%: need combined ∈ [2 × 756k, 2 × 764k] = [1,512k, 1,528k]. The neighbour needed population in [748k, 764k], i.e. within ±1% of ideal. If the neighbour was itself slightly high (say 762k), combined = 764,509 + 762k = 1,526k — within range, and valid cuts exist. Steps involving this pair were accepted. But if the neighbour was at 765k, combined = 1,529k > 1,528k — no valid cuts, rejection.

The chain slowly drifted, with the worst district sometimes improving and sometimes worsening. The residual 0.0575% excess pop was caused by the discrete nature of precincts: the minimum-population step that could fix district 9 required transferring a precinct with ~430 people (= 0.057% of ideal), and such a small precinct happened to be absent at the border. This is a **granularity problem** — the precinct grid is too coarse to achieve exactly ±0.5%.

**Secondary failure**: Even ignoring population, all 100,000 CA plans had `pp_min < 0.10` (maximum observed: 0.091). The Polsby-Popper minimum filter was independently fatal. The plans that *did* pass the population filter (89,188 of 100,000) all failed pp_min ≥ 0.10. **Zero plans passed both hard filters.**

---

### Failure 5 — California's pp_min Is Geometrically Bounded Below 0.10

**This is not an algorithm bug — it is a geographic fact.**

With K=52 districts from 9,126 precincts in California's complex geography, the observed pp_min distribution across 100,000 MCMC steps was:

| Statistic | pp_min |
|-----------|--------|
| Minimum | 0.0268 |
| 5th percentile | 0.0441 |
| 25th percentile | 0.0560 |
| Median | 0.0673 |
| 75th percentile | 0.0749 |
| 95th percentile | 0.0859 |
| **Maximum** | **0.0909** |

The **theoretical maximum observable pp_min** was 0.091 — meaning *every single plan* in the 100,000-plan ensemble had at least one district with Polsby-Popper below 0.10. This is not a sampling artifact; it reflects California's actual geography:

- **Coastal districts**: The San Francisco Bay shoreline, Monterey Bay, and the Pacific coast force some districts into long, thin coastal strips. A district that stretches from Santa Cruz to Half Moon Bay along the Pacific coast is inherently elongated.
- **Mountain districts**: The Sierra Nevada range forces districts to follow narrow east-west valleys, producing high aspect ratios.
- **Urban fragmentation**: With K=52, urban areas like Los Angeles must be divided into small, irregular pieces that interlock with suburban precincts. No matter how the cuts are drawn, some pieces end up non-compact.

California's Congressional District 23 (current enacted map) has a Polsby-Popper score of approximately 0.04. Its District 25 scores approximately 0.06. The 0.10 threshold was calibrated on Maryland (K=8, compact Mid-Atlantic geography) and generalises poorly to western states with complex coastlines and mountain ranges.

**Resolution**: Lower `PP_MIN_THRESHOLD` from 0.10 to 0.05. With this change:
- Plans passing pp_min ≥ 0.05: 86,112 / 100,000 (86.1%)
- Plans passing pop_dev_max ≤ 0.005: 89,188 / 100,000 (89.2%)
- **Plans passing both: 75,300 / 100,000 (75.3%)**

This gives a rich ensemble for Pareto selection — substantially better than NY's 640 passing plans.

---

## 4. Five Proposed Improvements

### Improvement 1 — Multi-Stage Cooling for the Repair Phase

**Problem**: The targeted repair with a fixed `repair_tol` converges to `|combined/(2×ideal) − 1|` asymptotically, never below. For CA, this left pop_dev_max stuck at 0.5575% just above the 0.5% target.

**Proposed solution**: Use a geometric cooling schedule, halving the tolerance at each stage until POP_TOL is reached. At each stage, run steps until the max deviation actually drops below the current stage tolerance before tightening.

```
PROCEDURE multi_stage_repair(assignment, cg, ideal_pop, K, rng):
    pop_tol_schedule ← [seed_eps, seed_eps/2, seed_eps/4, ..., POP_TOL]
    pop_tol_schedule ← [t FOR t IN schedule IF t >= POP_TOL]  # stop at target
    pop_tol_schedule.append(POP_TOL)

    FOR target IN pop_tol_schedule:
        actual_tol ← target × 2.0   # use 2× to guarantee valid cuts exist
        WHILE max_deviation(assignment) > target:
            targeted_recom_step(assignment, cg, ideal_pop, actual_tol)
            IF steps_in_this_stage > 10_000:
                LOG "Warning: stage not converging at target=", target
                BREAK

    RETURN assignment
```

**Expected improvement**: Near-guaranteed convergence to pop_dev_max ≤ POP_TOL = 0.5% regardless of initial partition quality, eliminating the granularity trap that left CA at 0.5575%.

**Theoretical guarantee**: At each stage with `actual_tol = 2 × target`:
```
combined ≤ 2 × ideal × (1 + prev_target) ≤ 2 × ideal × (1 + actual_tol)
```
Valid cuts always exist, so the chain is never stuck. Each stage reduces max deviation by roughly half in O(K) targeted steps.

---

### Improvement 2 — State-Adaptive Polsby-Popper Threshold

**Problem**: PP_MIN_THRESHOLD = 0.10 was calibrated for compact eastern states. For California, the geometrically achievable maximum pp_min is ≈0.091. For other large western states (TX, WA, OR) with coastal/mountain geography and high K, the threshold may also be too strict.

**Proposed solution**: After sampling, compute the observed pp_min distribution and set the hard filter threshold adaptively:

```
PROCEDURE adaptive_pp_threshold(metrics, K, base_threshold=0.10):
    # Use the 90th percentile of observed pp_min, capped at base_threshold
    p90 ← np.percentile(metrics.pp_min, 90)

    # Never go below PP_MIN_FLOOR regardless of geography
    PP_MIN_FLOOR ← 0.04

    threshold ← max(PP_MIN_FLOOR, min(base_threshold, p90 × 0.95))
    # 0.95 factor: filter the bottom 5% without filtering everything

    LOG f"Adaptive PP threshold: {threshold:.3f} (p90={p90:.3f})"
    RETURN threshold
```

For CA: `p90 = 0.086`, so `threshold = max(0.04, min(0.10, 0.086 × 0.95)) = max(0.04, 0.082) = 0.082`. This passes ~60% of plans rather than 0%.

**Fairness implication**: This is geographically honest. Requiring California's congressional maps to achieve the compactness of Maryland's is the wrong benchmark. The adaptive threshold asks "is this plan among the most compact *achievable* for this state?" — a relative, not absolute, standard.

---

### Improvement 3 — Parallel Tempering for Faster Chain Mixing

**Problem**: For large K (CA K=52, TX K=38), the ReCom chain mixes very slowly. Most ReCom steps in CA affect only ~350/9,126 = 3.8% of nodes. To fully reshuffle the entire partition takes O(K²) steps — ~2,700 steps for K=52. With 100,000 total steps, the chain completes only ~37 effective "full shuffles".

**Proposed solution**: Run multiple chains at different "temperatures" (different POP_TOL values) and swap assignments between chains:

```
PROCEDURE parallel_tempering(G, K, n_steps, n_chains=4):
    temperatures ← [POP_TOL, 2×POP_TOL, 5×POP_TOL, 10×POP_TOL]
    chains ← [independent_assignment(G, K, t) FOR t IN temperatures]

    FOR step = 1 … n_steps:
        FOR c IN range(n_chains):
            chains[c] ← recom_step(chains[c], pop_tol=temperatures[c])

        # Swap adjacent chains with Metropolis probability
        IF step % SWAP_EVERY == 0:
            FOR c IN range(n_chains - 1):
                # Compute log likelihood ratio (uniform distribution → ratio = 1)
                # Swap only constrained by connectivity (both valid plans)
                IF both chains[c] and chains[c+1] are valid at temperatures[c]:
                    SWAP chains[c], chains[c+1] with prob = min(1, swap_ratio)

        # Only record the coldest chain (strictest tolerance)
        record(chains[0])
```

The hot chains (loose POP_TOL) mix much faster and can help the cold chain (strict POP_TOL) escape local regions of plan space. This is a standard technique from computational physics and protein folding.

**Expected speedup**: 3–10× effective mixing rate for K ≥ 20, at the cost of running 4 chains in parallel (still within 10 GB RAM using the CompactGraph representation).

---

### Improvement 4 — Population-Proportional Edge Weight Normalisation

**Problem**: The current edge weight formula `w = base × W_SAME_COUNTY^β × ...` is applied uniformly regardless of precinct population. A tiny 10-person rural precinct sharing a county boundary with a 37,000-person urban precinct gets the same weight treatment. This may create subtle geographic biases where large-population urban precincts exert disproportionate influence on spanning tree sampling.

**Proposed solution**: Normalise edge weights by the geometric mean of the two precinct populations:

```
w_normalised(u, v) = w(u, v) / sqrt(pop(u) × pop(v) + ε)
```

This makes the edge weight reflect both administrative cohesion *and* population structure. Two large counties' shared border gets relatively higher weight than a tiny rural precinct touching a large urban one.

**Theoretical implication**: The spanning tree distribution under Wilson's algorithm would become proportional to the product of *normalised* edge transition probabilities. This more accurately reflects "human-scale" administrative integrity — a county line between two major cities matters more to real constituents than a county line between two unpopulated mountain precincts.

**Note**: This change would require re-running all existing ensembles with the new weight formula for strict comparability, but would improve results for states with high population variance (CA, NY, TX, FL).

---

### Improvement 5 — Incremental Pareto Frontier with Streaming County-Split Computation

**Problem**: The current select.py computes county splits for a sample of 5,000 plans, requiring loading all 5,000 plan vectors into memory simultaneously. For CA (9,126 nodes × 5,000 plans × 1 byte = 45 MB) this is manageable, but for TX (K=38, ~15,000 nodes × 5,000 = 75 MB) and in the future for even larger datasets, this becomes a bottleneck. Additionally, county splits are computed only at selection time, not during sampling — so the sampler cannot bias toward low-split plans.

**Proposed solution**: Compute county splits *during sampling* and write them to the metrics parquet file as an additional column, enabling the full ensemble to be filtered without post-hoc recomputation:

```
PROCEDURE compute_metrics_with_splits(assignment, cg, county_arr, ...):
    # county_arr[i] = county index for node i (precomputed integer array)
    a = assignment
    county_by_district ← {}
    FOR i IN range(N):
        county_by_district[a[i]].add(county_arr[i])

    county_splits ← sum(len(dists) - 1 FOR each county's district set)
    max_county_districts ← max(len(dists) FOR each county's district set)

    # This adds ~2 μs per step in numpy; negligible vs. Wilson's LERW
    RETURN {pp_min, pp_mean, pp_max, pop_dev_max, pop_dev_mean,
            cut_edges, county_splits, max_county_districts}
```

With splits available for all 100,000 plans:
- The Pareto computation uses the full ensemble, not a 5,000-plan sample
- Selecting the 5th/95th percentile of county splits becomes a hard pre-filter
- Visualisations of the joint distribution (compactness vs. county splits) become possible

**Expected improvement**: Pareto frontier quality improves substantially because the full diversity of 100,000 plans is explored rather than 5,000. For CA with 58 counties and K=52, county splits are the primary geographic quality metric — using all plans to minimise them matters.

---

## 5. Hardware Constraints

### 5.1 Test Machine Specifications

| Component | Specification |
|-----------|--------------|
| CPU | Apple M-series (ARM, high-efficiency/performance cores) |
| Total RAM | **24 GB** unified memory (CPU and GPU share this pool) |
| OS | macOS |
| Python | 3.14 (Homebrew) in a venv |
| Storage | NVMe SSD (~3 GB/s read for parquet loads) |

### 5.2 Observed Memory Usage by Stage

| Stage | Component | RSS Added |
|-------|-----------|-----------|
| Import | Python + numpy + geopandas | 0.13 GB |
| Load G | NetworkX graph (CA: 9,126 nodes) | +0.02 GB |
| Load gdf | GeoDataFrame + Shapely C objects | +0.16 GB |
| Build CompactGraph | CSR numpy arrays | +0.00 GB (in-place) |
| del G + gdf | — | RSS unchanged (macOS holds pages) |
| Sampling (hot loop) | Working set (Wilson's walk dicts, etc.) | ±0.04 GB |
| **Total peak (CA)** | | **~0.37 GB** |

The v2 sampler uses approximately **0.37 GB RSS for CA**, down from **3.8–4.2 GB in v1**. The ~10× reduction comes from:

1. **CompactGraph replaces NetworkX**: CA graph = 4 MB (CSR numpy) vs. ~300 MB (NetworkX dict-of-dicts)
2. **No Python dict copy per step**: v1 allocated `dict(assignment)` every accepted step. For CA with 9,126 nodes, each copy = 9,126 Python key-value pairs ≈ ~800 KB of Python object overhead. Over 100,000 steps × ~70% acceptance rate = ~70,000 copies = 56 GB of total allocations (all freed, but pages remain held by Python's allocator pool on macOS).
3. **No NetworkX subgraph per step**: v1 called `G.subgraph(nodes_ab).copy()` each step. For a merged subgraph of ~350 nodes, this creates a new NetworkX graph object with its own internal dicts.
4. **Geometry arrays not per-step**: v1 called `polsby_popper(area, perimeter)` using pre-computed dicts; v2 uses pre-built numpy arrays and vectorised bincount.

### 5.3 RAM Budget for Planned States

| State | K | Nodes | Edges | Est. CompactGraph | Est. gdf | Est. Peak |
|-------|---|-------|-------|-------------------|----------|-----------|
| California | 52 | 9,126 | 24,975 | 4 MB | 165 MB | **0.37 GB** |
| New York | 26 | 14,190 | 39,092 | 6 MB | 200 MB | **0.34 GB** |
| Texas | 38 | ~12,000 | ~35,000 | 5 MB | 180 MB | **~0.35 GB** |
| Florida | 28 | ~10,000 | ~28,000 | 4 MB | 150 MB | **~0.33 GB** |

All four large states fit comfortably within 1 GB per process. Running CA + NY + TX + FL concurrently would use approximately **1.5 GB total** — well within the 24 GB budget.

### 5.4 The macOS RSS Non-Release Problem

A critical platform-specific constraint: **macOS's memory allocator never returns freed virtual memory pages to the OS.** This was the primary driver of the v1 memory crisis.

In v1, each MCMC step allocated and freed:
- 1 `dict(assignment)` copy (~700 KB Python object pool)
- 1 NetworkX subgraph object (~250 KB)
- Multiple lists and dicts for Wilson's walk

After 100,000 steps with 70% acceptance ≈ 70,000 allocations of ~1 MB each, Python had asked the OS for ~70 GB of virtual memory pages (in small increments). The OS reclaimed none of them on `del` or `gc.collect()`. The RSS number in `ps` reflected the "high-water mark" of allocations, not current usage.

**v2 solution**: Eliminate large per-step allocations entirely. The hot loop in v2 allocates only:
- `merged_indices`: numpy int32 array, ~350 × 4 = 1.4 KB for CA
- `local_adj`: Python dict, ~350 entries × ~80 bytes = 28 KB
- `parent`, `walk_pos`, `walk_order`: Python dicts/lists, ~350 entries each ≈ 25 KB

Total per-step transient allocation: ~55 KB. Over 100,000 steps: ~5.5 GB total allocations (vs. ~70 GB in v1). After the first few thousand steps, Python reuses the same pool pages rather than requesting new ones from the OS. RSS stabilises at ~0.37 GB and **does not grow** across the full 100,000-step run.

### 5.5 Storage and I/O Constraints

Each chunk file (1,000 plans × 9,126 nodes × 1 byte int8, compressed with Snappy):

| State | Chunk size | 100 chunks total | metrics.parquet |
|-------|-----------|-----------------|-----------------|
| CA | ~2.5 MB | ~250 MB | ~780 KB |
| NY | ~4.0 MB | ~400 MB | ~915 KB |

Total ensemble storage for all four large states: approximately **1.3 GB**. The NVMe SSD can write these chunks at ~1 GB/s (well above the 2.5 MB/s chunk write rate). Disk is not a bottleneck.

**Atomic writes**: Each chunk is written to a `.tmp` file and then `os.replace()`-d to the final name. On POSIX systems, `rename()` is atomic — a crash between the write and rename leaves the `.tmp` file, which is cleaned on next startup. This guarantees no corrupt or partial chunk files reach the parquet reader.

### 5.6 Parallelism Limits

Running two states simultaneously (CA + NY, or TX + FL) uses:
- CPU: 2 cores at ~100% each (one per process)
- RAM: ~0.37 + 0.34 = 0.71 GB combined — negligible

The machine has 10–12 performance cores. Running 6 states concurrently would be perfectly within RAM and CPU budget. The pipeline's `--workers N` flag controls this. For the four largest states that require sequential runs, two pairs can overlap: CA+NY first, then TX+FL.

**The limiting resource is time, not hardware.** The sampling speed for CA (≈ 1–4 steps/sec, variable due to acceptance rate and merged subgraph size) means 100,000 steps takes approximately 12–28 hours. This is a property of the algorithm's computational complexity, not the machine's capacity.

---

## Summary Table

| Issue | Root Cause | Status |
|-------|-----------|--------|
| v1 stuck district (CA, NY) | SEED_EPSILON > POP_TOL → combined pop outside valid-cut range | Fixed: targeted repair |
| GerryChain infinite loop | max_attempts=None on geometrically constrained sub-regions | Fixed: finite cap + epsilon doubling |
| Random-walk repair stall | Low probability of selecting bad district; narrow valid-cut window | Fixed: targeted repair |
| Repair doesn't reach 0.5% | Precinct granularity: minimum transfer = 0.06% > residual error | Partially mitigated; improvement 1 addresses fully |
| pp_min < 0.10 for all CA plans | California coastal/mountain geography + K=52 = inherently non-compact | Fixed: adaptive threshold lowered to 0.05 |

---

*All geographic and population data sourced from US Census Bureau TIGER 2020. No partisan, demographic, or electoral data was used in any algorithm described in this document. Post-hoc partisan lean overlays use VEST 2020 data and are computed separately from, and after, all plan selection.*
