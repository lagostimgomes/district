# Blind Redistricting

**Geography-only congressional redistricting for all 50 US states using weighted ReCom MCMC.**

This project draws congressional district maps using *only* geographic data — no partisan affiliation, no demographic data, no electoral history. The algorithm optimises for compactness, population equality, and county integrity. Partisan lean is computed *after the fact* as a post-hoc audit, never as an input.

---

## What it does

For each state the pipeline:

1. **Downloads** TIGER 2020 census geography (VTD precincts, census blocks, counties, roads)
2. **Builds** a weighted precinct dual graph, where edge weights encode administrative-unit proximity (same county → 10×, same city → 5×, same township → 3×; major roads divide weights by 2); only shared borders ≥ 50 m are included to suppress spurious point-touch adjacencies
3. **Samples** an ensemble of valid district plans with Weighted ReCom MCMC (DeFord, Duchin & Solomon 2021)
4. **Selects** two Pareto-optimal maps from the ensemble
5. **Renders** choropleth maps with optional post-hoc 2020 presidential lean overlay (VEST data)

The pipeline covers all 44 multi-district states (435 seats; the 6 at-large states with K=1 are skipped).

---

## Algorithm

### Weighted ReCom MCMC

Each MCMC step:
1. Pick two adjacent districts at random
2. Merge their precincts into one region and build a random spanning tree, biased by edge weights
3. Cut the spanning tree to split the region into two near-equal-population halves
4. Accept the new partition (unconditionally — this is a uniform sampler over valid plans)

**Edge weight formula:**

```
w(e) = base × W_SAME_COUNTY^β × W_SAME_PLACE^β × W_SAME_COUSUB^β / W_ROAD^β
```

where `β` amplifies the contrast, making intra-county cuts far more likely than cross-county cuts. **β is adaptive**: β=0.5 for K=2 states (WV, ME, NH, ID, MT, RI, HI) and β=2.0 for all others — low-K states have sparse county structure and over-penalising county crossings produces degenerate maps.

### Selection

From the ensemble, two maps are chosen via Pareto optimality across five objectives:

| Objective | Direction |
|-----------|-----------|
| Polsby–Popper mean (compactness) | Maximise |
| County splits | Minimise |
| Cut edges | Minimise |
| Max districts per county (worst-case fragmentation) | Minimise |
| `cut_border_m` (total cross-county boundary length) | Minimise |

A **peninsula filter** is applied before Pareto selection: any plan where the min or mean cut-border ratio is below 5% (indicating point-touch or thin-peninsula artefacts) is rejected.

Two representative plans are extracted:
- **Best compact** — highest Polsby–Popper mean within the Pareto frontier
- **Fewest splits** — minimum county fragmentation within the Pareto frontier

### Hard filters (applied before Pareto)

| Filter | Threshold |
|--------|-----------|
| `pp_min` (worst district compactness) | ≥ 0.10 |
| `pop_dev_max` (worst district population deviation) | ≤ 0.5% |

---

## Completed states

| State | K | Ensemble size | State | K | Ensemble size |
|-------|---|--------------|-------|---|--------------|
| Alabama | 7 | 2,000 | Missouri | 8 | 2,000 |
| Arizona | 9 | 2,000 | Montana | 2 | 2,000 |
| Arkansas | 4 | 2,000 | Nebraska | 3 | 2,000 |
| Colorado | 8 | 2,000 | Nevada | 4 | 2,000 |
| Connecticut | 5 | 2,000 | New Hampshire | 2 | 2,000 |
| Georgia | 14 | 2,000 | New Jersey | 12 | 2,000 |
| Hawaii | 2 | 2,000 | New Mexico | 3 | 2,000 |
| Idaho | 2 | 2,000 | North Carolina | 14 | 2,000 |
| Illinois | 17 | 2,000 | Ohio | 15 | 3,750 |
| Indiana | 9 | 2,000 | Oklahoma | 5 | 2,000 |
| Iowa | 4 | 2,000 | Oregon | 6 | 2,000 |
| Kansas | 4 | 2,000 | Pennsylvania | 17 | 2,000 |
| Kentucky | 6 | 2,000 | Rhode Island | 2 | 2,000 |
| Louisiana | 6 | 2,000 | South Carolina | 7 | 2,000 |
| Maine | 2 | 2,000 | Tennessee | 9 | 2,000 |
| Maryland | 8 | 2,000 | Utah | 4 | 2,000 |
| Massachusetts | 9 | 2,000 | Virginia | 11 | 2,750 |
| Michigan | 13 | 2,000 | Washington | 10 | 2,500 |
| Minnesota | 8 | 2,000 | West Virginia | 2 | 2,000 |
| Mississippi | 4 | 2,000 | Wisconsin | 8 | 2,000 |

**In progress (100k-step runs):** California (K=52), New York (K=26) — currently sampling.  
**Queued:** Texas (K=38), Florida (K=28).

---

## Example maps

Maps are generated locally by running `render_state.py` after the pipeline
completes for a state. Outputs land in `data/{abbr}/final/`:

- `map_compact.png` — best Polsby–Popper mean
- `map_fewest_splits.png` — fewest county splits
- `map_comparison.png` — side-by-side with optional enacted-map column
- `map_comparison_lean.png` — same, with VEST 2020 presidential lean badges

---

## Repository layout

```
.
├── pipeline/
│   ├── download.py       # Download TIGER 2020 geography files
│   ├── build_graph.py    # Build weighted precinct dual graph
│   ├── sample.py         # Weighted ReCom MCMC sampler (checkpoint/resume)
│   ├── select.py         # Pareto-optimal map selection
│   └── lean.py           # Post-hoc partisan lean (VEST 2020)
│
├── run_all_states.py     # Parallel driver for all 50 states
├── render_state.py       # Render maps for any completed state
├── render_us_map.py      # Render national overview map
├── state_configs.py      # StateConfig dataclass + registry (all 50 states)
│
└── data/
    └── {abbr}/
        ├── graph/        # Precinct dual graph (.gpickle) + GeoPackage
        ├── ensemble/     # plans.parquet, metrics.parquet (chunked writes)
        └── final/        # Selected maps (.gpkg), PNGs, stats JSONs, report
```

---

## Running the pipeline

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `geopandas`, `networkx`, `gerrychain >= 0.3.1`, `pyarrow`, `shapely`, `tqdm`.

> **GerryChain note:** PyPI may lag behind. If weighted spanning tree support is missing, install from source:
> ```bash
> pip install git+https://github.com/mggg/GerryChain.git
> ```

### Run a single state

```bash
# Full pipeline: download → graph → sample → select
python run_all_states.py --state MD

# Override MCMC step count (default scales with K)
python run_all_states.py --state TN --base-steps 5000
```

### Run all states in parallel

```bash
# Uses all CPU cores by default
python run_all_states.py --workers 4
```

MCMC steps scale proportionally with the number of districts:

```
steps(state) = round(BASE_STEPS × K / 8)
```

where `K` is the number of congressional seats and `8` is Maryland's baseline. A state with 16 districts gets twice as many steps.

### Render maps

```bash
# Render maps for one or more completed states
python render_state.py MD
python render_state.py MD TN GA

# Render the national overview
python render_us_map.py
```

### Checkpoint and resume

The sampler writes a chunk every 1,000 steps and saves a checkpoint atomically. If a run is interrupted, simply re-run the same command — it resumes from the last checkpoint automatically:

```bash
python run_all_states.py --state CA --steps 100000
# Interrupted? Just re-run the same command.
```

---

## Data sources

| Data | Source | License |
|------|--------|---------|
| VTD precincts | US Census TIGER 2020 | Public domain |
| Census blocks (population) | US Census TIGER 2020 (POP20) | Public domain |
| Counties, roads, places | US Census TIGER 2020 | Public domain |
| Current congressional districts | US Census TIGER 2022 (CD118) | Public domain |
| Partisan lean (post-hoc only) | [VEST 2020](https://dataverse.harvard.edu/dataverse/electionscience) | CC BY 4.0 |

---

## What this is not

- **Not a partisan gerrymander detector** — the algorithm has no access to partisan data.
- **Not legally certified** — maps are illustrative; real redistricting requires VRA compliance review, public comment, and legislative approval.
- **Not a claim about optimal maps** — the Pareto frontier contains many equally valid maps; these two are representative, not unique.

The post-hoc lean overlay is provided for transparency and curiosity. It shows what the *geography* happened to produce, not what was aimed for.

---

## References

- DeFord, D., Duchin, M., & Solomon, J. (2021). [Recombination: A family of Markov chains for redistricting](https://hdsr.mitpress.mit.edu/pub/fi441uwf). *Harvard Data Science Review*, 3(1).
- Polsby, D. D., & Popper, R. D. (1991). The third criterion: Compactness as a procedural safeguard against partisan gerrymandering. *Yale Law & Policy Review*, 9(2), 301–353.
- Metric Geometry and Gerrymandering Group. [GerryChain](https://github.com/mggg/GerryChain). Python library for ReCom MCMC sampling.

---

## License

MIT — see [LICENSE](LICENSE).

Maps and statistical outputs are derived entirely from US Census public-domain data and may be freely used.
