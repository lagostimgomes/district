# State-Specific Notes and Edge Cases

This document records pipeline behaviour, algorithmic choices, and known limitations
for states that required special handling or produced noteworthy results.
For the general algorithm see [ALGORITHM.md](ALGORITHM.md).

---

## Table of Contents

1. [West Virginia — Appalachian terrain, K=2 peninsula problem, VTD coverage gap](#west-virginia)
2. [California — VTD fallback, K=52 scale, coastal geometry](#california)
3. [New York — Dense metro, K=26, Long Island](#new-york)
4. [Texas — K=38 scale, geographic diversity](#texas)
5. [Florida — Peninsula geography, coastal precincts](#florida)
6. [Maryland — Baltimore City as independent county](#maryland)
7. [Virginia — 38 independent cities](#virginia)
8. [Louisiana — Parishes as county-equivalents](#louisiana)
9. [Hawaii — Non-contiguous islands, VTD coverage gap](#hawaii)
10. [New England states — No county government, Maine VTD coverage gap](#new-england-states)
11. [Missouri and Nevada — Additional independent cities](#missouri-and-nevada)
12. [At-large states (K=1) — Skipped](#at-large-states)
13. [VTD coverage gap — states with incomplete precinct coverage](#vtd-coverage-gap)
14. [Compactness ceiling by state geography](#compactness-ceiling-by-state-geography)

---

## West Virginia

**K=2 | Ensemble: 20,000 steps | pp_mean=0.139 | county_splits=3**

### Geography

West Virginia has the most irregular shape of any contiguous US state. The Appalachian ridge-and-valley topography creates a jagged boundary with deep finger-like protrusions (the Eastern Panhandle, the Northern Panhandle). The state's own Polsby-Popper score is approximately 0.107 — meaning no internal 2-district split can produce a plan whose worst district substantially exceeds the compactness of the state outline itself. The practical PP ceiling for a 2-district WV plan is approximately 0.17–0.18.

### Peninsula connector problem

At β=2.0, the MCMC strongly prefers cross-county cuts. In a K=2 state with 55 counties, this causes the chain to thread the district boundary through a county-crossing precinct adjacency even when that adjacency represents a very short shared border (as little as 79 m). On a rendered map, such plans appear as isolated district "pockets" connected to the main district body by a narrow corridor — the "patchy" appearance.

Root causes identified by analysis of the precinct dual graph (895 nodes, 2,387 edges):

- 84% of each district's perimeter is the WV state boundary itself; only 16% is the inter-district cut line.
- Several precinct pairs share borders of < 100 m, which can become the sole connection between two precinct clusters.
- At β=2.0, the chain over-concentrates on plans that cross county lines via these short borders.

### Resolution: adaptive β

WV is run with **β=0.5** (see ALGORITHM.md §5.6). Results:

| Metric | β=2.0 run | β=0.5 run |
|--------|-----------|-----------|
| pp_mean | 0.151 | 0.139 |
| county_splits | 5 | **3** |
| min inter-district border | 79 m | **672 m** |
| Peninsula filter passes | No | **Yes** |
| Run time (20k steps) | 3.5 h | **15 min** |

The β=0.5 run is also dramatically faster: because the chain rejects fewer proposals (the β=2.0 chain frequently finds no valid cut through the constrained cross-county topology), effective samples per hour is much higher.

### Peninsula filter

Even with β=0.5, the selection step applies the peninsula filter (min cut-border / mean cut-border ≥ 5%). All 6 plans on the Pareto frontier for the β=0.5 ensemble pass this test; the final selected plan's shortest inter-district border is 672 m, well above the 445 m threshold (5% of mean edge length 8,895 m).

### MIN_BORDER_M

Raising `MIN_BORDER_M` from 1 m to 50 m eliminates 4 spurious precinct adjacencies in the WV graph that had shared borders of < 10 m. These were digitization artifacts near county-line corners. The 50 m threshold does not affect any legitimate precinct adjacency (minimum real shared border in the WV graph after filtering is ~79 m, which becomes the new floor after graph rebuild; that single remaining short edge is what the peninsula filter catches at selection time).

### VTD coverage gap

WV VTDs (Voting Tabulation Districts) cover only **45,364 km²** of the state's total **62,756 km²** — a coverage rate of **72.3%**. The remaining 17,391 km² consists of federal land (principally the Monongahela National Forest, Spruce Knob–Seneca Rocks National Recreation Area, and several other wilderness tracts) to which no precinct is assigned in the Census TIGER VTD files.

This is a second, independent cause of visual patchiness in WV's rendered maps, distinct from the peninsula connector problem:

- The peninsula connector problem produces isolated district *pockets* — real precinct polygons that appear disconnected from their district body.
- The coverage gap produces *holes* — interior areas of the state that are simply absent from the district GeoPackage entirely, rendering as the background colour.

**Fix:** `render_all_state_comparisons.py` draws a dissolved county outline (the full state footprint) as a neutral base layer before drawing the district polygons. Unpopulated gaps fill in as `#2a3140` (muted gray) rather than the dark `#161b22` background, eliminating the "patchwork" appearance.

WV has the worst land-coverage rate of any contiguous state. See the [VTD coverage gap](#vtd-coverage-gap) section for the full cross-state comparison.

---

## California

**K=52 | Ensemble: 100,000 steps | pp_mean=0.174 | county_splits=117**

### VTD fallback to census tracts

California does not publish Voting Tabulation Districts (VTDs). The pipeline falls back to **census tracts** as the atomic unit. California has 9,129 census tracts (vs ~1,000–3,000 VTDs for most comparably-sized states). This increases graph size but also provides finer-grained population control.

The census-tract fallback is applied automatically in `pipeline/build_graph.py` when no VTD shapefile is found under `data/ca/vtd_precincts/`.

### Graph scale

| Metric | Value |
|--------|-------|
| Nodes (census tracts) | 9,129 |
| Edges | ~25,000 |
| Total population | 39,538,223 |
| Ideal district population | ~760,351 |

### Seed partition tolerance

At K=52, GerryChain's `recursive_tree_part` requires up to 5,000 bipartition attempts per recursion level to find a population-balanced initial partition. Without the adaptive epsilon mechanism (Section 8.3 of ALGORITHM.md), the initializer times out. In practice, WV requires `SEED_EPSILON ≈ 2–4%` and the repair burn-in corrects any districts that exceed the strict 0.5% tolerance before recording begins.

### Coastal geometry

California's Pacific coastline introduces many coastal precincts with extremely irregular shapes (high-resolution shoreline). These precincts contribute high boundary length and reduce the state-wide pp_mean. The practical PP ceiling for any CA 52-district plan is approximately 0.18–0.20. The 100k-step ensemble (vs the default 13k) ensures adequate Pareto diversity despite these geometric constraints.

### County splits

California has 58 counties. With K=52, a minimum of 52 − (counties that fit entirely in one district) county splits is geometrically unavoidable. The blind plan achieves 117 splits, reflecting the need to subdivide Los Angeles County (K≈14 based on population alone) and other large counties.

---

## New York

**K=26 | Ensemble: 100,000 steps | pp_mean=0.188 | county_splits=70**

### Dense metro and Long Island

New York City's five boroughs (counties: New York, Kings, Queens, Bronx, Richmond) and Long Island (Nassau and Suffolk counties) present a specific challenge: very high population density means these counties must be split into multiple districts, yet their geographic compactness is limited by being surrounded by water on multiple sides.

The blind plan allocates approximately 14 districts to the NYC metro area (NYC boroughs + Long Island), matching the 14 seats that population proportionality requires. These districts tend to have lower PP scores because they are bounded by coastline on one or more sides.

### Graph size

New York has the largest precinct dual graph: approximately 14,191 nodes (VTDs). The 100k-step ensemble provides about 10,000 effective independent samples (estimated from integrated autocorrelation time ≈ 7–10 steps for NY).

---

## Texas

**K=38 | Ensemble: 100,000 steps | pp_mean=0.170 | county_splits=176**

### Geographic scale and diversity

Texas spans 268,596 square miles — the largest of the contiguous 48 states. It contains three geographically distinct regions that create different redistricting pressures:

- **Eastern Texas**: Piney Woods and Gulf Coast; county boundaries follow irregular waterways
- **Central Texas**: Hill Country and Balcones Escarpment; compact rectangular counties  
- **Western Texas**: Trans-Pecos desert; large, sparsely populated counties that often need to be grouped together to reach population targets

### County splits

With K=38 and 254 counties, Texas has more counties than any other state. The blind plan achieves 176 county splits. Harris County (Houston, population ~4.7M) alone requires approximately 7 districts purely on population grounds, contributing at least 6 splits.

### Large sparse counties in western Texas

Loving County (population 64) and several other Trans-Pecos counties have populations so small that they must be combined with many neighbors to reach the ~770k ideal district population. The pipeline handles this correctly via population-balanced merging in the MCMC, but it means the western TX districts are geographically very large.

---

## Florida

**K=28 | Ensemble: 100,000 steps | pp_mean=0.205 | county_splits=81**

### Peninsula geography

Florida's shape — a long peninsula — creates a one-dimensional redistricting problem for roughly half the state. Districts along the Gulf and Atlantic coasts are necessarily elongated (low PP), constrained between water and the spine of the state. The Miami-Fort Lauderdale-Palm Beach corridor (Miami-Dade, Broward, Palm Beach counties) is particularly constrained, requiring approximately 10 districts in a roughly linear arrangement.

### Coastal precincts

Florida has extensive coastal and island precincts with very high boundary-length-to-area ratios. These precincts lower district PP scores regardless of plan quality. The pp_mean=0.205 for Florida is higher than one might expect from its shape because the interior (from Orlando northward) has many compact rectangular districts.

---

## Maryland

**K=8 | Ensemble: 2,000 steps | pp_mean=0.195 | county_splits=22**

### Baltimore City as independent county

Maryland has 23 counties plus **Baltimore City** (FIPS code 510), which is legally independent of Baltimore County and is treated as a county-equivalent in all federal data. The pipeline correctly identifies it as a separate county from its TIGER GEOID, giving it the same `W_SAME_COUNTY = 10×` weight multiplier as any other county.

This means the pipeline treats Baltimore City and Baltimore County as two communities of equal administrative standing — a correct geographic encoding of Maryland law. The district boundary between them is weighted exactly as any other inter-county boundary.

### Historical note

Maryland was the first state developed in this pipeline (as a proof-of-concept), and its architecture (recursive bipartition initialization, ReCom MCMC, Pareto selection) was later generalised to all 50 states.

---

## Virginia

**K=11 | Ensemble: 2,750 steps | pp_mean=0.195 | county_splits=63**

### 38 independent cities

Virginia has 38 **independent cities** (FIPS codes 510–840) that are legally separate from any county. This is the most independent-city-heavy state in the US, and it creates a uniquely complex county-equivalent structure:

| Example | FIPS | Relationship to county |
|---------|------|----------------------|
| Richmond City | 760 | Independent of Henrico/Chesterfield |
| Alexandria | 510 | Independent of Arlington |
| Roanoke City | 770 | Independent of Roanoke County |
| Virginia Beach | 810 | Independent (no adjacent county) |

All 38 independent cities are treated as county-equivalents in the pipeline, receiving the same `W_SAME_COUNTY = 10×` weight multiplier. This means the MCMC prefers to keep each independent city whole within a district, which aligns with the Virginia redistricting tradition of respecting the city/county administrative distinction.

**Practical effect:** Virginia's 95 counties + 38 independent cities = 133 county-equivalent units for a K=11 state, giving a very high ratio of administrative units to districts. The blind plan achieves 63 splits across these 133 units — a split rate of 47%, significantly higher than the national average split rate, reflecting the dense patchwork of independent cities in the Richmond–Tidewater corridor.

---

## Louisiana

**K=6 | Ensemble: 2,000 steps | pp_mean=0.160 | county_splits=37**

### Parishes as county-equivalents

Louisiana uses **parishes** (not counties) as its primary administrative subdivision. TIGER/Line shapefiles classify Louisiana parishes as county-equivalents with standard GEOID format, so no special handling is required — the pipeline treats them identically to counties in other states.

Louisiana has 64 parishes. The low pp_mean=0.160 reflects the irregular geometry of parishes bordering the Mississippi River delta and Gulf Coast, where wetlands and bayous create extremely tortuous boundaries.

---

## Hawaii

**K=2 | Ensemble: 2,000 steps | pp_mean=0.252 | county_splits=1**

### Non-contiguous islands

Hawaii consists of multiple non-contiguous islands. The four counties are:

| County | Islands |
|--------|---------|
| Honolulu County | Oʻahu + Northwestern Hawaiian Islands |
| Maui County | Maui, Molokaʻi, Lānaʻi, Kahoʻolawe |
| Hawaiʻi County | Hawaiʻi (the Big Island) |
| Kauaʻi County | Kauaʻi, Niʻihau |

Non-contiguous island precincts are connected in the graph only if they touch (they don't — islands are by definition separated by ocean). The pipeline handles this correctly: each island is a disconnected component, and the largest component is retained. In practice, Oʻahu contains the majority of the population and remains the core of the graph.

The natural 2-district split for Hawaii is Oʻahu (District 1) and the outer islands (District 2). The blind algorithm consistently finds this division, resulting in just 1 county split (Honolulu County is split because its NW Hawaiian Islands portion is grouped with the outer-island district). pp_mean=0.252 reflects the compact, roughly elliptical shapes of the main islands.

### VTD coverage gap

Hawaii's TIGER county shapefile includes the state's full territorial footprint, which includes the surrounding Pacific Ocean out to the county boundaries (particularly for Honolulu County, whose jurisdiction extends to the Northwestern Hawaiian Islands). VTDs are assigned only to the inhabited land area of the main islands.

As a result, Hawaii has the lowest VTD coverage rate of any state: **10.4%** (2,951 km² of VTDs vs. 28,412 km² of county geometry). The gap is ocean, not land — so on a rendered map it is not visually problematic. The "holes" in the district polygons are correctly perceived as water. The state-outline base layer fix applied for WV is also applied to HI for consistency, but it has no visible effect because the county outline already traces the island coastlines.

### Projection

Hawaii uses `EPSG:26904` (UTM Zone 4N) rather than the continental US `EPSG:5070` (Albers Equal Area). This is specified in `StateConfig` for HI and applied at every stage of the pipeline.

---

## New England States

**Connecticut (K=5), Maine (K=2), Massachusetts (K=9), New Hampshire (K=2), Rhode Island (K=2), Vermont (K=1)**

### No county government

New England states have county boundaries drawn on maps, but counties carry little or no governmental function — towns and cities are the primary units of local administration. In particular:

- **Connecticut**: County government was abolished in 1960. The 8 counties exist only as judicial and planning districts.
- **New Hampshire**: County government is minimal; towns are the effective administrative units.
- **Massachusetts**: County government is largely defunct; several counties have been abolished.
- **Maine**: County government exists but is weaker than in most states.

**Effect on the pipeline:** The `W_SAME_COUNTY = 10×` multiplier still applies — county FIPS codes exist in TIGER data even where county government is defunct, and the weight applies based on geography, not on whether a government exercises that geography. However, in practice, the county-split objective carries less normative weight in these states: keeping a county whole in Connecticut is less meaningful than keeping one whole in Texas.

County-subdivision (`W_SAME_COUSUB = 3×`) is more meaningful in New England, since towns (called county subdivisions in TIGER) are the real administrative units. The pipeline's inclusion of cousub weights gives towns appropriate weight in the spanning tree bias.

**Maine VTD coverage note:** Maine's county geometry includes the Gulf of Maine out to territorial waters, giving it a measured VTD coverage rate of only **14.6%** (13,360 km² of VTDs vs. 91,633 km² of county geometry). Like Hawaii, the gap is ocean, not land, so no visual holes appear on rendered maps. The state-outline base layer fix is applied but has no visible effect.

**Rhode Island special note:** RI achieves the highest pp_mean of any state: 0.515. Rhode Island is a small, roughly rectangular state with few geographic constraints. Its two districts consistently come out as roughly equal halves of the state — one covering the Providence metro, one covering the southern coastline — both geometrically compact. The 2-district problem for a state this shape has a near-optimal solution.

---

## Missouri and Nevada

### Missouri — St. Louis City (FIPS 29510)

Missouri has one independent city: **St. Louis City**, which separated from St. Louis County in 1876. It is treated as a county-equivalent in TIGER and in the pipeline, receiving the same `W_SAME_COUNTY = 10×` weight as any other county. The practical effect is that the MCMC prefers to keep St. Louis City and St. Louis County in separate districts rather than combining them — which aligns with the legal and cultural reality of their distinct governance.

### Nevada — Carson City (FIPS 32510)

Nevada has one independent city: **Carson City**, the state capital. It is a county-equivalent in TIGER. With K=4 and a relatively sparse population outside Las Vegas, Carson City ends up grouped with adjacent counties in the western Nevada district rather than being split, which the county-weight correctly encourages.

---

## At-Large States

**Alaska (K=1), Delaware (K=1), North Dakota (K=1), South Dakota (K=1), Vermont (K=1), Wyoming (K=1)**

These six states each have a single congressional seat and are not redistricted. The pipeline skips them (`cfg.k == 1` check in `run_all_states.py`). Their partisan lean is added to `compute_lean_2024.py` as a fixed entry based on the state-level 2024 presidential result.

**Alaska note:** Alaska uses `EPSG:3338` (Alaska Albers Equal Area) as its projection. If Alaska is ever re-added as a multi-district state (which would require a census reapportionment giving it 2+ seats), the pipeline already has the correct projection in `StateConfig`.

---

## VTD Coverage Gap

Most states have VTD (Voting Tabulation District) shapefiles that cover 100% of the state's land area — even rural, sparsely populated counties are divided into precincts that tile the full geography. Three states are exceptions:

| State | VTD area (km²) | County area (km²) | Coverage | Gap source |
|-------|---------------|-------------------|----------|------------|
| Hawaii | 2,951 | 28,412 | **10.4%** | Pacific Ocean within county boundaries |
| Maine | 13,360 | 91,633 | **14.6%** | Gulf of Maine within county boundaries |
| West Virginia | 45,364 | 62,756 | **72.3%** | Federal land (national forests, wilderness) |

All other 41 multi-district states are at ≥92.9% coverage; 38 are at exactly 100%.

### Why Hawaii and Maine differ from West Virginia

For Hawaii and Maine the gap is **ocean**. TIGER county boundaries extend to territorial waters, so the computed county area is much larger than the land area. The VTDs, which only exist on inhabited land, naturally cover a small fraction of the total county geometry. On a rendered map the district polygons only need to fill the islands/land — the surrounding ocean is visually transparent. No "holes" appear.

For West Virginia the gap is **inland federal land** — chiefly the Monongahela National Forest (900,000+ acres) and adjacent wilderness areas. These tracts are interior to the state outline. A rendered map that draws only the precinct-derived district polygons will show visible dark holes in the middle of the state. This is visually incorrect and misleading.

### Rendering fix

`render_all_state_comparisons.py` addresses this by drawing a dissolved state outline (union of all county geometries) as a filled neutral-gray base layer (`#2a3140`) before drawing the district polygons. This ensures:

- WV's federal land gaps show as a muted gray rather than the dark background, eliminating the "patchwork" appearance.
- HI and ME are handled consistently, with no visible effect (the base layer traces the island/land coastlines that the district polygons already fill).

The national overview map (`render_us_map.py`) is less affected because its light-blue ocean background (`#C8DCF0`) causes any WV gaps to appear as a slightly wrong shade of blue — less visually jarring at national scale — but the comparison maps use a dark background where the gap is obvious.

---

## Compactness Ceiling by State Geography

The Polsby-Popper mean of the best blind plan is largely determined by the state's geography, not the algorithm's quality. States with simple, compact shapes yield naturally compact districts; states with complex coastlines, mountain terrain, or irregular borders cannot.

| State | pp_mean | Primary geometric constraint |
|-------|---------|------------------------------|
| Rhode Island | 0.515 | Near-rectangle; minimal coast complexity |
| Nebraska | 0.409 | Rectangular Midwest state |
| Nevada | 0.402 | Rectangular desert state |
| Kansas | 0.370 | Rectangular Great Plains |
| New Hampshire | 0.357 | Compact; minor coastal indentation |
| … | … | … |
| Ohio | 0.166 | Lake Erie shoreline, irregular NE border |
| Texas | 0.170 | Gulf coast estuaries; Rio Grande meanders |
| California | 0.174 | Pacific coastline; mountain terrain |
| Louisiana | 0.160 | Mississippi delta; bayou coastline |
| Tennessee | 0.164 | East–west elongation; Appalachian ridge |
| Georgia | 0.145 | Piedmont/coastal plain transition |
| West Virginia | 0.139 | Appalachian ridge-and-valley; panhandles |

**Interpretation:** A pp_mean of 0.14 for WV does not indicate a poor algorithm result — it reflects the near-impossibility of drawing two compact districts within a shape that is itself non-compact. The correct benchmark is the theoretical maximum for the state's geometry, not a universal target.

For states with pp_mean > 0.35, compactness is not a binding constraint; the Pareto frontier is dominated by county-split and cut-edges objectives. For states with pp_mean < 0.20, compactness is limited by geography and the algorithm is doing well to approach even 0.20.
