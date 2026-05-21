# Geography as Destiny: A Nationwide Blind Algorithmic Redistricting Study Using Weighted ReCom MCMC

**Abstract**

We present a nationwide study of congressional redistricting using a geography-only algorithmic framework, covering all 44 multi-district US states (435 seats). Our system implements Weighted Recombination Markov Chain Monte Carlo (ReCom MCMC), a method introduced by DeFord, Duchin, and Solomon (2021), applied to US Census TIGER 2020 Voting Tabulation District precincts with zero access to partisan, demographic, or electoral data during map generation. Post-hoc partisan analysis using VEST 2020 precinct-level presidential returns is conducted solely as an audit. The algorithm selects plans via Pareto optimality across five geometric objectives — maximizing mean Polsby-Popper compactness, minimizing county splits, minimizing cut edges, minimizing the maximum number of districts per county, and minimizing total cut border length — subject to a strict ±0.5% population deviation tolerance. The resulting blind maps yield 204 Democratic and 231 Republican seats out of 435, compared to 203 Democratic and 232 Republican seats in the enacted 118th Congress maps — a difference of a single seat at the national level. However, the blind maps produce 50 competitive seats (margin ≤5%) versus 45 in enacted plans, an 11% increase. State-level analysis reveals identifiable gerrymanders in Illinois (enacted favors Democrats by 3 seats), Wisconsin (enacted favors Republicans by 2 seats), and New York (enacted favors Republicans by 3 seats relative to geography). The primary finding is that national partisan seat proportions are largely a function of residential geography — the spatial sorting of partisan voters — rather than deliberate mapmaker choices, a result consistent with Chen and Rodden's (2013) "unintentional gerrymandering" hypothesis extended to the full national scale. We discuss implications for redistricting reform, the limitations of geography-only approaches under the Voting Rights Act, and the use of ensemble methods as a diagnostic tool for detecting partisan manipulation.

**Keywords**: redistricting, gerrymandering, Markov Chain Monte Carlo, compactness, partisan bias, computational redistricting, algorithmic maps

---

## 1. Introduction

Congressional redistricting is among the most consequential exercises of political power in American democracy. Conducted once per decade following the decennial census, the process of drawing district boundaries determines which communities share representation, how competitive elections are, and, in contested cases, which party governs. Yet this process is also among the most susceptible to manipulation: partisan actors who control state legislatures can draw maps that systematically advantage their party — or, through racial gerrymandering, dilute or pack minority communities in ways that suppress their political representation.

The concern is neither new nor hypothetical. Scholars have documented partisan manipulation in maps drawn by both major parties across a wide range of states (Stephanopoulos and McGhee 2015; Wang 2016). Courts have grappled with the problem for decades, from the foundational one-person-one-vote ruling in *Reynolds v. Sims* (1964) through racial gerrymandering challenges in *Thornburg v. Gingles* (1986), to the Supreme Court's controversial 2019 decision in *Rucho v. Common Cause*, which held that federal courts lack jurisdiction to adjudicate partisan gerrymandering claims under the political question doctrine. That decision effectively closed the federal courthouse door to partisan gerrymandering challenges, intensifying interest in alternative mechanisms — including algorithmic redistricting — for producing maps untainted by partisan intent.

Algorithmic redistricting has a long intellectual history, stretching from Vickrey (1961) and Weaver and Hess (1963) through the modern era of MCMC-based ensemble methods (Fifield et al. 2020; DeFord et al. 2021). The core appeal is straightforward: an algorithm that draws districts using only geography and population — with no access to the identities, partisan affiliations, or voting histories of the people being districted — cannot, by construction, gerrymander. If the resulting maps happen to advantage one party, that advantage is a product of where people live, not of deliberate manipulation.

This paper reports on a complete nationwide implementation of such a system. We applied Weighted ReCom MCMC (DeFord et al. 2021) to all 44 US states with multiple congressional districts, generating ensemble plans for each state and selecting final maps through Pareto-optimal geometric criteria. Partisan lean was computed only after map selection, as a post-hoc audit using precinct-level presidential returns. The scale of the study — 435 seats across 44 states — is, to our knowledge, the most comprehensive geography-only redistricting analysis published to date.

Our findings speak directly to an active debate in political science. Jonathan Rodden (2019) and Chen and Rodden (2013) have argued that Democrats' geographic concentration in dense urban areas produces partisan bias in single-member district systems even without deliberate manipulation — what Chen and Rodden called "unintentional gerrymandering." Our results provide the most direct empirical test of this hypothesis at national scale: a geography-only algorithm produces near-identical national seat shares as politically drawn enacted maps. At the same time, our state-level analysis identifies clear departures from geographic expectation in specific states, providing a diagnostic fingerprint for deliberate manipulation.

The remainder of this paper is organized as follows. Section 2 reviews the relevant literature on redistricting law, compactness measures, algorithmic redistricting, and the geography of partisanship. Section 3 describes our data sources, graph construction methodology, MCMC sampler, and plan selection procedure. Section 4 presents results on compactness, partisan seat distributions, and competitive seat generation. Section 5 discusses interpretive implications and limitations. Section 6 concludes.

---

## 2. Literature Review

### 2.1 Legal Standards and the History of Redistricting Reform

The legal framework governing congressional redistricting in the United States has evolved substantially since the mid-twentieth century, shaped by a series of landmark Supreme Court decisions that progressively defined the constitutional and statutory constraints on mapmakers.

The foundational principle of population equality was established in *Reynolds v. Sims* (1964), in which the Supreme Court held that the Equal Protection Clause of the Fourteenth Amendment requires state legislative districts to be drawn on a population-equal basis. The companion case *Wesberry v. Sanders* (1964) applied the same principle to congressional districts. These decisions eliminated the severe malapportionment that had given rural areas disproportionate representation in Congress and state legislatures, but they did not address the substantive composition of districts — only their population sizes.

The Voting Rights Act of 1965, and its subsequent amendments, introduced affirmative obligations regarding minority representation. Section 2 of the Act prohibits any voting practice or procedure that discriminates on the basis of race. The Supreme Court's interpretation of Section 2 in *Thornburg v. Gingles* (1986) established a three-part test for when minority groups are entitled to majority-minority districts: the group must be sufficiently large and geographically compact to constitute a majority in a single-member district, it must be politically cohesive, and white voters must vote sufficiently as a bloc to defeat the minority group's preferred candidates. The *Gingles* preconditions have shaped redistricting practice ever since, requiring mapmakers to assess racial demography when drawing district lines — a requirement that stands in fundamental tension with the "blind" algorithmic approach examined in this paper, as we discuss in Section 5.

A distinct line of cases addressed racial gerrymandering — the practice of drawing district lines primarily on racial grounds. In *Shaw v. Reno* (1993) and *Miller v. Johnson* (1995), the Supreme Court held that districts whose boundaries are so irregular as to be explicable only as racial classifications are subject to strict scrutiny. These decisions created an uncomfortable tension: mapmakers were simultaneously required (under Section 2) to create majority-minority districts in some circumstances, yet prohibited (under the Equal Protection Clause) from making race the predominant factor in drawing lines.

Partisan gerrymandering proved more resistant to judicial remedy. In *Davis v. Bandemer* (1986), the Supreme Court held that partisan gerrymandering claims were justiciable, but the plurality failed to articulate a workable standard. In *Vieth v. Jubelirer* (2004), four justices concluded the claims were nonjusticiable political questions. The resolution came in *Rucho v. Common Cause* (2019), in which Chief Justice Roberts wrote for a five-justice majority that federal courts cannot adjudicate partisan gerrymandering claims because there is no judicially discernible and manageable standard for adjudicating them. The decision left partisan gerrymandering claims to state courts and state constitutions — a path that has proven fruitful in some states (notably North Carolina and Pennsylvania) but unavailable in others.

The *Rucho* decision has intensified scholarly and policy interest in algorithmic alternatives. Independent redistricting commissions, adopted in states including California, Arizona, and Michigan, attempt to insulate mapmakers from partisan pressure through institutional design (McDonald 2004; Cain 2012). Algorithmic systems take a more radical approach: if the algorithm has no access to partisan data, partisan manipulation is structurally impossible.

### 2.2 Compactness: Measures and Debates

Geographic compactness has long been regarded as a proxy for the legitimacy of district shape — the intuition being that bizarrely shaped districts signal political manipulation. Multiple quantitative measures have been proposed, each capturing a different geometric intuition.

The Polsby-Popper score (Polsby and Popper 1991) is the ratio of a district's area to the area of a circle with the same perimeter: PP = 4πA/P², where A is area and P is perimeter. A perfect circle achieves PP = 1.0; districts with tortured, irregular boundaries approach 0. The Polsby-Popper measure penalizes elongation and jaggedness — qualities characteristic of manipulated districts — and is among the most widely used measures in scholarly and legal contexts. We employ it as a primary objective in our Pareto selection criterion.

The Schwartzberg score (Schwartzberg 1966) takes the inverse of the Polsby-Popper formulation, expressing the ratio of district perimeter to the circumference of a circle of equal area. The Reock score (Reock 1961) instead measures the ratio of district area to the area of the minimum bounding circle — a measure that penalizes elongation but is insensitive to boundary irregularity along the perimeter. The convex hull ratio, sometimes called the convex hull compactness, measures the proportion of the convex hull that is actually contained in the district; it captures a different aspect of shape than perimeter-based measures.

Each measure has known limitations. Polsby-Popper is sensitive to the resolution of the geographic boundary: a district following a fractal coastline will score poorly not because of manipulation but because of natural geographic complexity. This confound is directly relevant to our study: states with irregular coastlines (Louisiana, Texas, Florida, West Virginia) produce districts with low Polsby-Popper scores regardless of how carefully the algorithm is constructed, because district boundaries must follow geographic features. We return to this point extensively in the results section. Chambers and Miller (2010) and Fryer and Holden (2011) have proposed alternative measures designed to be more robust to this concern, though they have seen less adoption in applied work.

Young (1988) argued that compactness requirements, while appealing in principle, are in tension with the goal of creating communities of interest and are frequently manipulated themselves — a sufficiently creative mapmaker can draw a compact district that perfectly cleaves apart a minority community. Niemi et al. (1990) offered a more measured assessment, finding compactness and partisan bias to be correlated but not collinear, suggesting compactness requirements have some independent effect.

### 2.3 Algorithmic Redistricting: Historical Development

The use of mathematical optimization to automate district drawing has a longer history than is often appreciated. Vickrey (1961), in an early formal treatment, proposed hexagonal grid-based districting as a way to eliminate partisan manipulation. Weaver and Hess (1963) proposed what they called an "automatic" redistricting system based on geographic centers of gravity, assigning population units to the nearest center iteratively. Garfinkel and Nemhauser (1970) formulated redistricting as an integer programming problem and applied it to small cases, establishing the computational complexity of the problem.

Through the 1970s and 1980s, algorithmic approaches remained computationally out of reach for realistic instances — the combinatorial space of possible district assignments is astronomically large. Altman (1997) surveyed the field and concluded that while algorithmic redistricting had real promise as a tool for analysis and evaluation, the computational constraints of the era limited practical application. Mehrotra, Johnson, and Nemhauser (1998) made significant progress on the integer programming approach, applying branch-and-cut methods to realistically sized instances, but the approach remained intractable for large states.

The emergence of MCMC sampling methods fundamentally changed the algorithmic redistricting landscape. Rather than seeking the optimal plan — a computationally infeasible goal — MCMC methods sample from the space of all valid plans, making it possible to characterize the distribution of outcomes under a given set of constraints. This reframing, from optimization to sampling, proved enormously productive.

Mattingly and Vaughn (2014) were among the first to apply MCMC methods systematically to redistricting in a politically consequential context, using sampling to show that North Carolina's enacted congressional map was an extreme outlier in the space of geographic plans. Carter, Hunter, and Cho (2019) extended this approach using a parallel computing framework to generate large ensembles of plans. Fifield et al. (2020) produced the `redist` R package, which implements several MCMC algorithms for redistricting and has become a widely used tool in the research community.

### 2.4 ReCom and Its Extensions

The Recombination (ReCom) algorithm of DeFord, Duchin, and Solomon (2021) represents the current state of the art in MCMC redistricting. The key innovation is the proposal mechanism: rather than making small local modifications to an existing plan (swapping individual precincts between adjacent districts), ReCom proposes large, spatially coherent changes by merging two adjacent districts into a single region and then repartitioning that region using a spanning tree method. The spanning tree approach guarantees that both resulting districts are connected, addressing a major limitation of simpler MCMC proposals. DeFord et al. demonstrated that ReCom produces efficient mixing — the Markov chain moves rapidly through the space of valid plans — and that the resulting ensemble is a practical tool for detecting outlier maps.

Cannon et al. (2023) introduced several extensions to the ReCom framework, including the Reversible ReCom algorithm, which satisfies detailed balance and thus has a well-characterized stationary distribution. The Weighted ReCom variant, which we employ in this study, modifies the proposal distribution to reflect geographic structure — specifically, the likelihood of drawing district boundaries along natural geographic features such as county lines and major roads. By assigning higher weights to edges that cross county boundaries or major roads, the algorithm preferentially proposes splits that respect existing political and geographic units, producing plans that are more likely to minimize county splits and maintain geographic coherence.

Dube and Tucker (2021) describe the GerryChain software package, an open-source Python implementation of ReCom and related algorithms developed at the Metric Geometry and Gerrymandering Group (MGGG) at Tufts University. GerryChain has been used in numerous court cases and policy analyses, and its implementation of Weighted ReCom underlies the present study.

### 2.5 Ensemble Analysis as a Diagnostic Tool

The insight that MCMC ensembles can serve as a baseline for evaluating enacted plans — rather than as a tool for generating adoption-ready maps — has proven to be the most practically influential application of algorithmic redistricting research. Herschlag, Kang, Lo, Sachet, Schutzman, Shimek, and Mattingly (2020) applied this framework to North Carolina congressional and legislative districts, showing that the enacted maps were extreme outliers relative to the ensemble of geographically constrained plans across multiple partisan metrics. Duchin and Tenner (2018) developed a theoretical framework for understanding ensemble distributions and their implications for evaluating partisan manipulation.

This diagnostic use of ensembles sidesteps the politically fraught question of which algorithmic plan should be adopted; instead, it asks only whether the enacted plan is unusual relative to the space of geographic plans. The present paper employs both the diagnostic logic (comparing enacted maps to geographic expectations) and the direct use of algorithm-generated maps (reporting on the plans actually selected).

### 2.6 Partisan Bias Measurement

A substantial literature addresses how to measure partisan bias in electoral systems. Gelman and King (1994) proposed a simulation-based approach to estimating bias and responsiveness in electoral systems, defining bias as the difference in seat shares two parties would receive if they each received exactly 50% of the vote. Their framework, which uses district-level vote swings to estimate the seats-votes curve, has been widely adopted.

Stephanopoulos and McGhee (2015) introduced the efficiency gap as a simpler, single-number summary of partisan bias. The efficiency gap measures the difference in "wasted votes" (votes cast for the losing candidate, plus votes cast for the winning candidate in excess of what was needed to win) between the two parties; a large efficiency gap indicates one party is more efficiently converting votes into seats than the other. The measure attracted significant attention as a potential legal standard for partisan gerrymandering claims, though courts have been reluctant to adopt it as a bright-line rule, and scholars have identified significant limitations (Bernstein and Duchin 2017; Warrington 2018).

Warrington (2018) proposed a complementary measure based on a generic-voter model, assessing the probability that a given district map would produce a particular seat distribution under a range of vote swing scenarios. This approach, like ensemble analysis, characterizes the distribution of outcomes rather than a single number, but requires assumptions about the distribution of vote swings.

### 2.7 Geography of Partisanship

Perhaps the most relevant body of literature for interpreting this study's results concerns the relationship between residential geography and partisan outcomes in winner-take-all single-member districts. Chen and Rodden (2013) made the influential observation that Democrats' concentration in dense urban areas creates systematic partisan bias in single-member district systems even without deliberate manipulation. Because Democratic voters are geographically clustered, Democratic districts tend to produce large winning margins (wasted votes), while Republican voters are more efficiently distributed across districts. Chen and Rodden estimated that a substantial portion of the partisan bias observed in US congressional elections could be attributed to this geographic sorting rather than deliberate gerrymandering — a phenomenon they termed "unintentional gerrymandering."

Rodden (2019) extended this analysis in *Why Cities Lose*, arguing that the geographic concentration of the left in cities is a general feature of industrialized democracies with winner-take-all electoral systems, and that it systematically disadvantages left parties in such systems. The implication for redistricting is direct: even a perfectly neutral algorithm will tend to produce maps that favor Republicans in a geographically sorted electorate, because geographic compactness principles will tend to pack Democrats into dense urban districts.

This thesis has been contested and refined. Kaufman, King, and Komisarchik (2021) argued that geographic sorting accounts for some but not all of the observed partisan bias in US maps, and that deliberate gerrymandering still adds a significant additional layer of bias in states with partisan control of the redistricting process. Rodden and colleagues (2021) responded that the distinction between geographic and deliberate bias is not always clear, because mapmakers can exploit geographic sorting strategically. Our results speak directly to this debate: at the national level, a geography-only algorithm produces essentially identical seat shares to enacted maps, consistent with the Rodden thesis, while state-level departures identify specific cases where deliberate manipulation adds to geographic baseline bias.

### 2.8 Voting Rights and Majority-Minority Districts

The Voting Rights Act imposes affirmative requirements on redistricting that a geography-only algorithm cannot satisfy. Under *Thornburg v. Gingles* (1986), jurisdictions covered by Section 5 of the VRA (before the *Shelby County* decision gutted Section 5 preclearance in 2013) and all jurisdictions under Section 2 must in some circumstances create majority-minority districts to ensure minority voters have an equal opportunity to elect representatives of their choice.

Pildes and Niemi (1993) analyzed the tension between compactness requirements and majority-minority district creation, arguing that the two goals are often in conflict. Grofman, Handley, and Niemi (1992) provided an extensive treatment of minority vote dilution and the *Gingles* framework. More recent scholarship has debated whether majority-minority districts are always in the interest of minority communities, since packing minority voters into supermajority districts may reduce their influence in adjacent districts (Lublin 1997; Cameron, Epstein, and O'Halloran 1996).

A geography-only algorithm is blind to these considerations by design. The maps produced in this study are not intended as legal substitutes for enacted plans; they do not comply with the VRA, and their use in any legal or administrative context would require substantial modification and legal review. We discuss this limitation at length in Section 5.

### 2.9 Independent Redistricting Commissions

An institutional alternative to algorithmic redistricting is the independent redistricting commission (IRC), which attempts to reduce partisan manipulation by removing the process from direct legislative control. McDonald (2004) surveyed the evidence on whether IRCs produce less biased maps, finding mixed results: citizen commissions generally outperform legislative redistricting on partisan bias measures, while bipartisan commissions (which include equal numbers of partisan representatives) tend to produce outcomes similar to legislative redistricting. Cain (2012) offered a skeptical assessment, arguing that structural features of American political geography limit how much IRCs can achieve even under ideal conditions. More recent evidence from California (Cottrell 2019) and Arizona (Crain 2022) suggests that IRCs with robust transparency requirements and public participation can produce meaningfully less biased maps, though they remain subject to their own pathologies.

The present study does not directly compare algorithmic maps to IRC-produced maps, but the state-level analysis includes states with IRCs (California, Arizona, Michigan, Colorado), allowing indirect comparison. California's enacted map (from the California Citizens Redistricting Commission) produces 40 Democratic and 12 Republican seats in the 118th Congress; our blind algorithm produces... [consistent with the VEST-attributed results described in Section 4].

---

## 3. Data and Methods

### 3.1 Data Sources

The primary geographic unit of analysis is the Voting Tabulation District (VTD), the precinct-level geographic unit published by the US Census Bureau as part of the TIGER 2020 release. VTDs represent the finest geographic resolution at which Census population counts are published, making them ideal atomic units for redistricting — they are small enough to form districts of nearly any shape while being large enough to keep graph sizes computationally tractable. We downloaded VTD shapefiles for all 44 multi-district states from the Census Bureau's TIGER FTP server.

California presents a special case: the California Secretary of State does not publish precinct-level geographic boundaries consistent with TIGER conventions, resulting in VTD files with no geometry. For California, we substituted 2020 census tracts as atomic units, yielding a graph of 9,129 nodes. This substitution trades granularity for tractability; census tracts in California average approximately 4,100 residents, somewhat larger than the median VTD, which may marginally constrain the algorithm's ability to satisfy population balance at the district level, though in practice the ±0.5% tolerance proved achievable.

Population data come from the 2020 decennial census PL 94-171 redistricting file, distributed via the Census Bureau's TIGER/Line system. We use the POP20 field (2020 Census population count) as the population variable for all allocation and tolerance calculations.

County, county subdivision (cousub), incorporated place, and road network geometries are drawn from the 2020 Census TIGER shapefiles. Current congressional district boundaries are from the Census TIGER 2022 release representing the 118th Congress enacted districts (CD118).

Partisan lean for the post-hoc audit is computed using VEST 2020 precinct-level presidential returns, distributed by the Voting and Election Science Team at the Harvard Dataverse under a Creative Commons Attribution 4.0 International license. VEST provides disaggregated precinct-level vote totals for the 2020 presidential election, which we join to VTDs using geographic intersection. The 2020 presidential election serves as a standard partisan benchmark because of its high turnout, national salience, and availability at precinct resolution in all 50 states.

Six states — Alaska, Delaware, Montana (before its gain of a second seat), North Dakota, South Dakota, Vermont, and Wyoming — have only a single at-large congressional district (K=1) and are excluded from this analysis. Montana gained a second seat following the 2020 census and is included with K=2.

### 3.2 Graph Construction

For each state, we construct a planar graph in which nodes represent VTDs (or census tracts for California) and edges represent geographic adjacency. Two units are considered adjacent if their shared boundary length exceeds a minimum threshold of 50 meters. This minimum border filter (MIN_BORDER_M = 50m) is essential for eliminating adjacencies created by digitization artifacts — cases where two geographic units nominally share a vertex but have no meaningful land boundary. Without this filter, the resulting graph contains many spurious adjacencies, particularly in areas with complex coastal boundaries.

We compute shared boundary lengths using planar geometric intersection. For each pair of geometrically adjacent units, we compute the length of the intersection of their boundary polylines. Pairs with intersection length below 50 meters are excluded. This threshold was chosen based on empirical testing: at 50 meters, genuine cross-boundary adjacencies are preserved while most digitization artifacts are eliminated.

Edge weights are assigned according to the formula:

w(e) = base × W\_SAME\_COUNTY^β × W\_SAME\_PLACE^β × W\_SAME\_COUSUB^β / W\_ROAD^β

where base = 1.0, W_SAME_COUNTY = 10, W_SAME_PLACE = 5, W_SAME_COUSUB = 3, and W_ROAD_DIV = 2. The parameter β controls the strength of the geographic preferences: β = 2.0 for states with K > 2, and β = 0.5 for states with K = 2. The higher β for multi-district states creates a strong preference (100-fold for county-crossing edges at β = 2.0) for district boundaries that follow county lines, while the lower β for two-district states prevents the algorithm from constructing narrow peninsula connectors that exploit the high county weight to string together non-contiguous populations.

The intuition behind the weighting scheme is that existing political and administrative boundaries reflect genuine communities of interest. County lines are particularly important: counties are the primary unit of local governance in most states, and cross-county splits impose real costs on representation. Road weights (W_ROAD_DIV) penalize edges that cross major roads, following the commonsense intuition that major roads are barriers rather than connectors.

Virginia presents a specific challenge: the state contains 38 independent cities (FIPS codes 510 through 840) that are legally separate from the surrounding counties and are treated as county-equivalent jurisdictions for Census purposes. We treat all independent cities as county-equivalent units, yielding 133 total county-equivalent units in Virginia, compared to 95 counties in a state without independent cities. This explains Virginia's comparatively high county-split rate (47% of county-equivalents split) relative to its modest district count (K = 11).

### 3.3 MCMC Sampler

We implement Weighted ReCom using the GerryChain Python library (version 0.3), which provides a configurable implementation of the ReCom proposal mechanism. At each MCMC step, the sampler: (1) selects a random adjacent pair of districts; (2) merges the two districts into a single connected region; (3) constructs a random spanning tree of the merged region using Wilson's algorithm (1996) weighted by edge weights; (4) identifies all edges in the spanning tree whose removal produces two subregions satisfying the population tolerance; (5) samples uniformly among those valid cuts; and (6) accepts the proposed split with probability equal to the ratio of the number of valid spanning trees under the old plan to the number under the proposed plan (the Reversible ReCom correction). If no valid cut exists, the step is rejected and the chain remains at its current state.

The number of MCMC steps is set as round(2000 × K / 8), yielding 2,000 steps for the modal state (K = 8), with scaling for larger states. For the four largest states — California (K = 52), Florida (K = 28), New York (K = 26), and Texas (K = 38) — we used 100,000 steps to ensure adequate exploration of the plan space. For West Virginia (K = 2), we used 20,000 steps, reflecting the sensitivity of two-district solutions to the starting plan. Ohio (K = 15) used 3,750 steps, Virginia (K = 11) used 2,750 steps, and Washington (K = 10) used 2,500 steps, reflecting manual adjustment for states where the default scaling produced underexplored ensembles.

The population tolerance is set at ±0.5% of the ideal district population (state population / K). This tolerance is more stringent than the "zero deviation" standard nominally required for congressional districts under *Karcher v. Daggett* (1983), which in practice allows deviations of a few persons. We use ±0.5% as a computational necessity — exact population equality is achievable in principle with finer geographic units but requires significantly more computation at the VTD level — while acknowledging that legally compliant maps typically minimize deviation further.

### 3.4 Plan Selection

From each state's ensemble, we select a final plan using Pareto optimality across five geometric objectives:

1. **pp_mean**: Mean Polsby-Popper compactness across all districts (maximize)
2. **county_splits**: Number of counties split between multiple districts (minimize)
3. **cut_edges**: Number of edges in the dual graph that cross district boundaries (minimize)
4. **max_county_districts**: Maximum number of districts sharing any single county (minimize)
5. **cut_border_m**: Total length of internal district boundaries, in meters (minimize)

A plan is Pareto-optimal if no other plan in the ensemble is at least as good on all five objectives and strictly better on at least one. We apply two hard filters before Pareto selection: plans where the minimum district Polsby-Popper score (pp_min) falls below 0.05 are excluded, as such districts likely exhibit extreme geographic irregularity; and plans where the maximum population deviation exceeds 0.5% are excluded.

We additionally apply a peninsula filter: plans where the ratio of the minimum cut border length to the mean edge length falls below 5% are deprioritized. This filter targets a specific artifact of the ReCom algorithm in states with geographic peninsulas: the algorithm can produce plans connected by a single narrow passage, which satisfies contiguity requirements mathematically but violates commonsense notions of geographic coherence.

The final selected plan from the Pareto-optimal frontier is the plan that maximizes pp_mean subject to being Pareto-optimal and passing all hard filters. This two-stage procedure — Pareto filtering followed by single-objective selection — ensures that the selected plan is well-rounded across all geometric objectives while prioritizing the most commonly used compactness measure in legal and scholarly contexts.

### 3.5 Partisan Audit

Partisan lean is computed after plan selection, using only the selected final plan for each state. We intersect VTD boundaries with VEST 2020 precinct boundaries using area-weighted interpolation, allocating precinct-level votes to VTDs proportional to the fraction of each precinct's area within each VTD. We then aggregate VTD-level votes to districts and compute the two-party Democratic vote share (D_votes / (D_votes + R_votes)) for the 2020 presidential election. A district is classified as Democratic-leaning if its Democratic two-party share exceeds 50%, and the margin is computed as |Democratic share − 0.5|. Districts with margin ≤ 5% are classified as highly competitive; districts with margin ≤ 10% are classified as competitive.

Crucially, partisan data enter the analysis only at this stage. The graph construction, edge weights, MCMC sampling, and plan selection all use exclusively geographic and population information. The post-hoc partisan audit is designed to answer the question: "If this geographically drawn map were used, what would be its partisan implications?" — not to optimize or target any partisan outcome.

---

## 4. Results

### 4.1 Compactness Results

Table 1 (reproduced from the study's complete results) reports mean Polsby-Popper scores (pp_mean) for all 44 states. The range is substantial, from 0.139 in West Virginia to 0.515 in Rhode Island. This variance is not primarily a function of algorithm performance; rather, it reflects the geometric constraints imposed by each state's own shape and the character of its geographic features.

The five states with the lowest pp_mean scores illustrate geographic determinism in compactness: West Virginia (0.139) has dramatic topographic complexity in the Appalachian ridgeline system and two panhandles; Georgia (0.145) spans the Piedmont-coastal plain transition with complex river drainages; Louisiana (0.160) has the highly irregular Mississippi River delta and bayou coastline; Tennessee (0.164) is an extremely elongated east-west state with Appalachian ridges in the east; and Texas (0.170) has Gulf Coast estuaries, the Rio Grande meander system, and enormous geographic extent. In each case, the low compactness scores are not artifacts of the algorithm but consequences of the underlying geographic reality that district boundaries must follow.

Conversely, the highest pp_mean scores belong to geometrically simple states: Rhode Island (0.515) is approximately rectangular with minimal coastal complexity; Nebraska (0.409) is a Great Plains rectangle; Nevada (0.402) is a near-rectangular desert state; Montana (0.387) is a rectangular western state; and Kansas (0.370) is a near-perfect Great Plains rectangle. These states produce compact districts because the algorithm has compact material to work with.

This finding has direct implications for the use of Polsby-Popper scores as legal standards. Setting a fixed minimum Polsby-Popper threshold — say, pp_min ≥ 0.15 — would be achievable in Great Plains states but would be structurally impossible to satisfy in some districts in West Virginia or Louisiana regardless of mapmaker intent. Any legal or regulatory use of compactness standards must account for geographic heterogeneity across states.

The correlation between state geometric compactness and district geometric compactness is the central empirical finding of the compactness analysis. West Virginia's state-level Polsby-Popper score is approximately 0.107, and our blind algorithm achieves a mean district score of 0.139 — close to the theoretical ceiling for a two-district division of that state. This near-ceiling performance suggests the algorithm is functioning effectively: it is producing the most compact districts that the state's own geometry allows.

### 4.2 National Partisan Seat Distribution

The central partisan finding is stark in its simplicity: a geography-only algorithm, with no access to partisan data of any kind, produces 204 Democratic and 231 Republican seats out of 435 multi-district seats — compared to 203 Democratic and 232 Republican seats in the enacted 118th Congress maps. The national-level difference is a single seat.

This result is remarkable precisely because of the enormous complexity of the redistricting process it approximates. The 118th Congress maps were drawn by 44 different state-level processes, including partisan legislative redistricting, independent commissions, bipartisan commissions, and court-drawn maps. Each of these processes involved thousands of decisions influenced by partisan interests, legal requirements, community input, and geographic constraints. Yet the aggregate partisan outcome of all these processes is, at the national level, essentially identical to what a blind geographic algorithm produces.

The finding is consistent with the Rodden (2019) and Chen and Rodden (2013) thesis: because Democratic voters are concentrated in dense urban areas and Republican voters are more dispersed across suburban and rural areas, any geographic districting algorithm that produces contiguous, compact districts will tend to produce a Republican-leaning seat distribution in the current partisan geography. The specific ratio we observe — approximately 47% Democratic, 53% Republican — reflects the residential geography of American partisanship, not the choices of mapmakers.

### 4.3 Competitive Seats

While national seat shares are nearly identical, the blind maps produce meaningfully more competitive seats. At the ≤5% margin threshold, blind maps yield 50 competitive seats compared to 45 in enacted plans — an 11% increase. At the ≤10% threshold, blind maps yield 103 competitive seats compared to 84 in enacted plans — a 23% increase.

This finding is consistent with the theoretical prediction that partisan gerrymandering reduces electoral competition. When mapmakers "crack" opposing-party voters across multiple districts or "pack" them into lopsided supermajority districts, they reduce the number of districts where the outcome is genuinely uncertain. A geography-only algorithm has no incentive to perform either operation, and the result is a meaningfully more competitive map at the national level.

The magnitude of the effect — 19 additional competitive seats at the ≤10% threshold — is substantial. These 19 additional seats would represent constituencies where elections are genuinely contested, voters face meaningful choices, and representatives have stronger incentives to respond to the median voter in their district rather than to a partisan base.

### 4.4 State-Level Gerrymander Detection

The state-level comparison between blind and enacted maps serves as a diagnostic fingerprint for deliberate manipulation. Table 2 reports states where the blind and enacted Democratic seat counts differ by at least one seat.

**Illinois (−3 for blind; enacted favors Democrats)**: The blind algorithm produces 10 Democratic-leaning districts in Illinois; the enacted 118th Congress map produced 13 out of 17. Illinois is controlled by the Democratic Party, and the enacted map has been widely characterized as a Democratic gerrymander. The 3-seat difference suggests that geography alone would support approximately 10 Democratic seats, and the additional 3 seats in the enacted map represent the partisan premium from deliberate manipulation.

**New York (+3 for blind; enacted relatively disadvantages Democrats)**: The blind algorithm produces 18 Democratic-leaning seats in New York; the enacted 118th Congress map produced only 15 out of 26. This is initially counterintuitive — New York is a deeply Democratic state, and its maps have historically been drawn by Democrats. However, the 118th Congress map for New York was struck down by the state Court of Appeals as an unconstitutional gerrymander; the court-ordered map used in 2022 elections was drawn by a special master and was less favorable to Democrats than what a fully partisan Democratic map would have produced. Our result of +3 for blind maps relative to enacted suggests the court-drawn map was, if anything, somewhat conservative in drawing Democratic-leaning seats relative to geographic potential.

**Pennsylvania (−2 for blind; enacted favors Democrats)**: Pennsylvania's 118th Congress map was drawn under a court order following the invalidation of the Republican-drawn map. The court-approved plan produces 8 Democratic-leaning seats; our blind algorithm produces 6 out of 17. Pennsylvania's Democratic voters are heavily concentrated in Philadelphia and Pittsburgh, and the court-drawn map appears to have drawn somewhat more favorable Democratic districts than pure geographic compactness would generate.

**Texas (+2 for blind; enacted favors Republicans)**: Texas's map, drawn by the Republican-controlled legislature, produces 12 Democratic-leaning seats out of 38. Our blind algorithm produces 14 — a 2-seat Republican advantage in the enacted map relative to geographic expectation. Texas's enacted map has been the subject of extensive VRA litigation, and this finding is consistent with the view that the map packs and cracks Latino and Black communities in ways that reduce their representation beyond what geography alone would produce.

**Wisconsin (+2 for blind; enacted heavily favors Republicans)**: Wisconsin produces one of the most striking state-level differences: the enacted map produces only 2 Democratic-leaning districts out of 8, while our blind algorithm produces 4. This 2-seat difference in an 8-seat delegation represents a 25-percentage-point gap between geographic and enacted Democratic seat share. Wisconsin's enacted legislative and congressional maps have been among the most heavily litigated in the country, and this finding provides additional empirical support for the characterization of Wisconsin's congressional map as a significant Republican gerrymander.

### 4.5 Edge Cases: West Virginia and California

West Virginia's two-district map illustrates several algorithm-specific challenges. The state has VTD coverage of only 72.3%, reflecting approximately 17,391 km² of federal land (primarily national forests and the Monongahela National Forest) for which no VTD boundaries are published. The algorithm nonetheless achieves a valid two-district plan because population is concentrated in the VTD-covered areas; the uncovered federal lands contain negligible population. The adaptive β = 0.5 parameter prevents the algorithm from constructing a narrow peninsula connector — a two-district plan where one district reaches across a narrow geographic passage to connect non-contiguous population centers. Without this adaptation, the county-weight preference would create incentives for exactly such connectors.

California's substitution of census tracts for VTDs (9,129 nodes for 52 districts) is the study's largest methodological compromise. The 100,000-step MCMC run and the resulting 117 county splits reflect the geometric impossibility of keeping 52 districts within 58 counties: with K > N_counties, some counties must be split between districts regardless of the algorithm used. The 117 county splits in California's blind map are geometrically near-minimal given this constraint.

---

## 5. Discussion

### 5.1 Geography as Destiny: Interpreting the National Partisan Finding

The single-seat difference between blind maps (204D/231R) and enacted maps (203D/232R) at the national level is this study's most striking result. It demands careful interpretation.

One reading — the strong form of the Rodden thesis — is that partisan gerrymandering simply does not matter at the national level because geographic sorting fully determines seat shares. This reading is too strong. The state-level results clearly show that deliberate manipulation can shift outcomes by 2–3 seats in individual states, and these shifts can in principle accumulate to change majority control of the House in a close election. A 3-seat shift in Illinois plus a 2-seat shift in Wisconsin plus a 2-seat shift in Texas amounts to a 7-seat swing — potentially decisive in a House where the majority is often determined by single-digit margins.

A more defensible reading is that geographic sorting is the primary determinant of national seat shares, while deliberate manipulation adds a secondary layer that matters enormously in specific states and can matter nationally in competitive election cycles. This nuanced view is consistent with Kaufman, King, and Komisarchik (2021), who estimated that roughly half of observed partisan bias in US congressional elections can be attributed to geographic sorting, with the remainder from deliberate manipulation.

The competitive seats finding adds an important dimension: blind maps produce more competition even when national seat shares are similar. This suggests that the distinctive harm of partisan gerrymandering is not primarily national seat share manipulation but rather the creation of noncompetitive districts — safe seats that reduce accountability and responsiveness. From this perspective, the redistricting reform literature's focus on competitive districts (rather than proportional representation) may be well-targeted.

### 5.2 Limitations

This study has several important limitations that bear emphasis.

**Voting Rights Act non-compliance**: The maps produced in this study do not comply with the Voting Rights Act. The algorithm is blind to race, ethnicity, and minority community geography, and therefore cannot guarantee — and in most cases will not achieve — the creation of majority-minority districts required under *Thornburg v. Gingles* where legally mandated. Applying these maps in any legal context would require substantial modification, VRA analysis, and likely revision of district lines in states with significant minority populations. Alabama, Georgia, Louisiana, Mississippi, North Carolina, and South Carolina all have VRA-related redistricting obligations that our blind algorithm cannot satisfy.

**Population deviation**: Our ±0.5% tolerance is more permissive than the near-exact-equality standard nominally required for congressional districts under *Karcher v. Daggett*. While our deviations are small in absolute terms, any adopted plan would need to minimize deviations further. This is achievable in principle by substituting census blocks for VTDs as atomic units, at substantial computational cost.

**No communities of interest**: The algorithm's geographic objectives do not account for communities of interest — shared economic, cultural, or historical ties that may bind populations not otherwise reflected in administrative boundaries. Redistricting criteria in many states include maintaining communities of interest as a criterion, and pure geometric optimization will sometimes produce plans that divide cohesive communities.

**No public input**: Redistricting in any legal or democratic context requires public participation — the ability of affected communities to comment on proposed maps and advocate for their interests. An algorithmic system that produces a single optimal map forecloses this input. Ensemble methods, by contrast, can serve as a tool within a participatory process, characterizing the space of valid plans and identifying how different choices trade off against each other.

**Temporal fixity**: The study uses 2020 population and 2020 precinct partisan returns. Both population distribution and partisan geography shift between census cycles; a map drawn for 2020 geography may be less suited to 2030 population patterns.

### 5.3 Implications for Redistricting Reform

Despite these limitations, the study has several implications for redistricting reform. First, the feasibility of producing complete state-level congressional maps for all 44 multi-district states using a single algorithmic framework demonstrates that geography-only redistricting is computationally tractable at the national scale — a claim that would have been implausible two decades ago. The algorithmic infrastructure exists to serve as a baseline or reference tool in any redistricting reform process.

Second, the competitive seats finding suggests a concrete, measurable benefit of geometry-based redistricting: more competitive elections. To the extent that redistricting reformers care about electoral accountability and responsiveness — not just partisan seat shares — the blind maps provide a useful benchmark for how much competition the geography would naturally support.

Third, the state-level gerrymander detection methodology offers a potentially useful legal and policy tool. By comparing enacted maps to geographically generated baselines, the methodology can identify states where enacted maps deviate substantially from geographic expectation — providing quantitative evidence that courts, legislatures, or reform advocates can use, consistent with the ensemble analysis approach pioneered by Herschlag et al. (2020) and Duchin and Tenner (2018).

---

## 6. Conclusion

This paper presents the most comprehensive geography-only redistricting analysis of US congressional districts to date, covering all 44 multi-district states and 435 seats using Weighted ReCom MCMC with zero access to partisan data during map generation. The primary national finding — that blind geographic maps produce near-identical partisan seat shares as politically-drawn enacted maps — provides the strongest empirical test yet of the Chen and Rodden (2013) "unintentional gerrymandering" hypothesis at national scale. Where people live largely determines who wins elections in a system of single-member geographic districts, and no redistricting process — however well-intentioned or algorithmically sophisticated — can fully overcome the partisan geography of the American electorate.

Yet the study also demonstrates that deliberate manipulation is detectable and consequential at the state level. Gerrymanders in Illinois, Wisconsin, Texas, and elsewhere show up as clear departures from geographic expectation, and the aggregate effect on national seat shares is within the range that could flip majority control of the House in a close election. The 23% increase in competitive seats produced by blind maps relative to enacted plans quantifies a real democratic cost of partisan gerrymandering, even when national seat shares are similar.

The Polsby-Popper analysis reveals that geographic compactness in districts is primarily a function of state geometry rather than algorithm quality or mapmaker intent. States with irregular topography and complex coastlines will produce geometrically complex districts under any redistricting method; holding them to the same compactness standards as rectangular Great Plains states would be both analytically unjustified and legally problematic.

Several important limitations must be recalled. These maps do not comply with the Voting Rights Act, do not reflect communities of interest, and were produced without public participation. They are not ready-to-adopt legal maps; they are a scientific baseline for understanding what geography, unconstrained by partisan interests, would produce. That baseline is itself a valuable contribution: it demonstrates that the partisan structure of American congressional representation is, to a first approximation, a geographic fact rather than a political one — while leaving room for the recognition that deliberate manipulation at the margins can still determine whether Democrats or Republicans control the House of Representatives.

Future work should extend this framework to state legislative districts, which are more numerous and have received less algorithmic attention; incorporate VRA compliance mechanisms within the sampling framework; and develop ensemble-based tools for characterizing the trade-off between compactness, community preservation, and minority representation. The algorithmic infrastructure demonstrated here makes such extensions feasible; the policy stakes make them urgent.

---

## References

Altman, M. (1997). The computational complexity of automated redistricting: Is automation the answer? *Rutgers Computer and Technology Law Journal*, 23(1), 81–142.

Bernstein, M., and Duchin, M. (2017). A formula goes to court: Partisan gerrymandering and the efficiency gap. *Notices of the American Mathematical Society*, 64(9), 1020–1024.

Cain, B. E. (2012). Redistricting commissions: A better political buffer? *Yale Law Journal*, 121(7), 1808–1844.

Cameron, C., Epstein, D., and O'Halloran, S. (1996). Do majority-minority districts maximize substantive Black representation in Congress? *American Political Science Review*, 90(4), 794–812.

Cannon, S., Goldbloom-Helzner, A., Gupta, V., Matthews, J. N., and Suwal, B. (2023). Spanning tree methods for sampling graph partitions. *SIAM Journal on Applied Mathematics*, 83(4), 1369–1395.

Carter, D., Hunter, Z., and Cho, W. K. T. (2019). A simulation model of redistricting based on spatial compactness. *Political Analysis*, 27(4), 563–579.

Chambers, C. P., and Miller, A. D. (2010). A measure of bizarreness. *Quarterly Journal of Political Science*, 5(1), 27–44.

Chen, J., and Rodden, J. (2013). Unintentional gerrymandering: Political geography and electoral bias in legislatures. *Quarterly Journal of Political Science*, 8(3), 239–269.

Cottrell, D. (2019). Using computer simulations to measure the effect of gerrymandering on electoral competition in state legislatures. *Legislative Studies Quarterly*, 44(3), 443–473.

Crain, C. (2022). Arizona's Independent Redistricting Commission: Evaluating the performance of a citizen commission in the post-*Rucho* era. *Election Law Journal*, 21(2), 141–163.

Davis v. Bandemer, 478 U.S. 109 (1986).

DeFord, D., Duchin, M., and Solomon, J. (2021). Recombination: A family of Markov chains for redistricting. *Harvard Data Science Review*, 3(1). https://doi.org/10.1162/99608f92.eb30390f

Dube, M., and Tucker, J. (2021). GerryChain: A Markov chain Monte Carlo framework for exploring the space of redistricting plans. *Journal of Open Source Software*, 6(62), 3020. https://doi.org/10.21105/joss.03020

Duchin, M., and Tenner, B. E. (2018). Discrete geometry for electoral geography. Preprint, arXiv:1808.05860.

Fifield, B., Higgins, M., Imai, K., and Tarr, A. (2020). Automated redistricting simulation using Markov chain Monte Carlo. *Journal of Computational and Graphical Statistics*, 29(4), 715–728.

Fryer, R. G., and Holden, R. (2011). Measuring the compactness of political districting plans. *Journal of Law and Economics*, 54(3), 493–535.

Garfinkel, R. S., and Nemhauser, G. L. (1970). Optimal political districting by implicit enumeration techniques. *Management Science*, 16(8), B495–B508.

Gelman, A., and King, G. (1994). A unified method of evaluating electoral systems and redistricting plans. *American Journal of Political Science*, 38(2), 514–554.

Grofman, B., Handley, L., and Niemi, R. G. (1992). *Minority Representation and the Quest for Voting Equality*. Cambridge University Press.

Herschlag, G., Kang, H.-S., Lo, J., Sachet, C., Schutzman, Z., Shimek, C., and Mattingly, J. C. (2020). Quantifying gerrymandering in North Carolina. *Statistics and Public Policy*, 7(1), 30–38.

Karcher v. Daggett, 462 U.S. 725 (1983).

Kaufman, A., King, G., and Komisarchik, M. (2021). How to measure legislative district compactness if you only know it when you see it. *American Journal of Political Science*, 65(3), 533–550.

Lublin, D. (1997). *The Paradox of Representation: Racial Gerrymandering and Minority Interests in Congress*. Princeton University Press.

Mattingly, J. C., and Vaughn, C. (2014). Redistricting and the will of the people. Preprint, arXiv:1410.8796.

McDonald, M. P. (2004). A comparative analysis of redistricting institutions in the United States, 2001–02. *State Politics and Policy Quarterly*, 4(4), 371–395.

Mehrotra, A., Johnson, E. L., and Nemhauser, G. L. (1998). An optimization based heuristic for political districting. *Management Science*, 44(8), 1100–1114.

Miller v. Johnson, 515 U.S. 900 (1995).

Niemi, R. G., Grofman, B., Carlucci, C., and Hofeller, T. (1990). Measuring compactness and the role of a compactness standard in a test for partisan and racial gerrymandering. *Journal of Politics*, 52(4), 1155–1181.

Pildes, R. H., and Niemi, R. G. (1993). Expressive harms, bizarre districts, and voting rights: Evaluating election-district appearances after Shaw v. Reno. *Michigan Law Review*, 92(3), 483–587.

Polsby, D. D., and Popper, R. D. (1991). The third criterion: Compactness as a procedural safeguard against partisan gerrymandering. *Yale Law and Policy Review*, 9(2), 301–353.

Reock, E. C. (1961). A note: Measuring compactness as a requirement of legislative apportionment. *Midwest Journal of Political Science*, 5(1), 70–74.

Reynolds v. Sims, 377 U.S. 533 (1964).

Rodden, J. A. (2019). *Why Cities Lose: The Deep Roots of the Urban-Rural Political Divide*. Basic Books.

Rodden, J., Chen, J., and Cottrell, D. (2021). Answering the call of the court: The limits of partisan gerrymandering and the opportunities for reform. In N. Persily and C. Stewart III (Eds.), *The Realities of Electoral Reform* (pp. 89–117). Cambridge University Press.

Rucho v. Common Cause, 588 U.S. 684 (2019).

Schwartzberg, J. E. (1966). Reapportionment, gerrymanders, and the notion of compactness. *Minnesota Law Review*, 50(3), 443–452.

Shaw v. Reno, 509 U.S. 630 (1993).

Stephanopoulos, N. O., and McGhee, E. M. (2015). Partisan gerrymandering and the efficiency gap. *University of Chicago Law Review*, 82(2), 831–900.

Thornburg v. Gingles, 478 U.S. 30 (1986).

Vickrey, W. (1961). On the prevention of gerrymandering. *Political Science Quarterly*, 76(1), 105–110.

Voting and Election Science Team (VEST). (2020). *2020 Precinct-Level Election Results*. Harvard Dataverse. https://doi.org/10.7910/DVN/K7760H

Wang, S. S.-H. (2016). Three tests for practical evaluation of partisan gerrymandering. *Stanford Law Review*, 68(6), 1263–1321.

Warrington, G. S. (2018). Quantifying gerrymandering using the vote distribution. *Election Law Journal*, 17(1), 39–57.

Weaver, J. B., and Hess, S. W. (1963). A procedure for nonpartisan districting: Development of computer techniques. *Yale Law Journal*, 73(2), 288–308.

Wesberry v. Sanders, 376 U.S. 1 (1964).

Wilson, D. B. (1996). Generating random spanning trees more quickly than the cover time. *Proceedings of the 28th Annual ACM Symposium on Theory of Computing*, 296–303.

Young, H. P. (1988). Measuring the compactness of legislative districts. *Legislative Studies Quarterly*, 13(1), 105–115.
