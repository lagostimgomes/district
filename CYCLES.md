# Redistricting Cycle History

This document tracks each redistricting cycle run through the blind algorithm pipeline.
Run `python archive_cycle.py` after any significant redistricting event to create a snapshot.

---

## When to re-run

US Congressional redistricting is triggered by:
1. **Decennial Census** (2020 → enacted 2022, next: 2030 → enacted ~2032)
2. **Court orders** — states frequently redraw under litigation (NC, OH, NY, TX have all been ordered to redraw mid-cycle)
3. **State-initiated redraws** — some states permit voluntary re-draws between census years

A practical rule: **after each November election**, check whether any state's congressional map has changed since the last run. If yes, re-run that state and update the comparison data.

```bash
# Check for new enacted district shapefiles from Census
# TIGER files are published at: https://www2.census.gov/geo/tiger/TIGER{YEAR}/CD/

# Re-run a single state whose map was redrawn
python run_all_states.py --state NC --steps 20000

# Update partisan lean for all states
python compute_lean_2024.py          # update RESULTS_2024 dict first with new election results
python compute_enacted_lean.py       # update enacted district files first

# Archive the new cycle
python archive_cycle.py --cycle 2026 --tarball
# Then upload tarball to GitHub:
gh release create cycle-2026 data/archive/cycle_2026.tar.gz \
  --title "Redistricting Cycle 2026" \
  --notes "See cycles/2026/manifest.json"
```

---

## Cycle Archive

### Cycle 2024 — Initial Run

| Item | Value |
|------|-------|
| **Run date** | May 2026 |
| **Git tag** | `cycle-2024` |
| **Election data** | 2024 certified presidential results (AP / state canvasses) |
| **Precinct data** | VEST 2020 (Harvard Dataverse DOI: 10.7910/DVN/K7760H) |
| **Enacted districts** | 118th Congress — TIGER 2022 `tl_2022_XX_cd118.shp` |
| **Census data** | 2020 PL 94-171 redistricting file |
| **States completed** | 44 multi-district + 6 at-large = 50 |
| **MCMC steps** | 2,000–20,000 (scales with K; WV re-run at 20,000) |
| **Provenance (git)** | `cycles/2024/` — manifest.json, summaries/, data_sources.txt |
| **Binary archive** | `data/archive/cycle_2024/` (local) + GitHub Release `cycle-2024` |

**Key results (2024 cycle):**

| | Blind Algorithm | Enacted 118th |
|---|---|---|
| D seats (429 redistricted) | 202 (47.1%) | 203 (47.3%) |
| R seats (429 redistricted) | 227 (52.9%) | 226 (52.7%) |
| Competitive ≤ 8% | **80** | 70 |
| Toss-up ≤ 2% | **20** (10D/10R) | 17 (5D/12R) |
| Diversity mean (Herfindahl) | **0.521** | 0.511 |

---

## How to restore a prior cycle

```bash
# Download the tarball from GitHub Releases
gh release download cycle-2024 --pattern "*.tar.gz" --dir /tmp/
tar -xzf /tmp/cycle_2024.tar.gz -C data/archive/

# Restore final maps for all states
for abbr in $(ls data/archive/cycle_2024/finals/); do
    mkdir -p data/$abbr/final
    cp data/archive/cycle_2024/finals/$abbr/* data/$abbr/final/
done

# Restore enacted district files
for abbr in $(ls data/archive/cycle_2024/enacted_districts/); do
    mkdir -p data/$abbr/current_districts
    cp data/archive/cycle_2024/enacted_districts/$abbr/* data/$abbr/current_districts/
done

# Restore JSON summaries
cp data/archive/cycle_2024/summaries/*.json data/

# Re-render site assets (no re-sampling needed)
python render_partisan_spectrum.py
python render_margin_maps.py
python render_all_state_comparisons.py
```

---

## How to run a new cycle from scratch

### Step 1 — Update election results
Edit `compute_lean_2024.py` and update `RESULTS_2024` with the new presidential two-party shares.
For a new file, copy and rename to e.g. `compute_lean_2028.py`.

### Step 2 — Update enacted district shapefiles
Download the new TIGER CD shapefiles:
```bash
# Example for 2030 cycle:
# https://www2.census.gov/geo/tiger/TIGER2032/CD/
# Place in data/{abbr}/current_districts/ replacing existing tl_*_cd*.shp files
```

### Step 3 — Re-run the pipeline
```bash
# Full re-run (clears existing ensemble + final outputs):
rm -rf data/*/ensemble data/*/final
python run_all_states.py --workers 6

# Or update only specific states:
python run_all_states.py --state NC --state OH --steps 20000
```

### Step 4 — Recompute partisan and diversity analyses
```bash
python compute_lean_YYYY.py
python compute_enacted_lean.py
python compute_diversity.py
```

### Step 5 — Re-render and deploy
```bash
python render_partisan_spectrum.py
python render_margin_maps.py
python render_all_state_comparisons.py

# Copy maps to docs/
cp data/us_map*.png docs/maps/
for abbr in al ar az ca co ct fl ga hi ia id il in ks ky la ma md me mi mn mo ms mt nc ne nh nj nm nv ny oh ok or pa ri sc tn tx ut va wa wi wv; do
    cp data/$abbr/final/map_vs_enacted.png docs/maps/$abbr/
done

git add docs/ && git commit -m "Update site for redistricting cycle YYYY"
git push
```

### Step 6 — Archive the cycle
```bash
python archive_cycle.py --cycle YYYY --tarball
git add cycles/YYYY/
git commit -m "Archive redistricting cycle YYYY"
git push
# Upload binary archive (229 MB) to GitHub Releases:
gh release create cycle-YYYY data/archive/cycle_YYYY.tar.gz \
  --title "Redistricting Cycle YYYY" \
  --notes "See cycles/YYYY/manifest.json"
```

---

## What is and isn't tracked in git

| Data | Tracked? | Reason |
|------|----------|--------|
| Python scripts | ✅ Yes | Source of truth |
| `docs/` (HTML + PNGs) | ✅ Yes | Published website |
| `cycles/YYYY/manifest.json` | ✅ Yes | Lightweight provenance, checksums |
| `cycles/YYYY/summaries/*.json` | ✅ Yes | Key computed outputs (~200 KB) |
| `cycles/YYYY/data_sources.txt` | ✅ Yes | Re-download instructions |
| `data/archive/cycle_YYYY/finals/` | ❌ No (too large) | Stored in GitHub Release tarball |
| `data/archive/cycle_YYYY/enacted_districts/` | ❌ No (too large) | Stored in GitHub Release tarball |
| `data/vest/` | ❌ No (1.9 GB) | Re-download from Harvard Dataverse |
| `data/*/graph/` | ❌ No (~1 GB) | Rebuild with `pipeline/build_graph.py` |
| `data/*/ensemble/` | ❌ No (~200 MB) | Rebuild with `pipeline/sample.py` |
| `logs/` | ❌ No | Runtime output |
