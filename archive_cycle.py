"""
archive_cycle.py

Snapshot the current redistricting cycle into a self-contained archive that
can be restored, compared, and re-run after future redistricting events.

Usage
-----
    python archive_cycle.py                  # archives to data/archive/cycle_2024/
    python archive_cycle.py --cycle 2026     # use a specific cycle label
    python archive_cycle.py --tarball        # also create a .tar.gz for upload
    python archive_cycle.py --tarball --release  # create + upload a GitHub Release

Archive layout
--------------
data/archive/
  cycle_2024/
    manifest.json          ← committed to git (checksums, sources, provenance)
    summaries/             ← committed to git (small JSON outputs)
      lean_2024.json
      enacted_lean_2024.json
      diversity_results.json
    finals/                ← NOT committed (binaries); included in tarball
      al/  ar/  az/ ...    ← best_map_compact.gpkg + stats + report.json
    enacted_districts/     ← NOT committed; reference copies of tl_202X_XX_cd*.shp
    data_sources.txt       ← committed; URLs + checksums for re-downloading raw data
  cycle_2026/              ← created by the next cycle run
  ...
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT    = Path("data")
ARCHIVE_ROOT = DATA_ROOT / "archive"   # binary artifacts — NOT in git (too large)
CYCLES_ROOT  = Path("cycles")          # provenance metadata — committed to git
DOCS_MAPS    = Path("docs/maps")


# ── Utilities ──────────────────────────────────────────────────────────────────

def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


# ── VEST data sources (for reference / re-download) ───────────────────────────

VEST_BASE_URL = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/K7760H"
TIGER_BASE_URL = "https://www2.census.gov/geo/tiger/TIGER2020/VTD/"
ENACTED_BASE_URL = "https://www2.census.gov/geo/tiger/TIGER2022/CD/"

DATA_SOURCES = f"""# Data Sources — Redistricting Archive
# Generated automatically by archive_cycle.py

## VEST 2020 Precinct-Level Election Data
Source  : Harvard Dataverse
DOI     : 10.7910/DVN/K7760H
URL     : {VEST_BASE_URL}
Version : 2020 General Election (state shapefiles with G20PREDBID / G20PRERTRU columns)
Local   : data/vest/{{abbr}}/{{abbr}}_2020.shp  (one directory per state)
Note    : Download via compute_lean_2024.py or manually from Dataverse.
          Files are NOT included in git due to size (1.9 GB total).

## TIGER 2020 Census Geography (Voting Tabulation Districts)
Source  : US Census Bureau TIGER/Line
URL     : {TIGER_BASE_URL}
Files   : tl_2020_{{fips}}_vtd20.zip  (one per state)
Local   : data/{{abbr}}/graph/{{abbr}}_precincts_pop.gpkg  (processed output)
Note    : Downloaded automatically by pipeline/download.py.

## TIGER 2020 County and Road Shapefiles
County  : https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip
Roads   : https://www2.census.gov/geo/tiger/TIGER2020/PRIMARYROADS/tl_2020_us_primaryroads.zip
Local   : data/maryland/counties/  and  data/maryland/roads/

## Enacted 118th Congress Districts
Source  : US Census Bureau TIGER/Line
URL     : {ENACTED_BASE_URL}
Files   : tl_2022_{{fips}}_cd118.zip  (one per state)
Local   : data/{{abbr}}/current_districts/tl_2022_{{fips}}_cd118.shp
Note    : For future cycles, replace with tl_2024_*, tl_2032_*, etc.
          Update RESULTS_2024 dict in compute_lean_2024.py with new election results.

## 2020 Census PL 94-171 Redistricting Data
Source  : US Census Bureau
URL     : https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/
Format  : Pipe-delimited; P2 data at field indices 76-86 of segment 1 file
Local   : data/census/{{abbr}}/  (downloaded by compute_diversity.py)
Note    : For the 2030 cycle, use the 2030 PL 94-171 data when available.

## 2024 Presidential Election Results
Source  : Associated Press / certified state canvasses
Embedded: compute_lean_2024.py → RESULTS_2024 dict (two-party Harris share per state)
Note    : For future cycles, update RESULTS_2024 with the most recent presidential results.
          The uniform-swing method requires only the state-level two-party share.
"""


# ── Main archive function ──────────────────────────────────────────────────────

def archive_cycle(cycle: str, make_tarball: bool = False, make_release: bool = False):
    cycle_dir     = ARCHIVE_ROOT / f"cycle_{cycle}"   # binary outputs (untracked)
    cycle_meta    = CYCLES_ROOT  / cycle               # provenance metadata (tracked in git)
    print(f"\n=== Archiving cycle '{cycle}' ===")
    print(f"    Binary outputs → {cycle_dir}  (not in git)")
    print(f"    Provenance     → {cycle_meta}  (committed to git)\n")

    summaries_dir = cycle_meta    / "summaries"   # tracked
    finals_dir    = cycle_dir     / "finals"       # untracked (binary)
    enacted_dir   = cycle_dir     / "enacted_districts"  # untracked (binary)
    maps_dir      = cycle_dir     / "maps"         # untracked (binary)

    for d in [summaries_dir, finals_dir, enacted_dir, maps_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = {
        "cycle": cycle,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "git_commit": _git_commit(),
        "summaries": {},
        "finals": {},
        "enacted_districts": {},
        "maps": {},
        "notes": (
            "To restore: copy summaries/ JSON files to data/, copy finals/{abbr}/ to "
            "data/{abbr}/final/, copy enacted_districts/{abbr}/ to "
            "data/{abbr}/current_districts/, then re-run render scripts."
        ),
    }

    # ── 1. Summary JSONs ─────────────────────────────────────────────────────
    print("Copying summary JSON files…")
    for name in ["lean_2024.json", "enacted_lean_2024.json", "diversity_results.json"]:
        src = DATA_ROOT / name
        dst = summaries_dir / name
        if copy_if_exists(src, dst):
            manifest["summaries"][name] = {
                "size_bytes": dst.stat().st_size,
                "sha256": sha256(dst),
            }
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} not found — run compute scripts first")

    # ── 2. Per-state finals ───────────────────────────────────────────────────
    print("\nCopying per-state final outputs…")
    state_dirs = sorted([d for d in DATA_ROOT.iterdir()
                         if d.is_dir() and (d / "final").exists()
                         and d.name not in ("archive", "census", "vest", "maryland")])

    for state_dir in state_dirs:
        abbr = state_dir.name.upper()
        final_src = state_dir / "final"
        final_dst = finals_dir / state_dir.name

        # Key files to copy
        files_to_copy = [
            "best_map_compact.gpkg",
            "best_map_fewest_splits.gpkg",
            "best_map_compact_stats.json",
            "best_map_fewest_splits_stats.json",
            "report.json",
            "pareto_frontier.csv",
            "map_vs_enacted.png",
        ]
        copied = {}
        for fname in files_to_copy:
            src = final_src / fname
            dst = final_dst / fname
            if copy_if_exists(src, dst):
                copied[fname] = {"size_bytes": dst.stat().st_size, "sha256": sha256(dst)}

        if copied:
            manifest["finals"][abbr] = copied
            print(f"  ✓ {abbr}  ({len(copied)} files)")
        else:
            print(f"  ✗ {abbr}  (no final outputs found)")

    # ── 3. Enacted district shapefiles ───────────────────────────────────────
    print("\nCopying enacted district shapefiles…")
    for state_dir in state_dirs:
        enacted_src = state_dir / "current_districts"
        if not enacted_src.exists():
            continue
        enacted_dst = enacted_dir / state_dir.name
        enacted_dst.mkdir(parents=True, exist_ok=True)
        copied = {}
        for f in enacted_src.iterdir():
            dst = enacted_dst / f.name
            shutil.copy2(f, dst)
            copied[f.name] = {"size_bytes": dst.stat().st_size, "sha256": sha256(dst)}
        if copied:
            manifest["enacted_districts"][state_dir.name.upper()] = {
                "files": list(copied.keys()),
                "primary_shp": next((k for k in copied if k.endswith(".shp")), None),
            }
            print(f"  ✓ {state_dir.name.upper()}")

    # ── 4. Rendered national maps ─────────────────────────────────────────────
    print("\nCopying rendered national maps…")
    for png in sorted(DOCS_MAPS.glob("us_map*.png")):
        dst = maps_dir / png.name
        shutil.copy2(png, dst)
        manifest["maps"][png.name] = {"size_bytes": dst.stat().st_size, "sha256": sha256(dst)}
        print(f"  ✓ {png.name}")

    # Also copy the diversity comparison
    div_src = DOCS_MAPS / "diversity_comparison.png"
    if div_src.exists():
        shutil.copy2(div_src, maps_dir / "diversity_comparison.png")
        manifest["maps"]["diversity_comparison.png"] = {
            "size_bytes": (maps_dir / "diversity_comparison.png").stat().st_size,
            "sha256": sha256(maps_dir / "diversity_comparison.png"),
        }
        print("  ✓ diversity_comparison.png")

    # ── 5. Data sources reference (tracked) ──────────────────────────────────
    (cycle_meta / "data_sources.txt").write_text(DATA_SOURCES)
    print("\nWrote data_sources.txt")

    # ── 6. Manifest (tracked) ─────────────────────────────────────────────────
    manifest_path = cycle_meta / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_finals = len(manifest["finals"])
    n_enacted = len(manifest["enacted_districts"])
    n_maps = len(manifest["maps"])
    n_summaries = len(manifest["summaries"])
    total_size = sum(
        v.get("size_bytes", 0)
        for section in [manifest["summaries"], manifest["maps"]]
        for v in section.values()
    )
    for state_files in manifest["finals"].values():
        for v in state_files.values():
            total_size += v.get("size_bytes", 0)

    print(f"\n✓ Manifest written: {manifest_path}")
    print(f"  {n_summaries} summary JSONs  →  {cycle_meta}/summaries/  (tracked in git)")
    print(f"  {n_finals} states with final maps  →  {cycle_dir}/finals/  (binary, not in git)")
    print(f"  {n_enacted} states with enacted district files  →  {cycle_dir}/enacted_districts/")
    print(f"  {n_maps} national map PNGs  →  {cycle_dir}/maps/")
    print(f"  Total archived size: {total_size / 1e6:.1f} MB")

    # ── 7. Optional tarball (of the full binary archive) ──────────────────────
    if make_tarball:
        tarball_path = ARCHIVE_ROOT / f"cycle_{cycle}.tar.gz"
        print(f"\nCreating tarball: {tarball_path} …")
        import tarfile
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(cycle_dir, arcname=f"cycle_{cycle}")
        size_mb = tarball_path.stat().st_size / 1e6
        print(f"  ✓ {tarball_path}  ({size_mb:.1f} MB)")

        if make_release:
            _create_github_release(cycle, tarball_path)

    print(f"\n{'='*60}")
    print(f"Archive complete.")
    print(f"Commit provenance to git:")
    print(f"  git add cycles/{cycle}/")
    print(f"  git commit -m 'Archive redistricting cycle {cycle}'")
    print(f"  git push")
    if make_tarball:
        tarball_path = ARCHIVE_ROOT / f"cycle_{cycle}.tar.gz"
        print(f"\nUpload binary archive to GitHub Releases:")
        print(f"  gh release create cycle-{cycle} {tarball_path} \\")
        print(f"    --title 'Redistricting Cycle {cycle}' \\")
        print(f"    --notes 'See cycles/{cycle}/manifest.json for provenance details'")
    print(f"{'='*60}\n")

    return cycle_dir


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _create_github_release(cycle: str, tarball: Path):
    tag = f"cycle-{cycle}"
    title = f"Redistricting Cycle {cycle} — Final Maps & Data"
    notes = (
        f"## Redistricting Cycle {cycle}\n\n"
        f"Complete snapshot of blind redistricting results for all 44 multi-district US states.\n\n"
        f"### Contents\n"
        f"- `finals/` — selected district GeoPackages (.gpkg), compactness stats, Pareto frontier CSVs\n"
        f"- `enacted_districts/` — 118th Congress enacted shapefiles for comparison\n"
        f"- `summaries/` — lean_2024.json, enacted_lean_2024.json, diversity_results.json\n"
        f"- `maps/` — rendered national maps\n"
        f"- `data_sources.txt` — URLs and instructions to re-download raw VEST/TIGER data\n\n"
        f"### Reproducing from scratch\n"
        f"```bash\ngit checkout cycle-{cycle}\npython run_all_states.py\npython compute_lean_2024.py\n"
        f"python compute_enacted_lean.py\npython compute_diversity.py\n```\n\n"
        f"See CYCLES.md for full instructions."
    )
    try:
        cmd = [
            "gh", "release", "create", tag,
            str(tarball),
            "--title", title,
            "--notes", notes,
        ]
        subprocess.run(cmd, check=True)
        print(f"  ✓ GitHub Release created: {tag}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ GitHub Release creation failed: {e}")
        print(f"    Upload manually: gh release create {tag} {tarball}")
    except FileNotFoundError:
        print("  ✗ gh CLI not found — upload tarball manually to GitHub Releases")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archive current redistricting cycle data.")
    parser.add_argument("--cycle", default="2024",
                        help="Cycle label, e.g. '2024', '2026', '2030' (default: 2024)")
    parser.add_argument("--tarball", action="store_true",
                        help="Create a .tar.gz of the archive for upload")
    parser.add_argument("--release", action="store_true",
                        help="Upload the tarball as a GitHub Release (requires gh CLI)")
    args = parser.parse_args()

    if args.release and not args.tarball:
        args.tarball = True

    archive_cycle(args.cycle, make_tarball=args.tarball, make_release=args.release)
