"""
pipeline/download.py

Generalized download step for the 50-state congressional redistricting pipeline.

Downloads per-state TIGER 2020 geography files needed by pipeline/build_graph.py:
    - VTD precincts
    - Census blocks (tabblock20, includes POP20)
    - Places (incorporated cities/towns)
    - County subdivisions (minor civil divisions)
    - Current congressional districts (CD118; skipped for at-large states k==1)

National files (counties, primary roads) are shared across all states and are
assumed to be already present at the paths passed in by the caller.

Usage:
    from state_configs import ALL_STATES
    from pipeline.download import download_state
    from pathlib import Path

    cfg = ALL_STATES["24"]          # Maryland
    result = download_state(cfg, Path("data"), skip_water=True)

STRICTLY GEOGRAPHY-ONLY — ZERO PARTISAN OR DEMOGRAPHIC DATA
"""

import hashlib
import json
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from state_configs import StateConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path) -> bool:
    """Stream-download url to dest.  Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"    ERROR downloading {url}: {exc}")
        return False

    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=dest.name, leave=False,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
            bar.update(len(chunk))
    return True


def _unzip_file(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _save_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)


def _download_dataset(
    name: str,
    url: str,
    subdir: Path,
    manifest: dict,
    manifest_path: Path,
) -> bool:
    """
    Download and unzip a single dataset.  Idempotent — skips if already OK.
    Returns True on success.
    """
    if manifest.get(name, {}).get("status") == "OK":
        sha_prefix = manifest[name].get("sha256", "")[:12]
        print(f"  [{name}] Already downloaded — skipping ({sha_prefix}…)")
        return True

    filename = url.split("/")[-1]
    dest_zip = subdir / filename
    subdir.mkdir(parents=True, exist_ok=True)

    print(f"  [{name}] Downloading: {url}")
    ok = _download_file(url, dest_zip)
    if not ok:
        manifest[name] = {"url": url, "status": "FAILED", "sha256": None}
        _save_manifest(manifest, manifest_path)
        return False

    checksum = _sha256_file(dest_zip)
    print(f"  [{name}] SHA-256: {checksum[:24]}…")

    if dest_zip.suffix == ".zip":
        print(f"  [{name}] Extracting to {subdir}/")
        _unzip_file(dest_zip, subdir)

    manifest[name] = {
        "url": url,
        "local_zip": str(dest_zip),
        "subdir": str(subdir),
        "sha256": checksum,
        "status": "OK",
    }
    _save_manifest(manifest, manifest_path)
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_state(
    cfg: StateConfig,
    data_root: Path,
    skip_water: bool = True,
) -> dict:
    """
    Download all geography files for one state.

    Parameters
    ----------
    cfg        : StateConfig entry for the target state.
    data_root  : Root of data directory (e.g. Path("data")).
                 Per-state files land in data_root / cfg.abbr.lower() / ...
    skip_water : If True (default), skip per-county water downloads.
                 Pass False only if you need water-based edge weights.

    Returns
    -------
    manifest dict (also written to data_root/{abbr}/manifest.json).
    """
    fips = cfg.fips
    abbr = cfg.abbr.lower()
    state_dir = data_root / abbr
    state_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = state_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        with open(manifest_path) as fh:
            manifest = json.load(fh)

    print(f"[{cfg.abbr}] Downloading geography for {cfg.name} (FIPS {fips})")

    # 1. VTD precincts — TIGER2020PL uses STATE/{FIPS}_{STATENAME}/{FIPS}/ subdirs.
    #    Some states (CA, HI, MO, OR, ...) don't publish VTD data; fall back to
    #    census tracts (TIGER2020/TRACT/tl_2020_{fips}_tract.zip) in that case.
    tiger_state_name = cfg.name.upper().replace(" ", "_")
    vtd_url = (
        f"https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/"
        f"{fips}_{tiger_state_name}/{fips}/tl_2020_{fips}_vtd20.zip"
    )
    vtd_ok = _download_dataset(
        name="vtd_precincts",
        url=vtd_url,
        subdir=state_dir / "vtd_precincts",
        manifest=manifest,
        manifest_path=manifest_path,
    )
    if not vtd_ok:
        print(f"  [{cfg.abbr}] VTD not available — falling back to census tracts")
        tract_url = (
            f"https://www2.census.gov/geo/tiger/TIGER2020/TRACT/"
            f"tl_2020_{fips}_tract.zip"
        )
        _download_dataset(
            name="census_tracts",
            url=tract_url,
            subdir=state_dir / "census_tracts",
            manifest=manifest,
            manifest_path=manifest_path,
        )

    # 2. Census blocks (tabblock20 includes POP20 — no separate PL file needed).
    _download_dataset(
        name="census_blocks",
        url=(
            f"https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/"
            f"tl_2020_{fips}_tabblock20.zip"
        ),
        subdir=state_dir / "census_blocks",
        manifest=manifest,
        manifest_path=manifest_path,
    )

    # 3. Places (incorporated cities and towns).
    _download_dataset(
        name="places",
        url=(
            f"https://www2.census.gov/geo/tiger/TIGER2020/PLACE/"
            f"tl_2020_{fips}_place.zip"
        ),
        subdir=state_dir / "places",
        manifest=manifest,
        manifest_path=manifest_path,
    )

    # 4. County subdivisions (minor civil divisions).
    _download_dataset(
        name="county_subdivisions",
        url=(
            f"https://www2.census.gov/geo/tiger/TIGER2020/COUSUB/"
            f"tl_2020_{fips}_cousub.zip"
        ),
        subdir=state_dir / "county_subdivisions",
        manifest=manifest,
        manifest_path=manifest_path,
    )

    # 5. Current congressional districts (CD118); skip at-large states.
    if cfg.k > 1:
        _download_dataset(
            name="current_districts",
            url=(
                f"https://www2.census.gov/geo/tiger/TIGER2022/CD/"
                f"tl_2022_{fips}_cd118.zip"
            ),
            subdir=state_dir / "current_districts",
            manifest=manifest,
            manifest_path=manifest_path,
        )
    else:
        manifest["current_districts"] = {
            "status": "N/A",
            "note": f"At-large state (k=1) — no CD118 file downloaded",
        }
        _save_manifest(manifest, manifest_path)

    # 6. Water layers — per-county; skip unless explicitly requested.
    if skip_water:
        for layer_key in ("water_area", "water_linear"):
            if layer_key not in manifest:
                manifest[layer_key] = {
                    "status": "SKIPPED",
                    "note": "skip_water=True; water edge weights disabled",
                }
        _save_manifest(manifest, manifest_path)
    else:
        _download_per_county_water(cfg, state_dir, manifest, manifest_path)

    # Summary.
    n_ok = sum(1 for v in manifest.values() if v.get("status") in ("OK", "N/A", "SKIPPED"))
    n_fail = sum(1 for v in manifest.values() if v.get("status") == "FAILED")
    print(
        f"[{cfg.abbr}] Download complete — {n_ok} OK/skipped, {n_fail} failed. "
        f"Manifest: {manifest_path}"
    )

    return manifest


# ---------------------------------------------------------------------------
# Optional: per-county water download (skip_water=False path)
# ---------------------------------------------------------------------------


def _get_county_fips_list(cfg: StateConfig, county_shp: Path) -> list[str]:
    """
    Return list of full county FIPS codes (5-digit) for this state by reading
    the national county shapefile filtered to cfg.fips.
    """
    import geopandas as gpd

    counties = gpd.read_file(county_shp)
    state_col = next(
        (c for c in counties.columns if c.upper() in ("STATEFP", "STATEFP20")),
        None,
    )
    if state_col is None:
        raise ValueError(f"Cannot find STATEFP column in {county_shp}")
    state_counties = counties[counties[state_col] == cfg.fips]
    geoid_col = next(
        (c for c in state_counties.columns if c.upper() in ("GEOID", "GEOID20")),
        None,
    )
    if geoid_col is None:
        # Build GEOID from STATEFP + COUNTYFP.
        county_fp_col = next(
            (c for c in state_counties.columns if c.upper() in ("COUNTYFP", "COUNTYFP20")),
            None,
        )
        if county_fp_col is None:
            raise ValueError(f"Cannot derive county FIPS from {county_shp}")
        return [
            cfg.fips + row[county_fp_col]
            for _, row in state_counties.iterrows()
        ]
    return list(state_counties[geoid_col].values)


def _download_per_county_water(
    cfg: StateConfig,
    state_dir: Path,
    manifest: dict,
    manifest_path: Path,
    county_shp: Path | None = None,
) -> None:
    """Download AREAWATER and LINEARWATER files for every county in the state."""
    if county_shp is None:
        # Try to find it under data/maryland/counties/ (pre-downloaded national file).
        default_shp = Path("data/maryland/counties/tl_2020_us_county.shp")
        if not default_shp.exists():
            print(
                f"  [{cfg.abbr}] WARNING: county shapefile not found at {default_shp};"
                " cannot download per-county water. Pass county_shp explicitly."
            )
            return
        county_shp = default_shp

    try:
        county_fips_list = _get_county_fips_list(cfg, county_shp)
    except Exception as exc:
        print(f"  [{cfg.abbr}] WARNING: could not build county FIPS list: {exc}")
        return

    for layer, key in [("AREAWATER", "water_area"), ("LINEARWATER", "water_linear")]:
        subdir = state_dir / key
        subdir.mkdir(parents=True, exist_ok=True)

        if manifest.get(key, {}).get("status") == "OK":
            print(f"  [{cfg.abbr}] [{key}] Already downloaded — skipping")
            continue

        print(f"  [{cfg.abbr}] [{key}] Downloading {len(county_fips_list)} county files…")
        failed: list[str] = []
        for county_fips in tqdm(county_fips_list, desc=f"    {layer}"):
            filename = f"tl_2020_{county_fips}_{layer.lower()}.zip"
            url = f"https://www2.census.gov/geo/tiger/TIGER2020/{layer}/{filename}"
            dest_zip = subdir / filename
            if dest_zip.exists():
                continue
            ok = _download_file(url, dest_zip)
            if ok:
                _unzip_file(dest_zip, subdir / county_fips)
            else:
                failed.append(county_fips)

        if failed:
            print(f"  [{cfg.abbr}] [{key}] WARNING: failed counties: {failed}")
            manifest[key] = {"status": "PARTIAL", "failed": failed}
        else:
            manifest[key] = {
                "status": "OK",
                "subdir": str(subdir),
                "n_counties": len(county_fips_list),
            }
        _save_manifest(manifest, manifest_path)
