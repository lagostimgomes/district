"""
run_all_states.py

Parallel driver for the 50-state congressional redistricting pipeline.

Usage:
    # Run all multi-district states (k > 1) in parallel:
    python run_all_states.py

    # Run a single state by abbreviation:
    python run_all_states.py --state MD

    # Override the Maryland baseline step count (default 2000):
    python run_all_states.py --base-steps 5000

    # Override number of parallel workers:
    python run_all_states.py --workers 4

MCMC steps scale proportionally with K (number of districts):
    steps(state) = round(base_steps * K / K_maryland)   where K_maryland = 8

This keeps sampling effort proportional to the state's complexity.

Each state run executes four pipeline stages:
    1. download_state  — download TIGER 2020 geography files
    2. build_graph     — build weighted precinct dual graph
    3. run_sampling    — run weighted ReCom MCMC sampler
    4. select_maps     — select Pareto-optimal maps

Progress and errors are logged to logs/{abbr}_pipeline.log.
At-large states (k == 1: AK, DE, ND, SD, VT, WY) are skipped.

STRICTLY GEOGRAPHY-ONLY — ZERO PARTISAN OR DEMOGRAPHIC DATA
"""

import argparse
import logging
import multiprocessing
import time
import traceback
from pathlib import Path

from state_configs import ALL_STATES, STATES_BY_ABBR, StateConfig
from pipeline.download import download_state
from pipeline.build_graph import build_graph
from pipeline.sample import run_sampling
from pipeline.select import select_maps

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data")
LOGS_DIR = Path("logs")

# Maryland baseline: K=8 districts, 2 000 steps.
# All other states scale as: steps = round(BASE_STEPS * K / K_MD_BASELINE)
K_MD_BASELINE = 8
DEFAULT_BASE_STEPS = 2_000


def steps_for_state(k: int, base_steps: int = DEFAULT_BASE_STEPS) -> int:
    """Return the number of MCMC steps for a state with k districts."""
    return max(base_steps, round(base_steps * k / K_MD_BASELINE))

# National files pre-downloaded under data/maryland/ (shared across all states).
COUNTY_SHP = Path("data/maryland/counties/tl_2020_us_county.shp")
ROADS_SHP = Path("data/maryland/roads/tl_2020_us_primaryroads.shp")

# ---------------------------------------------------------------------------
# Per-state runner
# ---------------------------------------------------------------------------


def _setup_state_logger(abbr: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{abbr}_pipeline.log"
    logger = logging.getLogger(f"pipeline.{abbr}")
    logger.setLevel(logging.DEBUG)
    # File handler.
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    # Only add handler if not already present (guard against multiprocessing reuse).
    if not logger.handlers:
        logger.addHandler(fh)
    return logger


def run_state(
    fips: str,
    base_steps: int = DEFAULT_BASE_STEPS,
    n_steps: int | None = None,
) -> tuple[str, str, float]:
    """
    Run the full pipeline for one state.

    Parameters
    ----------
    fips       : 2-digit state FIPS code.
    base_steps : Maryland baseline; actual steps = round(base_steps * K / 8).
    n_steps    : Direct step count override — ignores base_steps scaling.

    Returns
    -------
    (abbr, status, elapsed_seconds)
    """
    cfg: StateConfig = ALL_STATES[fips]
    abbr = cfg.abbr
    if n_steps is None:
        n_steps = steps_for_state(cfg.k, base_steps)
    logger = _setup_state_logger(abbr)
    t0 = time.time()

    logger.info(f"=== Starting pipeline for {cfg.name} ({abbr}) — K={cfg.k}, steps={n_steps:,} ===")

    try:
        # Stage 1: download.
        logger.info("Stage 1: download")
        download_state(cfg, DATA_ROOT, skip_water=True)
        logger.info("Stage 1 complete")

        # Stage 2: build graph.
        logger.info("Stage 2: build_graph")
        build_graph(cfg, DATA_ROOT, COUNTY_SHP, ROADS_SHP, skip_water=True)
        logger.info("Stage 2 complete")

        # Stage 3: sample.
        logger.info(f"Stage 3: run_sampling ({n_steps:,} steps)")
        run_sampling(cfg, DATA_ROOT, n_steps=n_steps)
        logger.info("Stage 3 complete")

        # Stage 4: select maps.
        logger.info("Stage 4: select_maps")
        select_maps(cfg, DATA_ROOT)
        logger.info("Stage 4 complete")

        elapsed = time.time() - t0
        logger.info(f"=== Pipeline complete for {abbr} in {elapsed:.1f}s ===")
        return (abbr, "OK", elapsed)

    except Exception as exc:
        elapsed = time.time() - t0
        err_msg = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        logger.error(f"Pipeline FAILED for {abbr}: {err_msg}")
        logger.debug(tb)
        return (abbr, f"FAILED: {err_msg}", elapsed)


def _run_state_wrapper(args: tuple) -> tuple[str, str, float]:
    """Wrapper for multiprocessing.Pool.map (only takes one argument)."""
    fips, base_steps, steps_override = args
    return run_state(fips, base_steps=base_steps, n_steps=steps_override)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 50-state congressional redistricting pipeline."
    )
    parser.add_argument(
        "--state",
        metavar="ABBR",
        action="append",
        help="Run specific state(s) by 2-letter abbreviation. May be repeated (e.g. --state TX --state FL).",
    )
    parser.add_argument(
        "--base-steps",
        type=int,
        default=DEFAULT_BASE_STEPS,
        metavar="N",
        help=(
            f"Maryland baseline MCMC step count (default: {DEFAULT_BASE_STEPS:,}). "
            "Each state's steps = round(base_steps * K / 8)."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        metavar="N",
        help="Override: use exactly N steps for all specified states, ignoring proportional scaling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        metavar="N",
        help="Number of parallel workers (default: 6).",
    )
    args = parser.parse_args()

    base_steps    = args.base_steps
    steps_override = args.steps
    n_workers     = args.workers

    # Determine which states to run.
    if args.state:
        states_to_run = []
        for abbr_raw in args.state:
            abbr_upper = abbr_raw.upper()
            if abbr_upper not in STATES_BY_ABBR:
                parser.error(
                    f"Unknown state abbreviation '{abbr_upper}'. "
                    f"Valid values: {sorted(STATES_BY_ABBR.keys())}"
                )
            cfg = STATES_BY_ABBR[abbr_upper]
            if cfg.k == 1:
                print(
                    f"{cfg.name} ({abbr_upper}) is an at-large state (k=1) — "
                    "no redistricting to perform."
                )
                continue
            states_to_run.append(cfg)
        if not states_to_run:
            return
    else:
        # All multi-district states, sorted by k ascending (small states finish first).
        states_to_run = sorted(
            [cfg for cfg in ALL_STATES.values() if cfg.k > 1],
            key=lambda c: c.k,
        )

    print("=" * 68)
    print("50-State Congressional Redistricting Pipeline")
    print("=" * 68)
    print(f"States to run : {len(states_to_run)}")
    if steps_override:
        print(f"Steps         : {steps_override:,}  (direct override — no scaling)")
    else:
        print(f"Base steps    : {base_steps:,}  (Maryland, K=8)")
        print(f"Step scaling  : steps = round(base_steps × K / 8)")
    print(f"Workers       : {n_workers}")
    print(f"Data root     : {DATA_ROOT.resolve()}")
    print(f"County SHP    : {COUNTY_SHP}")
    print(f"Roads SHP     : {ROADS_SHP}")
    print("=" * 68)
    print(f"\n{'State':<6} {'K':>4}  {'Steps':>8}")
    print("-" * 22)
    for cfg in states_to_run:
        n = steps_override if steps_override else steps_for_state(cfg.k, base_steps)
        print(f"{cfg.abbr:<6} {cfg.k:>4}  {n:>8,}")
    print()

    run_args = [(cfg.fips, base_steps, steps_override) for cfg in states_to_run]
    t_start = time.time()

    if len(states_to_run) == 1:
        # Single state: run directly (no pool overhead, better tracebacks).
        results = [_run_state_wrapper(run_args[0])]
    else:
        with multiprocessing.Pool(processes=n_workers) as pool:
            results = pool.map(_run_state_wrapper, run_args)

    elapsed_total = time.time() - t_start

    # Summary table.
    print()
    print("=" * 68)
    print("Pipeline Summary")
    print("=" * 68)
    print(f"{'State':<8} {'Status':<50} {'Elapsed':>8}")
    print("-" * 68)
    n_ok = 0
    n_fail = 0
    for abbr, status, elapsed in sorted(results, key=lambda r: r[0]):
        status_short = status if len(status) <= 48 else status[:45] + "..."
        marker = "OK" if status == "OK" else "FAIL"
        print(f"{abbr:<8} [{marker}] {status_short:<46} {elapsed:>7.1f}s")
        if status == "OK":
            n_ok += 1
        else:
            n_fail += 1

    print("-" * 68)
    print(
        f"Total: {n_ok} OK, {n_fail} FAILED  |  "
        f"Wall time: {elapsed_total:.1f}s  |  Base steps: {base_steps:,}"
    )
    print("=" * 68)
    print(f"Logs: {LOGS_DIR.resolve()}/{{abbr}}_pipeline.log")

    if n_fail:
        print(
            f"\nWARNING: {n_fail} state(s) failed. "
            "Check the per-state log files for details."
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
