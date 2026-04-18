"""
Daily morning pipeline — run once before games start each day.

Order:
  1. Ingest any new completed games from recent seasons
  2. Rebuild pitch splits + pitcher profiles
  3. Rebuild park factors
  4. Rebuild hitter archetypes / similarity matrix
  5. Build today's projections

Usage:
  python run_daily.py                   # full pipeline for today
  python run_daily.py --date 2025-04-10 # project a specific date
  python run_daily.py --skip-ingest     # skip step 1 (features + projections only)
  python run_daily.py --only-projections # skip all feature rebuilds
"""

import argparse
import os
import sys
import time
from datetime import date

# Ensure the python/ directory is on the path so all submodules can find config
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from config import get_db


def _header(step: str):
    print(f'\n{"="*60}')
    print(f'  {step}')
    print(f'{"="*60}')


def step_ingest(seasons: list[int], force: bool = False):
    _header('Step 1 · Ingest new game feeds')
    from ingest.ingest_runner import ingest_season, ensure_indexes
    db = get_db()
    ensure_indexes(db)
    for season in sorted(seasons):
        ingest_season(db, season, force=force)


def step_pitch_splits():
    _header('Step 2 · Build pitch splits & pitcher profiles')
    from features.build_pitch_splits import run
    run()


def step_park_factors():
    _header('Step 3 · Build park factors')
    from features.build_park_factors import run
    run()


def step_archetypes():
    _header('Step 4 · Build hitter archetypes')
    from model.build_archetypes import run
    run()


def step_projections(proj_date: str):
    _header(f'Step 5 · Build projections for {proj_date}')
    from model.build_projections import run
    run(proj_date)


def main():
    parser = argparse.ArgumentParser(description='Daily MLB projection pipeline')
    parser.add_argument('--date', default=None,
                        help='Projection date (YYYY-MM-DD). Defaults to today (US/Central).')
    parser.add_argument('--seasons', nargs='+', type=int, default=None,
                        help='Seasons to ingest. Defaults to config DEFAULT_SEASONS.')
    parser.add_argument('--force-ingest', action='store_true',
                        help='Re-ingest already-done games')
    parser.add_argument('--skip-ingest', action='store_true',
                        help='Skip game ingestion (step 1)')
    parser.add_argument('--skip-features', action='store_true',
                        help='Skip feature rebuilds (steps 2-4)')
    parser.add_argument('--only-projections', action='store_true',
                        help='Only run projections (skip steps 1-4)')
    args = parser.parse_args()

    # Resolve today's date in US/Central if not provided
    if args.date:
        proj_date = args.date
    else:
        try:
            import zoneinfo
            from datetime import datetime
            tz = zoneinfo.ZoneInfo('America/Chicago')
            proj_date = datetime.now(tz).strftime('%Y-%m-%d')
        except Exception:
            proj_date = date.today().isoformat()

    print(f'\nProjection date: {proj_date}')

    skip_ingest    = args.skip_ingest or args.only_projections
    skip_features  = args.skip_features or args.only_projections

    t0 = time.time()

    # ── Step 1: ingest ────────────────────────────────────────────────
    if not skip_ingest:
        from config import DEFAULT_SEASONS
        seasons = args.seasons or DEFAULT_SEASONS
        step_ingest(seasons, force=args.force_ingest)

    # ── Steps 2-4: features ───────────────────────────────────────────
    if not skip_features:
        step_pitch_splits()
        step_park_factors()
        step_archetypes()

    # ── Step 5: projections ───────────────────────────────────────────
    step_projections(proj_date)

    elapsed = time.time() - t0
    print(f'\n✓ Pipeline complete in {elapsed:.1f}s\n')


if __name__ == '__main__':
    main()
