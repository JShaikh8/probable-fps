"""
Daily MLB projection pipeline.

Order:
  0. Reconcile yesterday's projections vs actuals (calibration feedback)
  1. Ingest new completed games
  2. Rebuild pitch splits + pitcher profiles
  3. Rebuild park factors
  4. Rebuild pitcher season stats (FIP etc.)
  5. Rebuild hitter recent form + spray profiles
  6. Rebuild hitter + pitcher archetypes / similarity
  7. Build today's hitter + pitcher projections
  8. Build today's NRFI projections

Usage:
    python run_daily.py
    python run_daily.py --date 2026-04-18
    python run_daily.py --skip-ingest
    python run_daily.py --only-projections
    python run_daily.py --skip-reconcile
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)


def _header(step: str):
    print(f'\n{"=" * 60}')
    print(f'  {step}')
    print(f'{"=" * 60}')


def step_reconcile(yesterday: str):
    _header(f'Step 0 · Reconcile {yesterday}')
    from model.build_reconciliation import run
    run(yesterday)


def step_ingest(seasons: list[int], force: bool = False):
    _header('Step 1 · Ingest new game feeds')
    from ingest.ingest_runner import ingest_season
    for season in sorted(seasons):
        ingest_season(season, force=force)


def step_pitch_splits():
    _header('Step 2 · Pitch splits & pitcher profiles')
    from features.build_pitch_splits import run
    run()


def step_park_factors():
    _header('Step 3 · Park factors')
    from features.build_park_factors import run
    run()


def step_pitcher_season_stats():
    _header('Step 4 · Pitcher season stats')
    from features.build_pitcher_game_stats import run
    run()


def step_hitter_ctx():
    _header('Step 5 · Hitter recent form + spray profiles')
    from features.build_hitter_recent_form import run as run_form
    from features.build_hitter_spray_profile import run as run_spray
    run_form()
    run_spray()


def step_archetypes():
    _header('Step 6 · Hitter & pitcher archetypes')
    from model.build_archetypes import run
    run()


def step_projections(proj_date: str):
    _header(f'Step 7 · Projections for {proj_date}')
    from model.build_projections import run
    run(proj_date)


def step_nrfi(proj_date: str):
    _header(f'Step 8 · NRFI projections for {proj_date}')
    from model.build_nrfi_projections import run
    run(proj_date)


def step_ml(proj_date: str, retrain: bool = False):
    _header(f'Step 9 · ML matchup classifiers + factor tuner ({"retrain + " if retrain else ""}predict) for {proj_date}')
    from model.ml_matchup import train as train_hitter, predict as predict_hitter
    from model.ml_pitcher_matchup import train as train_pitcher, predict as predict_pitcher
    from model.ml_pipeline import build_training_set, train as train_tuner, predict as predict_tuner
    if retrain:
        train_hitter()
        train_pitcher()
        df = build_training_set()
        train_tuner(df)
    predict_hitter(proj_date)
    predict_pitcher(proj_date)
    predict_tuner(proj_date)      # factor → tuned + blend


def main():
    parser = argparse.ArgumentParser(description='Daily MLB projection pipeline')
    parser.add_argument('--date', default=None,
                        help='Projection date (YYYY-MM-DD). Defaults to today (US/Central).')
    parser.add_argument('--seasons', nargs='+', type=int, default=None,
                        help='Seasons to ingest. Defaults to DEFAULT_SEASONS.')
    parser.add_argument('--force-ingest', action='store_true',
                        help='Re-ingest already-done games')
    parser.add_argument('--skip-ingest',     action='store_true')
    parser.add_argument('--skip-features',   action='store_true')
    parser.add_argument('--skip-reconcile',  action='store_true')
    parser.add_argument('--only-projections', action='store_true',
                        help='Skip ingest + features + reconcile; just projections')
    parser.add_argument('--retrain-ml', action='store_true',
                        help='Retrain ML models from historical data')
    parser.add_argument('--skip-export', action='store_true',
                        help="Don't write UI JSON snapshots after projections")
    args = parser.parse_args()

    if args.date:
        proj_date = args.date
    else:
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo('America/Chicago')
            proj_date = datetime.now(tz).strftime('%Y-%m-%d')
        except Exception:
            proj_date = date.today().isoformat()

    yesterday = (date.fromisoformat(proj_date) - timedelta(days=1)).isoformat()

    print(f'\nProjection date: {proj_date}')

    skip_ingest    = args.skip_ingest    or args.only_projections
    skip_features  = args.skip_features  or args.only_projections
    skip_reconcile = args.skip_reconcile or args.only_projections

    t0 = time.time()

    if not skip_reconcile:
        step_reconcile(yesterday)

    if not skip_ingest:
        from config import DEFAULT_SEASONS
        seasons = args.seasons or DEFAULT_SEASONS
        step_ingest(seasons, force=args.force_ingest)

    if not skip_features:
        step_pitch_splits()
        step_park_factors()
        step_pitcher_season_stats()
        step_hitter_ctx()
        step_archetypes()

    step_projections(proj_date)
    step_nrfi(proj_date)
    step_ml(proj_date, retrain=args.retrain_ml)

    # Step 10 — export JSON snapshots for the static Render UI. Idempotent;
    # runs after every pipeline so the UI is always one `git push` away from
    # being up to date. Skip with --skip-export if you only wanted DB writes.
    if not args.skip_export:
        _header('Step 10 · Export JSON snapshots for UI')
        from scripts.export_for_ui import main as export_main
        import sys as _sys
        _saved_argv = _sys.argv
        _sys.argv = ['export_for_ui', '--days', '7']
        try:
            export_main()
        finally:
            _sys.argv = _saved_argv

    elapsed = time.time() - t0
    print(f'\n✓ Pipeline complete in {elapsed:.1f}s\n')


if __name__ == '__main__':
    main()
