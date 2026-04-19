"""
Nightly reconciliation — compare yesterday's projections to final box scores.

For each hitter projection on `game_date`, reads the actual at-bats from that
game, computes actual DK/FD pts and stat totals, and writes the diff into
projection_actuals. Same for pitcher projections. Also reconciles NRFI
predictions against actual first-inning runs.

This is the feedback loop the pipeline was missing — once this runs nightly,
you can surface trailing MAE / Brier / calibration charts in the UI.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from config import get_engine, get_session
from db.io import bulk_upsert
from db.models import NrfiActual, PitcherProjectionActual, ProjectionActual


def _ni(v) -> int:
    """NaN-safe int cast."""
    if v is None:
        return 0
    try:
        if pd.isna(v):
            return 0
    except (TypeError, ValueError):
        pass
    return int(v)


def _nf(v) -> float:
    if v is None:
        return 0.0
    try:
        if pd.isna(v):
            return 0.0
    except (TypeError, ValueError):
        pass
    return float(v)


HIT_EVENTS = {'single', 'double', 'triple', 'home_run'}
WALK_EVENTS = {'walk', 'intent_walk'}
OUT_EVENTS = {
    'strikeout', 'strikeout_double_play', 'field_out', 'force_out',
    'grounded_into_double_play', 'double_play', 'triple_play',
    'sac_fly', 'sac_bunt', 'sac_fly_double_play', 'fielders_choice_out',
}


def run(game_date: str | None = None):
    if game_date is None:
        game_date = (date.today() - timedelta(days=1)).isoformat()

    engine = get_engine()
    print(f'Reconciling projections for {game_date}…')

    _reconcile_hitters(engine, game_date)
    _reconcile_pitchers(engine, game_date)
    _reconcile_nrfi(engine, game_date)

    print('Done.')


# ── Hitter reconciliation ─────────────────────────────────────────────

def _reconcile_hitters(engine, game_date: str):
    proj = pd.read_sql_query(
        f"""
        SELECT hitter_id, game_pk, game_date, proj, dk_pts
        FROM projections
        WHERE game_date = '{game_date}'
        """,
        engine,
    )
    if proj.empty:
        print('  hitters: no projections for date')
        return

    actual = pd.read_sql_query(
        f"""
        SELECT ab.game_pk, ab.hitter_id, ab.event_type, ab.rbi
        FROM at_bats ab
        JOIN games g ON g.game_pk = ab.game_pk AND g.status = 'final'
        WHERE g.game_date = '{game_date}'
          AND ab.hitter_id = ANY(
              SELECT DISTINCT hitter_id FROM projections WHERE game_date = '{game_date}'
          )
        """,
        engine,
    )
    if actual.empty:
        print('  hitters: no completed games yet for this date')
        return

    actual['is_hit']    = actual['event_type'].isin(HIT_EVENTS)
    actual['is_single'] = actual['event_type'] == 'single'
    actual['is_double'] = actual['event_type'] == 'double'
    actual['is_triple'] = actual['event_type'] == 'triple'
    actual['is_hr']     = actual['event_type'] == 'home_run'
    actual['is_bb']     = actual['event_type'].isin(WALK_EVENTS)
    actual['is_k']      = actual['event_type'].isin({'strikeout', 'strikeout_double_play'})

    rollup = actual.groupby(['game_pk', 'hitter_id']).agg(
        pa=('event_type', 'count'),
        h=('is_hit', 'sum'),
        single=('is_single', 'sum'),
        double=('is_double', 'sum'),
        triple=('is_triple', 'sum'),
        hr=('is_hr', 'sum'),
        bb=('is_bb', 'sum'),
        k=('is_k', 'sum'),
        rbi=('rbi', 'sum'),
    ).reset_index()

    # DraftKings hitter scoring: 1B=3, 2B=5, 3B=8, HR=10, RBI=2, R=2, BB=2, HBP=2, K=-0.5, SB=5
    # Pull runs + SB from hitter_game_stats (populated by ingest/fetch_runs.py).
    extra = pd.read_sql_query(
        f"""
        SELECT hitter_id, game_pk,
               runs, stolen_bases, caught_stealing
        FROM hitter_game_stats
        WHERE game_pk IN (
          SELECT DISTINCT game_pk FROM games WHERE game_date = '{game_date}'
        )
        """,
        engine,
    )
    rollup = rollup.merge(extra, on=['game_pk', 'hitter_id'], how='left')
    rollup['runs'] = rollup['runs'].fillna(0).astype(int)
    rollup['stolen_bases'] = rollup['stolen_bases'].fillna(0).astype(int)
    rollup['caught_stealing'] = rollup['caught_stealing'].fillna(0).astype(int)

    rollup['actual_dk_pts'] = (
        rollup['single'] * 3 + rollup['double'] * 5 +
        rollup['triple'] * 8 + rollup['hr'] * 10 +
        rollup['rbi']    * 2 + rollup['bb'] * 2 +
        rollup['runs']   * 2 +
        rollup['stolen_bases'] * 5 +
        rollup['caught_stealing'] * -2 +
        rollup['k']      * -0.5
    ).round(2)

    merged = proj.merge(rollup, on=['game_pk', 'hitter_id'], how='left')
    merged['actual_dk_pts'] = merged['actual_dk_pts'].fillna(0.0)
    merged['dk_error'] = merged['actual_dk_pts'] - merged['dk_pts']
    merged['abs_dk_error'] = merged['dk_error'].abs()

    records = []
    for _, r in merged.iterrows():
        records.append({
            'hitter_id':    int(r['hitter_id']),
            'game_pk':      int(r['game_pk']),
            'game_date':    game_date,
            'proj_dk_pts':  float(r['dk_pts']),
            'actual_dk_pts': float(r['actual_dk_pts']),
            'proj':         r['proj'] or {},
            'actual': {
                'pa':  _ni(r.get('pa')),
                'h':   _ni(r.get('h')),
                'hr':  _ni(r.get('hr')),
                'bb':  _ni(r.get('bb')),
                'k':   _ni(r.get('k')),
                'rbi': _ni(r.get('rbi')),
                'r':   _ni(r.get('runs')),
                'sb':  _ni(r.get('stolen_bases')),
            },
            'dk_error':     round(float(r['dk_error']), 2),
            'abs_dk_error': round(float(r['abs_dk_error']), 2),
        })

    session = get_session()
    try:
        bulk_upsert(session, ProjectionActual, records,
                    pk_cols=['hitter_id', 'game_pk'])
        session.commit()
        print(f'  hitters:  {len(records)} reconciled  (MAE {merged["abs_dk_error"].mean():.2f})')
    finally:
        session.close()


# ── Pitcher reconciliation ────────────────────────────────────────────

def _reconcile_pitchers(engine, game_date: str):
    proj = pd.read_sql_query(
        f"""
        SELECT pitcher_id, game_pk, game_date, proj, dk_pts
        FROM pitcher_projections
        WHERE game_date = '{game_date}'
        """,
        engine,
    )
    if proj.empty:
        print('  pitchers: no projections')
        return

    actual = pd.read_sql_query(
        f"""
        SELECT ab.game_pk, ab.pitcher_id, ab.event_type
        FROM at_bats ab
        JOIN games g ON g.game_pk = ab.game_pk AND g.status = 'final'
        WHERE g.game_date = '{game_date}'
          AND ab.pitcher_id = ANY(
              SELECT DISTINCT pitcher_id FROM pitcher_projections WHERE game_date = '{game_date}'
          )
        """,
        engine,
    )
    if actual.empty:
        print('  pitchers: no completed games yet')
        return

    actual['is_k']   = actual['event_type'].isin({'strikeout', 'strikeout_double_play'})
    actual['is_bb']  = actual['event_type'].isin(WALK_EVENTS)
    actual['is_h']   = actual['event_type'].isin(HIT_EVENTS)
    actual['is_hr']  = actual['event_type'] == 'home_run'
    actual['is_out'] = actual['event_type'].isin(OUT_EVENTS)

    g = actual.groupby(['game_pk', 'pitcher_id']).agg(
        bf=('event_type', 'count'),
        k=('is_k', 'sum'),
        bb=('is_bb', 'sum'),
        h=('is_h', 'sum'),
        hr=('is_hr', 'sum'),
        outs=('is_out', 'sum'),
    ).reset_index()
    g['ip'] = g['outs'] / 3.0

    # DK pitcher: IP=2.25, K=2, H=-0.6, BB=-0.6, HR=0 (ER=-2 but we lack ER)
    g['actual_dk_pts'] = (
        g['ip'] * 2.25 + g['k'] * 2.0 + g['h'] * -0.6 + g['bb'] * -0.6
    ).round(2)

    merged = proj.merge(g, on=['game_pk', 'pitcher_id'], how='left')
    merged['actual_dk_pts'] = merged['actual_dk_pts'].fillna(0.0)
    merged['dk_error'] = merged['actual_dk_pts'] - merged['dk_pts']
    merged['abs_dk_error'] = merged['dk_error'].abs()

    records = []
    for _, r in merged.iterrows():
        records.append({
            'pitcher_id':   int(r['pitcher_id']),
            'game_pk':      int(r['game_pk']),
            'game_date':    game_date,
            'proj_dk_pts':  float(r['dk_pts']),
            'actual_dk_pts': float(r['actual_dk_pts']),
            'proj':         r['proj'] or {},
            'actual': {
                'ip':  round(_nf(r.get('ip')), 2),
                'k':   _ni(r.get('k')),
                'bb':  _ni(r.get('bb')),
                'h':   _ni(r.get('h')),
                'hr':  _ni(r.get('hr')),
            },
            'dk_error':     round(float(r['dk_error']), 2),
            'abs_dk_error': round(float(r['abs_dk_error']), 2),
        })

    session = get_session()
    try:
        bulk_upsert(session, PitcherProjectionActual, records,
                    pk_cols=['pitcher_id', 'game_pk'])
        session.commit()
        print(f'  pitchers: {len(records)} reconciled  (MAE {merged["abs_dk_error"].mean():.2f})')
    finally:
        session.close()


# ── NRFI reconciliation ───────────────────────────────────────────────

def _reconcile_nrfi(engine, game_date: str):
    proj = pd.read_sql_query(
        f"""
        SELECT game_pk, nrfi_prob
        FROM nrfi_projections
        WHERE game_date = '{game_date}'
        """,
        engine,
    )
    if proj.empty:
        print('  nrfi: no projections')
        return

    # First-inning runs from at_bats: count RBI contributions in inning=1
    fi = pd.read_sql_query(
        f"""
        SELECT ab.game_pk, ab.half_inning, SUM(ab.rbi) AS runs
        FROM at_bats ab
        JOIN games g ON g.game_pk = ab.game_pk AND g.status = 'final'
        WHERE g.game_date = '{game_date}'
          AND ab.inning = 1
        GROUP BY ab.game_pk, ab.half_inning
        """,
        engine,
    )
    if fi.empty:
        print('  nrfi: no completed games yet')
        return

    pivot = fi.pivot_table(index='game_pk', columns='half_inning',
                           values='runs', aggfunc='sum').fillna(0)
    pivot['home_fi_runs'] = pivot.get('bottom', 0)
    pivot['away_fi_runs'] = pivot.get('top', 0)
    pivot['actual_nrfi'] = (pivot['home_fi_runs'] + pivot['away_fi_runs']) == 0
    pivot = pivot.reset_index()

    merged = proj.merge(pivot[['game_pk', 'home_fi_runs', 'away_fi_runs', 'actual_nrfi']],
                        on='game_pk', how='inner')

    records = []
    for _, r in merged.iterrows():
        records.append({
            'game_pk':             int(r['game_pk']),
            'game_date':           game_date,
            'predicted_nrfi_prob': float(r['nrfi_prob']),
            'actual_nrfi':         bool(r['actual_nrfi']),
            'home_fi_runs':        int(r['home_fi_runs']),
            'away_fi_runs':        int(r['away_fi_runs']),
        })

    session = get_session()
    try:
        bulk_upsert(session, NrfiActual, records,
                    pk_cols=['game_pk'])
        session.commit()
        # Simple Brier score
        brier = ((merged['nrfi_prob'] - merged['actual_nrfi'].astype(float)) ** 2).mean()
        print(f'  nrfi:     {len(records)} reconciled  (Brier {brier:.4f})')
    finally:
        session.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='YYYY-MM-DD (default: yesterday)')
    args = parser.parse_args()
    run(game_date=args.date)
