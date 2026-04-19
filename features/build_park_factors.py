"""
Build park factors per venue from at_bats.
Writes: park_factors (one row per venue_id).
"""
from __future__ import annotations

import pandas as pd

from config import get_engine, get_session
from db.models import ParkFactor
from db.io import bulk_upsert


def run(seasons: list[int] | None = None):
    engine = get_engine()

    where = f"AND ab.season IN ({','.join(str(s) for s in seasons)})" if seasons else ''

    print('Loading at-bats for park factors…')
    df = pd.read_sql_query(
        f"""
        SELECT g.venue_id, ab.event_type, ab.exit_velocity, ab.launch_angle,
               ab.hit_coord_x, ab.hit_coord_y, ab.season
        FROM at_bats ab
        JOIN games g ON g.game_pk = ab.game_pk
        WHERE g.venue_id IS NOT NULL
        {where}
        """,
        engine,
    )

    if df.empty:
        print('No data. Run ingest first.')
        return

    print(f'  {len(df):,} at-bats across {df["venue_id"].nunique()} venues')

    df['is_hr']    = df['event_type'] == 'home_run'
    df['is_hit']   = df['event_type'].isin({'single', 'double', 'triple', 'home_run'})
    df['is_k']     = df['event_type'].isin({'strikeout', 'strikeout_double_play'})
    df['is_bb']    = df['event_type'].isin({'walk', 'intent_walk'})
    df['hard_hit'] = df['exit_velocity'].fillna(0) >= 95

    league_hr   = df['is_hr'].mean()
    league_hit  = df['is_hit'].mean()
    league_hard = df['hard_hit'].mean()
    league_k    = df['is_k'].mean()
    league_bb   = df['is_bb'].mean()

    records: list[dict] = []
    for venue_id, grp in df.groupby('venue_id'):
        n = len(grp)
        if n < 500:
            continue
        records.append({
            'venue_id': int(venue_id),
            'hr_factor':       _ratio(grp['is_hr'].mean(),    league_hr),
            'hit_factor':      _ratio(grp['is_hit'].mean(),   league_hit),
            'hard_hit_factor': _ratio(grp['hard_hit'].mean(), league_hard),
            'k_factor':        _ratio(grp['is_k'].mean(),     league_k),
            'bb_factor':       _ratio(grp['is_bb'].mean(),    league_bb),
            'sample_size':     int(n),
            'hit_locations':   _hit_location_profile(grp),
        })

    session = get_session()
    try:
        bulk_upsert(session, ParkFactor, records, pk_cols=['venue_id'])
        session.commit()
        print(f'  park factors: {len(records)} rows')
    finally:
        session.close()


def _ratio(a, b):
    return round(float(a / b), 4) if b else 1.0


def _hit_location_profile(grp: pd.DataFrame) -> dict:
    bip = grp[grp['hit_coord_x'].notna() & grp['hit_coord_y'].notna()]
    if len(bip) < 50:
        return {}

    def zone(row):
        x, y = row['hit_coord_x'], row['hit_coord_y']
        if x <= 100: h = 'pull'
        elif x <= 150: h = 'center'
        else: h = 'oppo'
        if y >= 160:   d = 'infield'
        elif y >= 130: d = 'shallow'
        else:          d = 'deep'
        return f'{h}_{d}'

    bip = bip.copy()
    bip['zone'] = bip.apply(zone, axis=1)
    total = len(bip)
    return {
        zone_key: round(cnt / total, 4)
        for zone_key, cnt in bip['zone'].value_counts().items()
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', nargs='+', type=int)
    args = parser.parse_args()
    run(seasons=args.seasons)
