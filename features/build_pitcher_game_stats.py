"""
Per-start pitcher stats aggregated to season level (avg IP, K, BB, H, HR, FIP).
Writes: pitcher_season_stats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import get_engine, get_session
from db.models import PitcherSeasonStats
from db.io import bulk_upsert


STARTER_MIN_OUTS = 9       # >= 3 IP to count as a start
FIP_CONSTANT     = 3.10

OUT_EVENTS = {
    'strikeout', 'strikeout_double_play',
    'field_out', 'force_out', 'grounded_into_double_play',
    'double_play', 'triple_play', 'sac_fly', 'sac_bunt',
    'sac_fly_double_play', 'fielders_choice_out',
}


def run(seasons: list[int] | None = None):
    engine = get_engine()
    where = f"WHERE season IN ({','.join(str(s) for s in seasons)})" if seasons else ''

    print('Loading at-bats for pitcher stats…')
    df = pd.read_sql_query(
        f"""
        SELECT pitcher_id, game_pk, season, event_type
        FROM at_bats
        {where}
        """,
        engine,
    )
    if df.empty:
        print('No data.')
        return
    print(f'  {len(df):,} at-bats')

    df['is_k']    = df['event_type'].isin({'strikeout', 'strikeout_double_play'})
    df['is_bb']   = df['event_type'].isin({'walk', 'intent_walk'})
    df['is_h']    = df['event_type'].isin({'single', 'double', 'triple', 'home_run'})
    df['is_hr']   = df['event_type'] == 'home_run'
    df['is_out']  = df['event_type'].isin(OUT_EVENTS)

    # Per-game aggregates per pitcher
    gstats = df.groupby(['pitcher_id', 'game_pk', 'season']).agg(
        bf=('pitcher_id', 'count'),
        k=('is_k', 'sum'),
        bb=('is_bb', 'sum'),
        h=('is_h', 'sum'),
        hr=('is_hr', 'sum'),
        outs=('is_out', 'sum'),
    ).reset_index()

    starts = gstats[gstats['outs'] >= STARTER_MIN_OUTS].copy()
    if starts.empty:
        print('No qualifying starts.')
        return
    starts['ip'] = starts['outs'] / 3.0
    starts['fip'] = (
        (13 * starts['hr'] + 3 * starts['bb'] - 2 * starts['k'])
        / starts['ip'].replace(0, np.nan)
        + FIP_CONSTANT
    )

    records: list[dict] = []
    for (pid, season), grp in starts.groupby(['pitcher_id', 'season']):
        if len(grp) < 2:
            continue
        records.append({
            'pitcher_id': int(pid),
            'season': int(season),
            'avg_ip':   round(float(grp['ip'].mean()), 2),
            'avg_k':    round(float(grp['k'].mean()), 2),
            'avg_bb':   round(float(grp['bb'].mean()), 2),
            'avg_h':    round(float(grp['h'].mean()), 2),
            'avg_hr':   round(float(grp['hr'].mean()), 3),
            'fip':      round(float(grp['fip'].mean()), 2) if grp['fip'].notna().any() else 0.0,
            'games_started': int(len(grp)),
        })

    session = get_session()
    try:
        bulk_upsert(session, PitcherSeasonStats, records,
                    pk_cols=['pitcher_id', 'season'])
        session.commit()
        print(f'  pitcher season stats: {len(records)} rows')
    finally:
        session.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', nargs='+', type=int)
    args = parser.parse_args()
    run(seasons=args.seasons)
