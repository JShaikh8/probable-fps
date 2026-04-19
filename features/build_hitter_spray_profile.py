"""
Per-hitter spray tendency profiles from hit coordinates.
Writes: hitter_spray_profiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import get_engine, get_session
from db.models import HitterSprayProfile
from db.io import bulk_upsert


# Horizontal zones (MLB coord space, ~0-250, home plate near x=125)
PULL_X_RHH = 110
OPPO_X_RHH = 140
PULL_X_LHH = 140
OPPO_X_LHH = 110

INFIELD_Y = 160
SHALLOW_Y = 130

SEASON_WEIGHTS = {2026: 1.0, 2025: 0.90, 2024: 0.75, 2023: 0.60, 2022: 0.45, 2021: 0.30, 2020: 0.20}
MIN_BIP = 30

BIP_EVENTS = {'single', 'double', 'triple', 'home_run',
              'field_out', 'grounded_into_double_play', 'force_out',
              'sac_fly', 'fielders_choice', 'fielders_choice_out'}


def _classify(row: pd.Series) -> tuple[str, str]:
    x, y = row['hit_coord_x'], row['hit_coord_y']
    hand = row.get('hitter_side') or 'R'
    if hand == 'L':
        h = 'pull' if x > PULL_X_LHH else ('oppo' if x < OPPO_X_LHH else 'center')
    else:
        h = 'pull' if x < PULL_X_RHH else ('oppo' if x > OPPO_X_RHH else 'center')
    if y >= INFIELD_Y:   d = 'infield'
    elif y >= SHALLOW_Y: d = 'shallow'
    else:                d = 'deep'
    return h, d


def run():
    engine = get_engine()

    print('Loading at-bats for spray profiles…')
    df = pd.read_sql_query(
        """
        SELECT hitter_id, hitter_side, hit_coord_x, hit_coord_y,
               event_type, exit_velocity, launch_angle, season
        FROM at_bats
        WHERE hit_coord_x IS NOT NULL AND hit_coord_y IS NOT NULL
        """,
        engine,
    )
    if df.empty:
        print('No at-bats with hit coords. Exiting.')
        return
    print(f'  {len(df):,} at-bats across {df["hitter_id"].nunique()} hitters')

    df['weight'] = df['season'].map(SEASON_WEIGHTS).fillna(0.25)
    bip = df[df['event_type'].isin(BIP_EVENTS)].copy()
    if bip.empty:
        print('No BIP.')
        return

    classified = bip.apply(_classify, axis=1)
    bip[['hzone', 'dzone']] = pd.DataFrame(classified.tolist(), index=bip.index)

    records: list[dict] = []
    for hitter_id in bip['hitter_id'].unique():
        hdf = bip[bip['hitter_id'] == hitter_id]
        if len(hdf) < MIN_BIP:
            continue

        total_w = hdf['weight'].sum()
        if not total_w:
            continue

        pull_w   = hdf.loc[hdf['hzone'] == 'pull',   'weight'].sum()
        center_w = hdf.loc[hdf['hzone'] == 'center', 'weight'].sum()
        oppo_w   = hdf.loc[hdf['hzone'] == 'oppo',   'weight'].sum()
        deep_w   = hdf.loc[hdf['dzone'] == 'deep',   'weight'].sum()
        shallow_w = hdf.loc[hdf['dzone'] == 'shallow', 'weight'].sum()
        infield_w = hdf.loc[hdf['dzone'] == 'infield', 'weight'].sum()

        hr_rows = hdf[hdf['event_type'] == 'home_run']
        hr_pull_pct = 0.0
        if len(hr_rows) >= 3:
            hr_pull_w = hr_rows.loc[hr_rows['hzone'] == 'pull', 'weight'].sum()
            hr_total_w = hr_rows['weight'].sum()
            hr_pull_pct = round(float(hr_pull_w / hr_total_w), 4) if hr_total_w else 0.0

        ev_df = hdf[hdf['exit_velocity'].notna()]
        la_df = hdf[hdf['launch_angle'].notna()]
        avg_ev = float(np.average(ev_df['exit_velocity'], weights=ev_df['weight'])) if len(ev_df) >= 10 else None
        avg_la = float(np.average(la_df['launch_angle'],  weights=la_df['weight'])) if len(la_df) >= 10 else None

        records.append({
            'hitter_id':    int(hitter_id),
            'pull_pct':     round(float(pull_w / total_w), 4),
            'center_pct':   round(float(center_w / total_w), 4),
            'oppo_pct':     round(float(oppo_w / total_w), 4),
            'deep_pct':     round(float(deep_w / total_w), 4),
            'shallow_pct':  round(float(shallow_w / total_w), 4),
            'infield_pct':  round(float(infield_w / total_w), 4),
            'hr_pull_pct':  hr_pull_pct,
            'avg_exit_velo':    round(avg_ev, 2) if avg_ev is not None else None,
            'avg_launch_angle': round(avg_la, 2) if avg_la is not None else None,
        })

    session = get_session()
    try:
        bulk_upsert(session, HitterSprayProfile, records, pk_cols=['hitter_id'])
        session.commit()
        print(f'  spray profiles: {len(records)} rows')
    finally:
        session.close()


if __name__ == '__main__':
    run()
