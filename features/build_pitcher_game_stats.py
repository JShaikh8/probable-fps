"""
Per-start pitcher stats aggregated to season level.

Computes: avg_ip, avg_k, avg_bb, avg_h, avg_hr, fip, games_started
          hr9_vs_l, hr9_vs_r, k_pct_vs_l, k_pct_vs_r
          fb_pct_allowed, barrel_pct_allowed

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

FB_TRAJ = {'fly_ball', 'popup', 'pop_up'}
BARREL_MIN_EV = 95.0
BARREL_LA_LO  = 25.0
BARREL_LA_HI  = 35.0


def run(seasons: list[int] | None = None):
    engine = get_engine()
    where = f"WHERE season IN ({','.join(str(s) for s in seasons)})" if seasons else ''

    print('Loading at-bats for pitcher stats…')
    df = pd.read_sql_query(
        f"""
        SELECT pitcher_id, game_pk, season, event_type,
               hitter_side, exit_velocity, launch_angle, trajectory
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

    # Per-game aggregates per pitcher (used for avg IP and FIP)
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

    # ── Phase-1: per-season handedness + batted-ball-allowed rollups ──
    # Count PAs (as rows in at_bats, inclusive of all outcomes) vs each hand
    hand_pa = (df.groupby(['pitcher_id', 'season', 'hitter_side'])
                 .agg(pa=('pitcher_id', 'count'),
                      k=('is_k', 'sum'),
                      hr=('is_hr', 'sum'))
                 .reset_index())
    # Pivot to one row per (pitcher, season)
    pa_l = hand_pa[hand_pa['hitter_side'] == 'L'].rename(columns={'pa': 'pa_l', 'k': 'k_l', 'hr': 'hr_l'})[
        ['pitcher_id', 'season', 'pa_l', 'k_l', 'hr_l']
    ]
    pa_r = hand_pa[hand_pa['hitter_side'] == 'R'].rename(columns={'pa': 'pa_r', 'k': 'k_r', 'hr': 'hr_r'})[
        ['pitcher_id', 'season', 'pa_r', 'k_r', 'hr_r']
    ]

    # Batted-ball allowed — FB% and barrel% on BIP with EV+LA or trajectory
    bip = df[df['exit_velocity'].notna() & df['launch_angle'].notna()].copy()
    bip['is_barrel'] = (
        (bip['exit_velocity'] >= BARREL_MIN_EV) &
        (bip['launch_angle'].between(BARREL_LA_LO, BARREL_LA_HI))
    )
    barrel_agg = (bip.groupby(['pitcher_id', 'season'])
                     .agg(n_bip=('pitcher_id', 'count'),
                          n_barrel=('is_barrel', 'sum'))
                     .reset_index())

    traj_nonnull = df[df['trajectory'].notna()].copy()
    traj_nonnull['is_fb'] = traj_nonnull['trajectory'].isin(FB_TRAJ)
    fb_agg = (traj_nonnull.groupby(['pitcher_id', 'season'])
                          .agg(n_traj=('pitcher_id', 'count'),
                               n_fb=('is_fb', 'sum'))
                          .reset_index())

    records: list[dict] = []
    for (pid, season), grp in starts.groupby(['pitcher_id', 'season']):
        if len(grp) < 2:
            continue
        rec = {
            'pitcher_id': int(pid),
            'season': int(season),
            'avg_ip':   round(float(grp['ip'].mean()), 2),
            'avg_k':    round(float(grp['k'].mean()), 2),
            'avg_bb':   round(float(grp['bb'].mean()), 2),
            'avg_h':    round(float(grp['h'].mean()), 2),
            'avg_hr':   round(float(grp['hr'].mean()), 3),
            'fip':      round(float(grp['fip'].mean()), 2) if grp['fip'].notna().any() else 0.0,
            'games_started': int(len(grp)),
        }

        total_ip = float(grp['ip'].sum()) or 0.0

        def _split(hand_df, ip_frac_hint):
            """HR/9 and K% vs a handedness, scaled to PAs vs that hand."""
            sub = hand_df[(hand_df['pitcher_id'] == pid) & (hand_df['season'] == season)]
            if sub.empty or total_ip <= 0:
                return None, None
            pa = float(sub[f'pa_{ip_frac_hint}'].sum())
            k = float(sub[f'k_{ip_frac_hint}'].sum())
            hr = float(sub[f'hr_{ip_frac_hint}'].sum())
            if pa < 30:
                return None, None
            # Estimate IP vs this hand as (hand PAs / total PAs) * total IP
            all_pa = float(hand_pa[(hand_pa['pitcher_id'] == pid) & (hand_pa['season'] == season)]['pa'].sum()) or 1.0
            ip_hand = total_ip * (pa / all_pa)
            hr9 = (hr / ip_hand) * 9.0 if ip_hand > 0 else None
            kpct = k / pa if pa > 0 else None
            return (round(hr9, 3) if hr9 is not None else None,
                    round(kpct, 4) if kpct is not None else None)

        rec['hr9_vs_l'], rec['k_pct_vs_l'] = _split(pa_l, 'l')
        rec['hr9_vs_r'], rec['k_pct_vs_r'] = _split(pa_r, 'r')

        brow = barrel_agg[(barrel_agg['pitcher_id'] == pid) & (barrel_agg['season'] == season)]
        if not brow.empty and int(brow['n_bip'].iloc[0]) >= 30:
            rec['barrel_pct_allowed'] = round(float(brow['n_barrel'].iloc[0] / brow['n_bip'].iloc[0]), 4)
        else:
            rec['barrel_pct_allowed'] = None

        frow = fb_agg[(fb_agg['pitcher_id'] == pid) & (fb_agg['season'] == season)]
        if not frow.empty and int(frow['n_traj'].iloc[0]) >= 30:
            rec['fb_pct_allowed'] = round(float(frow['n_fb'].iloc[0] / frow['n_traj'].iloc[0]), 4)
        else:
            rec['fb_pct_allowed'] = None

        records.append(rec)

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
