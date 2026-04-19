"""
Aggregate hitter × pitch-family splits and pitcher arsenal profiles.

Reads from at_bats + pitches, writes hitter_pitch_splits and pitcher_profiles.
"""
from __future__ import annotations

import pandas as pd

from config import get_engine, get_session
from db.models import HitterPitchSplit, PitcherProfile
from db.io import bulk_upsert


# Swing / whiff MLBAM result codes
SWING_CODES = {'S', 'W', 'F', 'T', 'L', 'M', 'O', 'Q', 'R', 'X'}
WHIFF_CODES = {'S', 'W', 'T', 'L', 'M', 'O', 'Q', 'R'}

HIT_EVENTS = {'single', 'double', 'triple', 'home_run'}


def run(seasons: list[int] | None = None):
    engine = get_engine()

    where_pitch = f"WHERE p.season IN ({','.join(str(s) for s in seasons)})" if seasons else ''
    where_ab    = f"WHERE season IN ({','.join(str(s) for s in seasons)})"   if seasons else ''

    print('Loading pitches…')
    pitches = pd.read_sql_query(
        f"""
        SELECT p.hitter_id, p.pitcher_id, p.pitch_type, p.pitch_family,
               p.pitch_result, p.start_speed, p.spin_rate, p.px, p.pz,
               p.balls, p.strikes, p.pitch_index, p.at_bat_index, p.game_pk,
               p.season
        FROM pitches p
        {where_pitch}
        """,
        engine,
    )

    print('Loading at-bats…')
    at_bats = pd.read_sql_query(
        f"""
        SELECT game_pk, at_bat_index, hitter_id, pitcher_id,
               event, event_type, exit_velocity, launch_angle,
               hardness, trajectory, rbi, season
        FROM at_bats
        {where_ab}
        """,
        engine,
    )

    if pitches.empty or at_bats.empty:
        print('No data found. Run ingest first.')
        return

    print(f'  {len(pitches):,} pitches, {len(at_bats):,} at-bats')

    # Drop pitches with no family classification
    pitches = pitches[pitches['pitch_family'].notna()].copy()
    pitches['is_swing'] = pitches['pitch_result'].isin(SWING_CODES)
    pitches['is_whiff'] = pitches['pitch_result'].isin(WHIFF_CODES)

    # Velocity / spin tiers (top third per pitch family × season)
    print('Computing velocity/spin tier thresholds…')
    velo_p67 = (pitches.groupby(['pitch_family', 'season'])['start_speed']
                .quantile(0.67).rename('velo_p67').reset_index())
    spin_p67 = (pitches.groupby(['pitch_family', 'season'])['spin_rate']
                .quantile(0.67).rename('spin_p67').reset_index())
    pitches = pitches.merge(velo_p67, on=['pitch_family', 'season'], how='left')
    pitches = pitches.merge(spin_p67, on=['pitch_family', 'season'], how='left')
    pitches['is_high_velo'] = pitches['start_speed'] >= pitches['velo_p67']
    pitches['is_high_spin'] = pitches['spin_rate'].notna() & (pitches['spin_rate'] >= pitches['spin_p67'])

    # Merge last pitch of each AB with at-bat result
    last_idx = pitches.groupby(['game_pk', 'at_bat_index'])['pitch_index'].idxmax()
    last_pitches = pitches.loc[last_idx].copy()
    ab_cols = ['game_pk', 'at_bat_index', 'event_type', 'exit_velocity', 'launch_angle', 'hardness', 'trajectory']
    merged = last_pitches.merge(
        at_bats[ab_cols],
        on=['game_pk', 'at_bat_index'], how='left',
    )

    session = get_session()
    try:
        print('Building hitter pitch splits…')
        _write_hitter_splits(session, pitches, merged)

        print('Building pitcher profiles…')
        _write_pitcher_profiles(session, pitches, merged)

        session.commit()
    finally:
        session.close()

    print('Done.')


def _write_hitter_splits(session, pitches: pd.DataFrame, merged: pd.DataFrame):
    records: list[dict] = []
    groups = merged.groupby(['hitter_id', 'pitch_family', 'season'])

    for (hitter_id, pitch_family, season), grp in groups:
        pa = len(grp)
        if pa < 5:
            continue

        et = grp['event_type'].fillna('')
        hits = int(et.isin(HIT_EVENTS).sum())
        ab = int((~et.isin({'walk', 'intent_walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt'})).sum())
        bb = int(et.isin({'walk', 'intent_walk'}).sum())
        k  = int(et.isin({'strikeout', 'strikeout_double_play'}).sum())
        hr = int((et == 'home_run').sum())
        dbl = int((et == 'double').sum())
        tri = int((et == 'triple').sum())
        hbp = int((et == 'hit_by_pitch').sum())

        avg = hits / ab if ab > 0 else 0.0
        obp = (hits + bb + hbp) / pa if pa > 0 else 0.0
        slg_num = int((et == 'single').sum()) + dbl * 2 + tri * 3 + hr * 4
        slg = slg_num / ab if ab > 0 else 0.0

        ev = grp['exit_velocity'].dropna()
        avg_ev = float(ev.mean()) if len(ev) else None
        avg_la = float(grp['launch_angle'].dropna().mean()) if grp['launch_angle'].notna().any() else None

        if len(ev):
            hard_hit = float((ev >= 95).sum() / len(ev))
        elif 'hardness' in grp.columns and grp['hardness'].notna().any():
            bip = grp['hardness'].notna()
            hard_hit = float((grp.loc[bip, 'hardness'] == 'hard').sum() / bip.sum())
        else:
            hard_hit = None

        ph_all = pitches[
            (pitches['hitter_id'] == hitter_id) &
            (pitches['pitch_family'] == pitch_family) &
            (pitches['season'] == season)
        ]
        total_p = len(ph_all)
        swings  = int(ph_all['is_swing'].sum())
        whiffs  = int(ph_all['is_whiff'].sum())
        swing_pct = float(swings / total_p) if total_p else None
        whiff_pct = float(whiffs / swings) if swings else None

        def _tier_whiff(sub: pd.DataFrame) -> float | None:
            s = int(sub['is_swing'].sum())
            w = int(sub['is_whiff'].sum())
            return round(float(w / s), 4) if s >= 5 else None

        high_velo_whiff = _tier_whiff(ph_all[ph_all['is_high_velo']])
        high_spin_whiff = _tier_whiff(ph_all[ph_all['is_high_spin']])

        records.append({
            'hitter_id': int(hitter_id),
            'pitch_family': pitch_family,
            'season': int(season),
            'pa': int(pa), 'ab': ab, 'hits': hits,
            'hr': hr, 'bb': bb, 'k': k,
            'avg': round(avg, 4), 'slg': round(slg, 4),
            'obp': round(obp, 4), 'ops': round(obp + slg, 4),
            'swing_pct': round(swing_pct, 4) if swing_pct is not None else None,
            'whiff_pct': round(whiff_pct, 4) if whiff_pct is not None else None,
            'avg_exit_velo': round(avg_ev, 1) if avg_ev is not None else None,
            'avg_launch_angle': round(avg_la, 1) if avg_la is not None else None,
            'hard_hit_pct': round(hard_hit, 4) if hard_hit is not None else None,
            'high_velo_whiff_pct': high_velo_whiff,
            'high_spin_whiff_pct': high_spin_whiff,
        })

    bulk_upsert(session, HitterPitchSplit, records,
                pk_cols=['hitter_id', 'pitch_family', 'season'])
    print(f'  hitter splits: {len(records)} rows')


def _write_pitcher_profiles(session, pitches: pd.DataFrame, merged: pd.DataFrame):
    records: list[dict] = []
    groups = pitches.groupby(['pitcher_id', 'season'])

    for (pitcher_id, season), grp in groups:
        total = len(grp)
        if total < 50:
            continue

        arsenal: dict = {}
        for pf, sub in grp.groupby('pitch_family'):
            n = len(sub)
            whiffs = int(sub['is_whiff'].sum())
            swings = int(sub['is_swing'].sum())
            arsenal[pf] = {
                'count':    int(n),
                'usagePct': round(n / total, 4),
                'avgSpeed': round(float(sub['start_speed'].dropna().mean()), 1) if sub['start_speed'].notna().any() else None,
                'avgSpin':  round(float(sub['spin_rate'].dropna().mean()), 0) if sub['spin_rate'].notna().any() else None,
                'whiffPct': round(float(whiffs / swings), 4) if swings else None,
            }

        ab_sub = merged[(merged['pitcher_id'] == pitcher_id) & (merged['season'] == season)]
        k_pct = None
        if len(ab_sub):
            k = int(ab_sub['event_type'].isin({'strikeout', 'strikeout_double_play'}).sum())
            k_pct = round(float(k / len(ab_sub)), 4)

        records.append({
            'pitcher_id': int(pitcher_id),
            'season': int(season),
            'arsenal': arsenal,
            'total_pitches': int(total),
            'k_pct': k_pct or 0.0,
            'primary_pitch': max(arsenal, key=lambda kk: arsenal[kk]['usagePct']) if arsenal else None,
        })

    bulk_upsert(session, PitcherProfile, records,
                pk_cols=['pitcher_id', 'season'])
    print(f'  pitcher profiles: {len(records)} rows')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', nargs='+', type=int)
    args = parser.parse_args()
    run(seasons=args.seasons)
