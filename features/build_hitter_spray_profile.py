"""
Build per-hitter spray tendency profiles from historical at-bat coordinates.

Uses hitCoordX / hitCoordY from mlb_at_bats to classify each ball-in-play
into pull / center / oppo zones, accounting for batter handedness.

MLB Statcast coordinate system (bird's-eye view):
  - Home plate ≈ (125, 204)
  - Lower X  = left-field side   → pull for RHH, oppo for LHH
  - Higher X = right-field side  → pull for LHH, oppo for RHH
  - Lower Y  = deeper outfield
  - Higher Y = closer to home

Outputs per hitter (weighted 2025 > 2024 > ...):
  pullPct, centerPct, oppoPct       -- horizontal spray
  deepPct, shallowPct, infieldPct   -- depth zones
  avgExitVelo, avgLaunchAngle       -- overall contact quality
  pullHrPct                         -- % of HRs hit to pull side

Output collection: mlb_hitter_spray_profiles
"""
from __future__ import annotations
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, '..')
from config import get_db
from pymongo import UpdateOne

# Horizontal zone thresholds (from 0–250 MLB pixel space, home plate at x≈125)
PULL_X_RHH  = 110   # RHH pulls left  → x < PULL_X_RHH
OPPO_X_RHH  = 140   # RHH oppo right  → x > OPPO_X_RHH
PULL_X_LHH  = 140   # LHH pulls right → x > PULL_X_LHH (inverted)
OPPO_X_LHH  = 110   # LHH oppo left   → x < OPPO_X_LHH

# Depth thresholds
INFIELD_Y   = 160   # y >= 160  = infield
SHALLOW_Y   = 130   # 130 <= y < 160 = shallow outfield
# y < 130 = deep outfield

SEASON_WEIGHTS = {2025: 1.0, 2024: 0.85, 2023: 0.70, 2022: 0.55, 2021: 0.40}
MIN_BIP = 30


def _classify_spray(row: pd.Series) -> tuple[str, str]:
    """Return (horizontal_zone, depth_zone) for one at-bat."""
    x, y = row['hitCoordX'], row['hitCoordY']
    hand = row.get('hitterHand', 'R') or 'R'

    # Horizontal
    if hand == 'L':
        if x > PULL_X_LHH:
            h = 'pull'
        elif x < OPPO_X_LHH:
            h = 'oppo'
        else:
            h = 'center'
    else:  # R or switch (treat as R)
        if x < PULL_X_RHH:
            h = 'pull'
        elif x > OPPO_X_RHH:
            h = 'oppo'
        else:
            h = 'center'

    # Depth
    if y >= INFIELD_Y:
        d = 'infield'
    elif y >= SHALLOW_Y:
        d = 'shallow'
    else:
        d = 'deep'

    return h, d


def run():
    db = get_db()

    print('Loading at-bats for spray profiles...')
    cur = db.mlb_at_bats.find(
        {
            'hitCoordX': {'$exists': True, '$ne': None},
            'hitCoordY': {'$exists': True, '$ne': None},
        },
        {
            '_id': 0,
            'hitterId': 1, 'hitterName': 1, 'hitterHand': 1,
            'hitCoordX': 1, 'hitCoordY': 1,
            'eventType': 1, 'exitVelocity': 1, 'launchAngle': 1,
            'season': 1,
        }
    )
    df = pd.DataFrame(list(cur))

    if df.empty:
        print('No at-bat data with hit coordinates. Exiting.')
        return

    print(f'  {len(df):,} at-bats with hit coords across {df["hitterId"].nunique()} hitters')

    df['weight'] = df['season'].map(SEASON_WEIGHTS).fillna(0.25)

    # Keep only balls in play (exclude HR per BIP spray, but include HR in HR-pull calc)
    bip_types = {'single', 'double', 'triple', 'home_run',
                 'field_out', 'grounded_into_double_play', 'force_out',
                 'sac_fly', 'fielders_choice', 'fielders_choice_out'}
    bip = df[df['eventType'].isin(bip_types)].copy()

    if bip.empty:
        print('No balls in play found.')
        return

    # Classify each BIP
    classified = bip.apply(_classify_spray, axis=1)
    bip[['hzone', 'dzone']] = pd.DataFrame(classified.tolist(), index=bip.index)

    ops = []
    hitter_ids = bip['hitterId'].unique()
    print(f'  Building spray profiles for {len(hitter_ids)} hitters...')

    for hitter_id in hitter_ids:
        hdf = bip[bip['hitterId'] == hitter_id]
        if len(hdf) < MIN_BIP:
            continue

        w = hdf['weight'].values
        total_w = w.sum()
        if total_w == 0:
            continue

        # Horizontal spray (weighted)
        pull_w   = hdf.loc[hdf['hzone'] == 'pull',   'weight'].sum()
        center_w = hdf.loc[hdf['hzone'] == 'center', 'weight'].sum()
        oppo_w   = hdf.loc[hdf['hzone'] == 'oppo',   'weight'].sum()

        # Depth (weighted)
        deep_w    = hdf.loc[hdf['dzone'] == 'deep',    'weight'].sum()
        shallow_w = hdf.loc[hdf['dzone'] == 'shallow', 'weight'].sum()
        infield_w = hdf.loc[hdf['dzone'] == 'infield', 'weight'].sum()

        # HR pull tendency
        hr_rows = hdf[hdf['eventType'] == 'home_run']
        hr_pull_pct = None
        if len(hr_rows) >= 3:
            hr_pull_w  = hr_rows.loc[hr_rows['hzone'] == 'pull', 'weight'].sum()
            hr_total_w = hr_rows['weight'].sum()
            hr_pull_pct = round(float(hr_pull_w / hr_total_w), 4) if hr_total_w > 0 else None

        # Contact quality (weighted avg exit velo / launch angle)
        ev_df  = hdf[hdf['exitVelocity'].notna()]
        la_df  = hdf[hdf['launchAngle'].notna()]
        avg_ev = float(np.average(ev_df['exitVelocity'], weights=ev_df['weight'])) if len(ev_df) >= 10 else None
        avg_la = float(np.average(la_df['launchAngle'],  weights=la_df['weight'])) if len(la_df) >= 10 else None

        hitter_name = hdf['hitterName'].iloc[-1] if 'hitterName' in hdf.columns else ''
        hitter_hand = hdf['hitterHand'].dropna().iloc[-1] if hdf['hitterHand'].notna().any() else ''

        ops.append(UpdateOne(
            {'hitterId': int(hitter_id)},
            {'$set': {
                'hitterId':    int(hitter_id),
                'hitterName':  hitter_name,
                'hitterHand':  hitter_hand,
                'sampleBIP':   int(len(hdf)),
                'pullPct':     round(float(pull_w   / total_w), 4),
                'centerPct':   round(float(center_w / total_w), 4),
                'oppoPct':     round(float(oppo_w   / total_w), 4),
                'deepPct':     round(float(deep_w   / total_w), 4),
                'shallowPct':  round(float(shallow_w / total_w), 4),
                'infieldPct':  round(float(infield_w / total_w), 4),
                'hrPullPct':   hr_pull_pct,
                'avgExitVelo': round(avg_ev, 2) if avg_ev is not None else None,
                'avgLaunchAngle': round(avg_la, 2) if avg_la is not None else None,
            }},
            upsert=True,
        ))

    if ops:
        db.mlb_hitter_spray_profiles.create_index('hitterId', unique=True)
        result = db.mlb_hitter_spray_profiles.bulk_write(ops, ordered=False)
        print(f'  spray profiles: {result.upserted_count} inserted, {result.modified_count} updated')
    else:
        print('  No profiles met minimum BIP threshold.')

    print('Done.')


if __name__ == '__main__':
    run()
