from __future__ import annotations
"""
Aggregate hitter × pitch type splits from mlb_pitches + mlb_at_bats.

Output collection: mlb_hitter_pitch_splits
One document per (hitterId, pitchType, season) with:
  pa, ab, hits, doubles, triples, hr, bb, k, hbp,
  avg, obp, slg, ops,
  swingPct, whiffPct, chasePct,
  avgExitVelo, avgLaunchAngle, hardHitPct (exitVelo >= 95)

Also builds: mlb_pitcher_pitch_results
One doc per (pitcherId, pitchType, season) with usage%, avg velo, whiff%, k_per_pitch
"""
import sys
import pandas as pd
sys.path.insert(0, '..')
from config import get_db

# Pitch type groupings (collapse rare variants)
PITCH_FAMILY = {
    'FF': 'fastball', 'FA': 'fastball', 'FT': 'sinker', 'SI': 'sinker',
    'FC': 'cutter',
    'SL': 'slider', 'ST': 'slider',
    'CU': 'curveball', 'KC': 'curveball', 'CS': 'curveball',
    'CH': 'changeup', 'FS': 'splitter', 'FO': 'changeup',
    'KN': 'knuckleball',
    'EP': 'eephus',
}

# MLB pitch result codes → outcome category
SWING_CODES  = {'S', 'W', 'F', 'T', 'L', 'M', 'O', 'Q', 'R', 'X'}  # any swing
WHIFF_CODES  = {'S', 'W', 'T', 'L', 'M', 'O', 'Q', 'R'}             # swing and miss
BALL_CODES   = {'B', 'I', 'P'}
IN_ZONE      = None  # we'll approximate from px/pz

CONTACT_EVENTS = {'single', 'double', 'triple', 'home_run',
                  'field_out', 'grounded_into_double_play', 'force_out',
                  'sac_fly', 'sac_bunt', 'fielders_choice', 'double_play',
                  'triple_play', 'sac_fly_double_play'}
HIT_EVENTS     = {'single', 'double', 'triple', 'home_run'}


def run(seasons: list[int] | None = None):
    db = get_db()
    season_filter = {'season': {'$in': seasons}} if seasons else {}

    print('Loading pitches...')
    pitches_cur = db.mlb_pitches.find(season_filter, {
        '_id': 0, 'hitterId': 1, 'hitterName': 1, 'hitterHand': 1,
        'pitcherId': 1, 'pitcherHand': 1,
        'pitchType': 1, 'pitchResult': 1, 'startSpeed': 1, 'spinRate': 1,
        'px': 1, 'pz': 1, 'balls': 1, 'strikes': 1,
        'pitchIndex': 1, 'season': 1, 'atBatIndex': 1, 'gamePk': 1,
    })
    pitches = pd.DataFrame(list(pitches_cur))

    print('Loading at-bats...')
    abs_cur = db.mlb_at_bats.find(season_filter, {
        '_id': 0, 'gamePk': 1, 'atBatIndex': 1,
        'hitterId': 1, 'pitcherId': 1,
        'event': 1, 'eventType': 1,
        'exitVelocity': 1, 'launchAngle': 1,
        'hardness': 1, 'trajectory': 1,  # Statcast qualitative fallbacks
        'rbi': 1, 'isOut': 1, 'season': 1,
    })
    at_bats = pd.DataFrame(list(abs_cur))

    if pitches.empty or at_bats.empty:
        print('No data found. Run ingest first.')
        return

    print(f'  {len(pitches):,} pitches, {len(at_bats):,} at-bats')

    # Map pitch type to family
    pitches['pitchFamily'] = pitches['pitchType'].map(PITCH_FAMILY).fillna('other')
    pitches['isSwing']     = pitches['pitchResult'].isin(SWING_CODES)
    pitches['isWhiff']     = pitches['pitchResult'].isin(WHIFF_CODES)

    # ── Velocity / spin tier flags ──────────────────────────────────
    # Tag each pitch as high-velocity or high-spin relative to league average
    # for that pitch family × season (top third = "high")
    print('Computing velocity/spin tier thresholds...')
    velo_p67 = (
        pitches.groupby(['pitchFamily', 'season'])['startSpeed']
        .quantile(0.67).rename('velo_p67').reset_index()
    )
    spin_p67 = (
        pitches.groupby(['pitchFamily', 'season'])['spinRate']
        .quantile(0.67).rename('spin_p67').reset_index()
    )
    pitches = pitches.merge(velo_p67, on=['pitchFamily', 'season'], how='left')
    pitches = pitches.merge(spin_p67, on=['pitchFamily', 'season'], how='left')
    pitches['isHighVelo'] = pitches['startSpeed'] >= pitches['velo_p67']
    pitches['isHighSpin'] = (
        pitches['spinRate'].notna() & (pitches['spinRate'] >= pitches['spin_p67'])
    )

    # Merge last-pitch of AB with at-bat result (last pitch = max pitchIndex per AB)
    last_pitch_idx = pitches.groupby(['gamePk', 'atBatIndex'])['pitchIndex'].idxmax()
    last_pitches = pitches.loc[last_pitch_idx].copy()
    ab_cols = ['gamePk', 'atBatIndex', 'eventType', 'exitVelocity', 'launchAngle']
    for col in ['hardness', 'trajectory']:
        if col in at_bats.columns:
            ab_cols.append(col)
    merged = last_pitches.merge(
        at_bats[ab_cols],
        on=['gamePk', 'atBatIndex'], how='left',
    )

    # ── Hitter × pitch family × season splits ──────────────────────
    print('Building hitter pitch splits...')
    _build_hitter_splits(db, pitches, merged)

    # ── Pitcher × pitch family × season profile ────────────────────
    print('Building pitcher pitch profiles...')
    _build_pitcher_profiles(db, pitches, merged)

    print('Done.')


def _build_hitter_splits(db, pitches: pd.DataFrame, merged: pd.DataFrame):
    groups = merged.groupby(['hitterId', 'pitchFamily', 'season'])
    records = []

    for (hitter_id, pitch_family, season), grp in groups:
        pa    = len(grp)
        if pa < 5:
            continue

        et    = grp['eventType'].fillna('')
        hits  = et.isin(HIT_EVENTS).sum()
        ab    = (~et.isin({'walk', 'intent_walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt'})).sum()
        bb    = et.isin({'walk', 'intent_walk'}).sum()
        k     = et.isin({'strikeout', 'strikeout_double_play'}).sum()
        hr    = (et == 'home_run').sum()
        dbl   = (et == 'double').sum()
        tri   = (et == 'triple').sum()
        hbp   = (et == 'hit_by_pitch').sum()

        avg   = hits / ab if ab > 0 else 0
        obp   = (hits + bb + hbp) / pa if pa > 0 else 0
        slg_num = (
            (et == 'single').sum() +
            dbl * 2 + tri * 3 + hr * 4
        )
        slg = slg_num / ab if ab > 0 else 0

        ev     = grp['exitVelocity'].dropna()
        avg_ev = float(ev.mean()) if len(ev) > 0 else None
        avg_la = float(grp['launchAngle'].dropna().mean()) if grp['launchAngle'].notna().any() else None

        # hardHitPct: prefer exitVelocity >= 95; fall back to hardness == 'hard'
        if len(ev) > 0:
            hard_hit = float((ev >= 95).sum() / len(ev))
        elif 'hardness' in grp.columns and grp['hardness'].notna().any():
            bip = grp['hardness'].notna()
            hard_hit = float((grp.loc[bip, 'hardness'] == 'hard').sum() / bip.sum())
        else:
            hard_hit = None

        # Swing/whiff from all pitches in this family for this hitter/season
        ph_all = pitches[
            (pitches['hitterId'] == hitter_id) &
            (pitches['pitchFamily'] == pitch_family) &
            (pitches['season'] == season)
        ]
        total_p = len(ph_all)
        swing_pct = float(ph_all['isSwing'].sum() / total_p) if total_p > 0 else None
        whiff_pct = float(ph_all['isWhiff'].sum() / ph_all['isSwing'].sum()) if ph_all['isSwing'].sum() > 0 else None
        avg_speed = float(ph_all['startSpeed'].dropna().mean()) if ph_all['startSpeed'].notna().any() else None

        # High-velocity tier whiff rate (top-33% speed for this pitch family/season)
        def _tier_whiff(sub: pd.DataFrame) -> float | None:
            swings = int(sub['isSwing'].sum())
            whiffs = int(sub['isWhiff'].sum())
            return round(float(whiffs / swings), 4) if swings >= 5 else None

        high_velo_whiff = _tier_whiff(ph_all[ph_all['isHighVelo']]) if 'isHighVelo' in ph_all.columns else None
        high_spin_whiff = _tier_whiff(ph_all[ph_all['isHighSpin']]) if 'isHighSpin' in ph_all.columns else None

        records.append({
            'hitterId':       int(hitter_id),
            'hitterName':     grp['hitterName'].iloc[0] if 'hitterName' in grp.columns else '',
            'pitchFamily':    pitch_family,
            'season':         int(season),
            'pa': int(pa), 'ab': int(ab), 'hits': int(hits),
            'doubles': int(dbl), 'triples': int(tri), 'hr': int(hr),
            'bb': int(bb), 'k': int(k), 'hbp': int(hbp),
            'avg': round(avg, 4), 'obp': round(obp, 4),
            'slg': round(slg, 4), 'ops': round(obp + slg, 4),
            'swingPct': round(swing_pct, 4) if swing_pct is not None else None,
            'whiffPct': round(whiff_pct, 4) if whiff_pct is not None else None,
            'avgFacedSpeed': round(avg_speed, 1) if avg_speed is not None else None,
            'avgExitVelo': round(avg_ev, 1) if avg_ev is not None else None,
            'avgLaunchAngle': round(avg_la, 1) if avg_la is not None else None,
            'hardHitPct': round(hard_hit, 4) if hard_hit is not None else None,
            'highVeloWhiffPct': high_velo_whiff,
            'highSpinWhiffPct': high_spin_whiff,
        })

    from pymongo import UpdateOne
    ops = [
        UpdateOne(
            {'hitterId': r['hitterId'], 'pitchFamily': r['pitchFamily'], 'season': r['season']},
            {'$set': r}, upsert=True,
        )
        for r in records
    ]
    if ops:
        db.mlb_hitter_pitch_splits.create_index(
            [('hitterId', 1), ('pitchFamily', 1), ('season', 1)], unique=True
        )
        result = db.mlb_hitter_pitch_splits.bulk_write(ops, ordered=False)
        print(f'  hitter splits: {result.upserted_count} inserted, {result.modified_count} updated')


def _build_pitcher_profiles(db, pitches: pd.DataFrame, merged: pd.DataFrame):
    """One doc per pitcher/season: arsenal mix, avg velo per pitch, whiff%, command."""
    groups = pitches.groupby(['pitcherId', 'season'])
    records = []

    for (pitcher_id, season), grp in groups:
        total = len(grp)
        if total < 50:
            continue

        arsenal = {}
        for pf, sub in grp.groupby('pitchFamily'):
            n = len(sub)
            whiffs = sub['isWhiff'].sum()
            swings = sub['isSwing'].sum()
            arsenal[pf] = {
                'count':    int(n),
                'usagePct': round(n / total, 4),
                'avgSpeed': round(float(sub['startSpeed'].dropna().mean()), 1) if sub['startSpeed'].notna().any() else None,
                'avgSpin':  round(float(sub['spinRate'].dropna().mean()), 0) if sub['spinRate'].notna().any() else None,
                'whiffPct': round(float(whiffs / swings), 4) if swings > 0 else None,
            }

        # Overall strikeout rate from at-bat outcomes
        ab_sub = merged[(merged['pitcherId'] == pitcher_id) & (merged['season'] == season)]
        k_pct = None
        if len(ab_sub) > 0:
            k = ab_sub['eventType'].isin({'strikeout', 'strikeout_double_play'}).sum()
            k_pct = round(float(k / len(ab_sub)), 4)

        records.append({
            'pitcherId':   int(pitcher_id),
            'pitcherName': grp['pitcherName'].iloc[0] if 'pitcherName' in grp.columns else '',
            'pitcherHand': grp['pitcherHand'].iloc[0] if 'pitcherHand' in grp.columns else '',
            'season':      int(season),
            'totalPitches': int(total),
            'arsenal':     arsenal,
            'primaryPitch': max(arsenal, key=lambda k: arsenal[k]['usagePct']) if arsenal else None,
            'kPct':        k_pct,
        })

    from pymongo import UpdateOne
    ops = [
        UpdateOne(
            {'pitcherId': r['pitcherId'], 'season': r['season']},
            {'$set': r}, upsert=True,
        )
        for r in records
    ]
    if ops:
        db.mlb_pitcher_profiles.create_index([('pitcherId', 1), ('season', 1)], unique=True)
        result = db.mlb_pitcher_profiles.bulk_write(ops, ordered=False)
        print(f'  pitcher profiles: {result.upserted_count} inserted, {result.modified_count} updated')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', nargs='+', type=int)
    args = parser.parse_args()
    run(seasons=args.seasons)
