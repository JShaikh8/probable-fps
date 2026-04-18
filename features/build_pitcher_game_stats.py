"""
Build per-start stats for pitchers from mlb_at_bats.
Computes IP, K, BB, H, HR per start and season-level aggregates including FIP.

Output collection: mlb_pitcher_season_stats
"""
from __future__ import annotations
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, '..')
from config import get_db
from pymongo import UpdateOne

STARTER_MIN_OUTS = 9    # >= 3 IP to count as a start
FIP_CONSTANT     = 3.10  # league-average FIP constant

HIT_EVENTS = {'single', 'double', 'triple', 'home_run'}


def run(seasons: list[int] | None = None):
    db = get_db()
    season_filter = {'season': {'$in': seasons}} if seasons else {}

    print('Loading at-bats for pitcher stats...')
    cur = db.mlb_at_bats.find(season_filter, {
        '_id': 0,
        'pitcherId': 1, 'pitcherName': 1, 'pitcherHand': 1,
        'gamePk': 1, 'season': 1,
        'eventType': 1, 'isOut': 1,
    })
    df = pd.DataFrame(list(cur))
    if df.empty:
        print('No data.')
        return

    print(f'  {len(df):,} at-bats loaded')

    df['isK']    = df['eventType'].isin({'strikeout', 'strikeout_double_play'})
    df['isBB']   = df['eventType'].isin({'walk', 'intent_walk'})
    df['isH']    = df['eventType'].isin(HIT_EVENTS)
    df['isHR']   = df['eventType'] == 'home_run'
    df['isOut_'] = df['isOut'].fillna(False)

    # Per-game stats
    game_stats = df.groupby(
        ['pitcherId', 'pitcherName', 'pitcherHand', 'gamePk', 'season']
    ).agg(
        bf    = ('pitcherId', 'count'),
        k     = ('isK',    'sum'),
        bb    = ('isBB',   'sum'),
        h     = ('isH',    'sum'),
        hr    = ('isHR',   'sum'),
        outs  = ('isOut_', 'sum'),
    ).reset_index()

    # Only keep starts (>= 3 innings)
    starts = game_stats[game_stats['outs'] >= STARTER_MIN_OUTS].copy()
    starts['ip'] = starts['outs'] / 3.0

    # FIP per start: (13·HR + 3·BB - 2·K) / IP + constant
    starts['fip'] = (
        (13 * starts['hr'] + 3 * starts['bb'] - 2 * starts['k'])
        / starts['ip'].replace(0, np.nan)
        + FIP_CONSTANT
    )

    print('Aggregating season stats...')
    records = []
    for (pitcher_id, season), grp in starts.groupby(['pitcherId', 'season']):
        n = len(grp)
        if n < 2:
            continue

        total_ip  = grp['ip'].sum()
        total_bf  = grp['bf'].sum()
        total_k   = grp['k'].sum()
        total_bb  = grp['bb'].sum()
        total_h   = grp['h'].sum()
        total_hr  = grp['hr'].sum()
        fip_val   = float(grp['fip'].mean()) if grp['fip'].notna().any() else None

        records.append({
            'pitcherId':    int(pitcher_id),
            'pitcherName':  grp['pitcherName'].iloc[0],
            'pitcherHand':  grp['pitcherHand'].iloc[0],
            'season':       int(season),
            'gamesStarted': int(n),
            'avgIP':        round(float(grp['ip'].mean()), 2),
            'avgK':         round(float(grp['k'].mean()), 2),
            'avgBB':        round(float(grp['bb'].mean()), 2),
            'avgH':         round(float(grp['h'].mean()), 2),
            'avgHR':        round(float(grp['hr'].mean()), 3),
            'kPct':         round(float(total_k / total_bf), 4) if total_bf > 0 else 0,
            'bbPct':        round(float(total_bb / total_bf), 4) if total_bf > 0 else 0,
            'hrPer9':       round(float(total_hr / total_ip * 9), 2) if total_ip > 0 else 0,
            'whip':         round(float((total_h + total_bb) / total_ip), 3) if total_ip > 0 else 0,
            'fip':          round(fip_val, 2) if fip_val is not None else None,
        })

    ops = [
        UpdateOne(
            {'pitcherId': r['pitcherId'], 'season': r['season']},
            {'$set': r}, upsert=True,
        )
        for r in records
    ]
    if ops:
        db.mlb_pitcher_season_stats.create_index(
            [('pitcherId', 1), ('season', 1)], unique=True
        )
        result = db.mlb_pitcher_season_stats.bulk_write(ops, ordered=False)
        print(f'  pitcher season stats: {result.upserted_count} inserted, {result.modified_count} updated')
    print('Done.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', nargs='+', type=int)
    args = parser.parse_args()
    run(seasons=args.seasons)
