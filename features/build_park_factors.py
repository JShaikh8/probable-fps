"""
Build park factors per venue from historical at-bat data.
Computes run factor and HR factor normalized to league average (1.00 = neutral).

Output collection: mlb_park_factors
"""
from __future__ import annotations
import sys
import pandas as pd
sys.path.insert(0, '..')
from config import get_db
from pymongo import UpdateOne


def run(seasons: list[int] | None = None):
    db = get_db()
    season_filter = {'season': {'$in': seasons}} if seasons else {}

    print('Loading at-bats for park factors...')
    cur = db.mlb_at_bats.find(season_filter, {
        '_id': 0, 'venueId': 1, 'venueName': 1,
        'eventType': 1, 'exitVelocity': 1, 'launchAngle': 1,
        'hitCoordX': 1, 'hitCoordY': 1, 'season': 1,
    })
    df = pd.DataFrame(list(cur))

    if df.empty:
        print('No data. Run ingest first.')
        return

    print(f'  {len(df):,} at-bats across {df["venueId"].nunique()} venues')

    df['isHR']     = df['eventType'] == 'home_run'
    df['isHit']    = df['eventType'].isin({'single', 'double', 'triple', 'home_run'})
    df['isDouble'] = df['eventType'] == 'double'
    df['isTriple'] = df['eventType'] == 'triple'
    df['isK']      = df['eventType'].isin({'strikeout', 'strikeout_double_play'})
    df['isBB']     = df['eventType'].isin({'walk', 'intent_walk'})
    df['hardHit']  = (df['exitVelocity'].fillna(0) >= 95)

    # League-wide rates
    league_hr_rate   = df['isHR'].mean()
    league_hit_rate  = df['isHit'].mean()
    league_hard_rate = df['hardHit'].mean()
    league_k_rate    = df['isK'].mean()
    league_bb_rate   = df['isBB'].mean()

    groups = df.groupby(['venueId', 'venueName'])
    records = []

    for (venue_id, venue_name), grp in groups:
        n = len(grp)
        if n < 500:
            continue

        hr_rate   = grp['isHR'].mean()
        hit_rate  = grp['isHit'].mean()
        hard_rate = grp['hardHit'].mean()
        k_rate    = grp['isK'].mean()
        bb_rate   = grp['isBB'].mean()

        # Hit location tendency: divide field into zones, count hit frequency
        hit_locations = _build_hit_location_profile(grp)

        records.append({
            'venueId':      int(venue_id),
            'venueName':    venue_name,
            'sampleSize':   int(n),
            'hrFactor':     round(hr_rate / league_hr_rate, 4) if league_hr_rate > 0 else 1.0,
            'hitFactor':    round(hit_rate / league_hit_rate, 4) if league_hit_rate > 0 else 1.0,
            'hardHitFactor': round(hard_rate / league_hard_rate, 4) if league_hard_rate > 0 else 1.0,
            'kFactor':      round(k_rate / league_k_rate, 4) if league_k_rate > 0 else 1.0,
            'bbFactor':     round(bb_rate / league_bb_rate, 4) if league_bb_rate > 0 else 1.0,
            'hitLocations': hit_locations,
        })

    ops = [
        UpdateOne({'venueId': r['venueId']}, {'$set': r}, upsert=True)
        for r in records
    ]
    if ops:
        db.mlb_park_factors.create_index('venueId', unique=True)
        result = db.mlb_park_factors.bulk_write(ops, ordered=False)
        print(f'  park factors: {result.upserted_count} inserted, {result.modified_count} updated')
    print('Done.')


def _build_hit_location_profile(grp: pd.DataFrame) -> dict:
    """
    Divide the field into 9 zones using normalized hit coordinates.
    Returns percentage of balls in play landing in each zone.

    MLB hit coords: (0,0) top-left, (250,250) = roughly center field area.
    We use a simplified 3×3 grid: pull/center/oppo × infield/shallow/deep
    """
    bip = grp[grp['hitCoordX'].notna() & grp['hitCoordY'].notna()]
    if len(bip) < 50:
        return {}

    # Rough field zones based on coordinate ranges
    def zone(row):
        x, y = row['hitCoordX'], row['hitCoordY']
        # Horizontal: pull (<= 100), center (101-150), oppo (>150)
        if x <= 100:
            h = 'pull'
        elif x <= 150:
            h = 'center'
        else:
            h = 'oppo'
        # Depth: infield (y >= 160), shallow (130-159), deep (< 130)
        if y >= 160:
            depth = 'infield'
        elif y >= 130:
            depth = 'shallow'
        else:
            depth = 'deep'
        return f'{h}_{depth}'

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
