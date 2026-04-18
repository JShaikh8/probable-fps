"""
Backfill hitData (exitVelocity, launchAngle, hitCoordX/Y, trajectory, hardness)
for all existing at_bats by re-fetching game feeds from the MLB API.

This script:
  1. Finds all distinct gamePks in mlb_at_bats
  2. Re-fetches each game's feed/live endpoint
  3. Extracts hitData from the play-ending pitch event (where it actually lives)
  4. Updates only the hitData fields in existing at_bat documents

Run: python3 backfill_hit_data.py [--seasons 2024 2025] [--limit N]
"""
from __future__ import annotations
import sys
import time
import argparse
from datetime import datetime
sys.path.insert(0, '..')
from config import get_db, MLB_API_BASE2
from fetch_game import fetch_game_feed
from pymongo import UpdateOne
import requests

BATCH_SIZE  = 200   # bulk write batch size
SLEEP_MS    = 80    # ms between API calls (avoid rate limiting)
MAX_RETRIES = 3


def _extract_hit_data_by_ab(feed: dict) -> dict[int, dict]:
    """
    Returns {atBatIndex: hit_data_dict} for all plays in the feed
    that have hitData on their play events.
    """
    live_data = feed.get('liveData', {})
    all_plays = live_data.get('plays', {}).get('allPlays', [])
    result = {}
    for play in all_plays:
        if not play.get('about', {}).get('isComplete', False):
            continue
        ab_index = play['about'].get('atBatIndex')
        for ev in play.get('playEvents', []):
            if ev.get('hitData'):
                hd = ev['hitData']
                coords = hd.get('coordinates', {})
                result[ab_index] = {
                    'exitVelocity':  hd.get('launchSpeed'),
                    'launchAngle':   hd.get('launchAngle'),
                    'totalDistance': hd.get('totalDistance'),
                    'trajectory':    hd.get('trajectory'),
                    'hardness':      hd.get('hardness'),
                    'hitLocation':   hd.get('location'),
                    'hitCoordX':     coords.get('coordX'),
                    'hitCoordY':     coords.get('coordY'),
                }
                break  # one hitData per at-bat
    return result


def run(seasons: list[int] | None = None, limit: int | None = None,
        skip_populated: bool = True):
    db = get_db()

    # Build the set of gamePks to process
    match = {}
    if seasons:
        match['season'] = {'$in': seasons}
    if skip_populated:
        # Only process games where hitData is missing (exitVelocity still null)
        match['exitVelocity'] = None

    print('Finding gamePks to backfill...')
    pipeline = [
        {'$match': match},
        {'$group': {'_id': '$gamePk', 'season': {'$first': '$season'}}},
        {'$sort': {'season': -1, '_id': 1}},
    ]
    if limit:
        pipeline.append({'$limit': limit})

    game_pks = [(doc['_id'], doc['season']) for doc in db.mlb_at_bats.aggregate(pipeline)]
    print(f'  {len(game_pks)} games to backfill')

    total_updated = 0
    total_skipped = 0
    total_errors  = 0

    for game_idx, (game_pk, season) in enumerate(game_pks):
        if game_idx % 50 == 0:
            print(f'  [{game_idx}/{len(game_pks)}] updated={total_updated} skipped={total_skipped} errors={total_errors}')

        # Fetch game feed with retry
        feed = None
        for attempt in range(MAX_RETRIES):
            try:
                feed = fetch_game_feed(game_pk)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f'    !! gamePk {game_pk}: {e}')
                    total_errors += 1
                time.sleep(1.0)

        if feed is None:
            total_skipped += 1
            time.sleep(SLEEP_MS / 1000)
            continue

        hit_map = _extract_hit_data_by_ab(feed)
        if not hit_map:
            total_skipped += 1
            time.sleep(SLEEP_MS / 1000)
            continue

        # Bulk update just the hitData fields for this game
        ops = []
        for ab_index, hd in hit_map.items():
            ops.append(UpdateOne(
                {'gamePk': game_pk, 'atBatIndex': ab_index},
                {'$set': hd},
            ))

        if ops:
            result = db.mlb_at_bats.bulk_write(ops, ordered=False)
            total_updated += result.modified_count

        time.sleep(SLEEP_MS / 1000)

    print(f'\nBackfill complete: {total_updated} at-bats updated, '
          f'{total_skipped} games skipped, {total_errors} errors')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', nargs='+', type=int,
                        help='Seasons to backfill (e.g. --seasons 2024 2025). Default: all.')
    parser.add_argument('--limit', type=int,
                        help='Max games to process (for testing).')
    parser.add_argument('--all', action='store_true',
                        help='Re-process all games, even already-populated ones.')
    args = parser.parse_args()
    run(
        seasons=args.seasons,
        limit=args.limit,
        skip_populated=not args.all,
    )
