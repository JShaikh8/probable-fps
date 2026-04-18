"""
Bulk ingestion runner — resume-safe.
Tracks which gamePks are already ingested in mlb_games_log.
Run: python ingest_runner.py --seasons 2020 2021 2022 2023 2024
"""
from __future__ import annotations
import argparse
import sys
import time
from datetime import datetime

from tqdm import tqdm
from pymongo import UpdateOne

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from config import get_db, DEFAULT_SEASONS
from fetch_season import fetch_season_games
from fetch_game import fetch_game_feed, parse_game

BATCH_SIZE   = 200   # bulk_write batch size
RATE_LIMIT_S = 0.15  # seconds between game fetches (~6-7 req/s, well under MLB limits)


def ensure_indexes(db):
    db.mlb_pitches.create_index([('gamePk', 1), ('atBatIndex', 1), ('pitchIndex', 1)], unique=True)
    db.mlb_at_bats.create_index([('gamePk', 1), ('atBatIndex', 1)], unique=True)
    db.mlb_games_log.create_index('gamePk', unique=True)
    db.mlb_pitches.create_index([('hitterId', 1), ('pitchType', 1)])
    db.mlb_pitches.create_index([('pitcherId', 1), ('pitchType', 1)])
    db.mlb_pitches.create_index('season')
    db.mlb_at_bats.create_index([('hitterId', 1), ('season', 1)])
    db.mlb_at_bats.create_index([('pitcherId', 1), ('season', 1)])
    db.mlb_at_bats.create_index('venueId')


def already_ingested(db, game_pk: int) -> bool:
    return db.mlb_games_log.find_one({'gamePk': game_pk, 'status': 'done'}) is not None


def mark_ingested(db, game_pk: int, pitch_count: int, ab_count: int):
    db.mlb_games_log.update_one(
        {'gamePk': game_pk},
        {'$set': {'status': 'done', 'pitchCount': pitch_count, 'abCount': ab_count, 'ingestedAt': datetime.utcnow()}},
        upsert=True,
    )


def mark_failed(db, game_pk: int, error: str):
    db.mlb_games_log.update_one(
        {'gamePk': game_pk},
        {'$set': {'status': 'failed', 'error': error, 'failedAt': datetime.utcnow()}},
        upsert=True,
    )


def upsert_docs(collection, docs: list[dict], key_fields: list[str]):
    if not docs:
        return
    ops = []
    for doc in docs:
        key = {f: doc[f] for f in key_fields}
        ops.append(UpdateOne(key, {'$set': doc}, upsert=True))
        if len(ops) >= BATCH_SIZE:
            collection.bulk_write(ops, ordered=False)
            ops = []
    if ops:
        collection.bulk_write(ops, ordered=False)


def ingest_season(db, season: int, force: bool = False):
    print(f'\n── Season {season} ──────────────────────────────────')
    games = fetch_season_games(season)
    print(f'  {len(games)} completed games found')

    skipped = errors = ingested = 0

    for game_meta in tqdm(games, desc=f'{season}', unit='game'):
        gpk = game_meta['gamePk']

        if not force and already_ingested(db, gpk):
            skipped += 1
            continue

        try:
            feed = fetch_game_feed(gpk)
            if feed is None:
                mark_failed(db, gpk, 'feed/live returned 404')
                errors += 1
                continue

            pitches, at_bats = parse_game(feed, game_meta)

            upsert_docs(db.mlb_pitches,  pitches,  ['gamePk', 'atBatIndex', 'pitchIndex'])
            upsert_docs(db.mlb_at_bats,  at_bats,  ['gamePk', 'atBatIndex'])
            mark_ingested(db, gpk, len(pitches), len(at_bats))
            ingested += 1

        except Exception as exc:
            mark_failed(db, gpk, str(exc))
            errors += 1

        time.sleep(RATE_LIMIT_S)

    print(f'  ✓ ingested={ingested}  skipped={skipped}  errors={errors}')


def main():
    parser = argparse.ArgumentParser(description='Ingest MLB game feeds into MongoDB')
    parser.add_argument('--seasons', nargs='+', type=int, default=DEFAULT_SEASONS)
    parser.add_argument('--force', action='store_true', help='Re-ingest already-done games')
    args = parser.parse_args()

    db = get_db()
    ensure_indexes(db)

    for season in sorted(args.seasons):
        ingest_season(db, season, force=args.force)

    print('\nDone.')


if __name__ == '__main__':
    main()
