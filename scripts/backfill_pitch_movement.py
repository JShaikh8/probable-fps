"""
Backfill pitch-movement fields (pfx_x, pfx_z, x0, z0, extension, plate_time)
onto existing `pitches` rows.

The main ingest_runner.py uses on_conflict_do_nothing, so a --force re-ingest
won't overwrite existing rows with the new movement columns. This script fixes
that retroactively: for every completed game in games_log, it re-fetches the
feed and UPDATEs each pitch row with its movement values.

Resume-safe — idempotent update, skips games where all movement fields are
already populated.

Run:
    python -m scripts.backfill_pitch_movement                     # all seasons
    python -m scripts.backfill_pitch_movement --seasons 2024 2025 # subset
    python -m scripts.backfill_pitch_movement --limit 1000        # first 1k
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from sqlalchemy import text
from tqdm import tqdm

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..'))

from config import get_engine
from ingest.fetch_game import fetch_game_feed, parse_game


RATE_LIMIT_S = 0.1
BATCH_UPDATE_SIZE = 500


def backfill(seasons: list[int] | None = None, limit: int | None = None):
    engine = get_engine()

    where_clauses = ["gl.status = 'done'"]
    params: dict = {}
    if seasons:
        where_clauses.append("g.season = ANY(:seasons)")
        params['seasons'] = seasons

    # Only process games where at least one pitch is missing movement data —
    # saves time on already-backfilled games.
    where_clauses.append("""
        EXISTS (
          SELECT 1 FROM pitches p
          WHERE p.game_pk = g.game_pk AND p.pfx_x IS NULL
          LIMIT 1
        )
    """)

    where_sql = ' AND '.join(where_clauses)
    limit_sql = f' LIMIT {int(limit)}' if limit else ''

    with engine.connect() as c:
        rows = c.execute(text(f"""
            SELECT g.game_pk, g.game_date, g.season
            FROM games g
            JOIN games_log gl ON gl.game_pk = g.game_pk
            WHERE {where_sql}
            ORDER BY g.game_date ASC
            {limit_sql}
        """), params).fetchall()

    total = len(rows)
    print(f'Games needing movement backfill: {total:,}')
    if not total:
        print('Nothing to do.')
        return

    done = failed = 0
    with engine.begin() as conn:
        pass   # noop — hint Postgres to warm

    for game_pk, game_date, season in tqdm(rows, desc='backfilling'):
        try:
            feed = fetch_game_feed(int(game_pk))
            if feed is None:
                failed += 1
                continue
            _, _, pitches, _ = parse_game(feed, {
                'game_pk': int(game_pk),
                'game_date': game_date,
                'season': season,
            })
            if not pitches:
                continue

            # Bulk update — only touch movement columns to minimize row churn
            updates = [
                {
                    'game_pk':    int(p['game_pk']),
                    'at_bat_index': int(p['at_bat_index']),
                    'pitch_index':  int(p['pitch_index']),
                    'pfx_x':      p.get('pfx_x'),
                    'pfx_z':      p.get('pfx_z'),
                    'x0':         p.get('x0'),
                    'z0':         p.get('z0'),
                    'extension':  p.get('extension'),
                    'plate_time': p.get('plate_time'),
                }
                for p in pitches
            ]

            with engine.begin() as conn:
                for i in range(0, len(updates), BATCH_UPDATE_SIZE):
                    chunk = updates[i:i + BATCH_UPDATE_SIZE]
                    conn.execute(text("""
                        UPDATE pitches SET
                            pfx_x = :pfx_x,
                            pfx_z = :pfx_z,
                            x0 = :x0,
                            z0 = :z0,
                            extension = :extension,
                            plate_time = :plate_time
                        WHERE game_pk = :game_pk
                          AND at_bat_index = :at_bat_index
                          AND pitch_index = :pitch_index
                    """), chunk)
            done += 1
        except Exception as exc:
            failed += 1
            print(f'  !! {game_pk}: {exc}')
        time.sleep(RATE_LIMIT_S)

    print(f'\nDone. {done} games updated, {failed} failed.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seasons', nargs='+', type=int)
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()
    backfill(seasons=args.seasons, limit=args.limit)
