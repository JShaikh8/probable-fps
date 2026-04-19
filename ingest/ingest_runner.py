"""
Bulk ingestion runner — resume-safe.

Tracks which gamePks are already ingested in `games_log` and skips them on re-runs.

Run:
    python -m ingest.ingest_runner --seasons 2024 2025
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..'))

from config import get_session, DEFAULT_SEASONS
from db.models import AtBat, Game, GameLog, Pitch, Player, Team, Venue
from ingest.fetch_season import fetch_season_games
from ingest.fetch_game import fetch_game_feed, parse_game


BATCH_SIZE   = 500    # rows per bulk insert
RATE_LIMIT_S = 0.15   # seconds between game fetches


def already_ingested(session, game_pk: int) -> bool:
    row = session.execute(
        select(GameLog.status).where(GameLog.game_pk == game_pk)
    ).scalar_one_or_none()
    return row == 'done'


def mark_log(session, game_pk: int, status: str, *, pitch_count: int = 0,
             ab_count: int = 0, error: str | None = None):
    stmt = pg_insert(GameLog).values(
        game_pk=game_pk,
        status=status,
        pitch_count=pitch_count,
        ab_count=ab_count,
        ingested_at=datetime.utcnow(),
        error=error,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[GameLog.game_pk],
        set_={
            'status': stmt.excluded.status,
            'pitch_count': stmt.excluded.pitch_count,
            'ab_count': stmt.excluded.ab_count,
            'ingested_at': stmt.excluded.ingested_at,
            'error': stmt.excluded.error,
        },
    )
    session.execute(stmt)


def _upsert_team(session, team_id: int | None, name: str | None):
    if not team_id:
        return
    stmt = pg_insert(Team).values(team_id=team_id, name=name or '')
    stmt = stmt.on_conflict_do_update(
        index_elements=[Team.team_id],
        set_={'name': stmt.excluded.name},
    )
    session.execute(stmt)


def _upsert_venue(session, venue_id: int | None, name: str | None):
    if not venue_id:
        return
    stmt = pg_insert(Venue).values(venue_id=venue_id, name=name or '')
    stmt = stmt.on_conflict_do_update(
        index_elements=[Venue.venue_id],
        set_={'name': stmt.excluded.name},
    )
    session.execute(stmt)


def upsert_game_meta(session, game_meta: dict):
    """Upsert a game row (plus its team/venue FKs) from the schedule API."""
    _upsert_team(session, game_meta.get('home_team_id'), game_meta.get('home_team_name'))
    _upsert_team(session, game_meta.get('away_team_id'), game_meta.get('away_team_name'))
    _upsert_venue(session, game_meta.get('venue_id'), game_meta.get('venue_name'))

    payload = {k: v for k, v in game_meta.items()
               if k in {'game_pk', 'game_date', 'season', 'home_team_id',
                        'away_team_id', 'venue_id', 'status', 'double_header',
                        'game_time_utc'}}
    stmt = pg_insert(Game).values(**payload)
    update_cols = {k: getattr(stmt.excluded, k) for k in payload if k != 'game_pk'}
    stmt = stmt.on_conflict_do_update(
        index_elements=[Game.game_pk],
        set_=update_cols,
    )
    session.execute(stmt)


def apply_game_update(session, game_update: dict):
    """Merge post-ingest fields (scores, weather) into the game row."""
    stmt = pg_insert(Game).values(
        game_pk=game_update['game_pk'],
        home_score=game_update.get('home_score'),
        away_score=game_update.get('away_score'),
        weather=game_update.get('weather'),
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[Game.game_pk],
        set_={
            'home_score': stmt.excluded.home_score,
            'away_score': stmt.excluded.away_score,
            'weather': stmt.excluded.weather,
        },
    )
    session.execute(stmt)


def bulk_insert_ignore(session, model, rows: list[dict], pk_cols: list[str]):
    """Insert rows; skip any whose natural key collides."""
    if not rows:
        return
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i:i + BATCH_SIZE]
        stmt = pg_insert(model.__table__).values(chunk).on_conflict_do_nothing(
            index_elements=pk_cols,
        )
        session.execute(stmt)


def upsert_players(session, players: list[dict]):
    """Upsert player metadata (name, side/hand) seen in a game."""
    if not players:
        return
    for p in players:
        stmt = pg_insert(Player).values(**p)
        update_cols = {k: getattr(stmt.excluded, k) for k in p if k != 'player_id'}
        stmt = stmt.on_conflict_do_update(
            index_elements=[Player.player_id],
            set_=update_cols,
        )
        session.execute(stmt)


def ingest_season(season: int, force: bool = False):
    print(f'\n── Season {season} ──────────────────────────────────')
    games = fetch_season_games(season)
    print(f'  {len(games)} completed games found')

    skipped = errors = ingested = 0
    session = get_session()

    try:
        # Seed the `games` table with metadata so downstream joins work even
        # if full ingestion fails for a particular gamePk.
        for g in games:
            upsert_game_meta(session, g)
        session.commit()

        for game_meta in tqdm(games, desc=f'{season}', unit='game'):
            gpk = game_meta['game_pk']

            if not force and already_ingested(session, gpk):
                skipped += 1
                continue

            try:
                feed = fetch_game_feed(gpk)
                if feed is None:
                    mark_log(session, gpk, 'failed', error='feed/live returned 404')
                    session.commit()
                    errors += 1
                    continue

                game_update, at_bats, pitches, players = parse_game(feed, game_meta)

                apply_game_update(session, game_update)
                upsert_players(session, players)
                bulk_insert_ignore(
                    session, AtBat, at_bats,
                    pk_cols=['game_pk', 'at_bat_index'],
                )
                bulk_insert_ignore(
                    session, Pitch, pitches,
                    pk_cols=['game_pk', 'at_bat_index', 'pitch_index'],
                )
                mark_log(session, gpk, 'done',
                         pitch_count=len(pitches), ab_count=len(at_bats))
                session.commit()
                ingested += 1

            except Exception as exc:
                session.rollback()
                try:
                    mark_log(session, gpk, 'failed', error=str(exc)[:500])
                    session.commit()
                except Exception:
                    session.rollback()
                errors += 1

            time.sleep(RATE_LIMIT_S)

    finally:
        session.close()

    print(f'  ✓ ingested={ingested}  skipped={skipped}  errors={errors}')


def main():
    parser = argparse.ArgumentParser(description='Ingest MLB game feeds into Postgres')
    parser.add_argument('--seasons', nargs='+', type=int, default=DEFAULT_SEASONS)
    parser.add_argument('--force', action='store_true',
                        help='Re-ingest games already marked done')
    args = parser.parse_args()

    for season in sorted(args.seasons):
        ingest_season(season, force=args.force)

    print('\nDone.')


if __name__ == '__main__':
    main()
