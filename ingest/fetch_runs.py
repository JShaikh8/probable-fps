"""
Supplementary ingest: per-hitter runs scored + stolen bases per game.

MLB's /boxscore endpoint returns full player stats including `runs` and
`stolenBases`. We add these to a new `hitter_game_stats` table so DK/FD
targets can include them (each run = 2 DK pts, each SB = 5 DK pts).

Run:
    python -m ingest.fetch_runs
    python -m ingest.fetch_runs --seasons 2024 2025 2026
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import requests
from sqlalchemy import BigInteger, Column, DateTime, Integer, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..'))

from config import DEFAULT_SEASONS, get_engine, get_session
from db.models import Base


# ── Schema for per-hitter per-game counts (runs, SB) ──────────────
class HitterGameStats(Base):
    __tablename__ = 'hitter_game_stats'
    __table_args__ = {'extend_existing': True}
    hitter_id = Column(Integer, primary_key=True)
    game_pk = Column(BigInteger, primary_key=True)
    runs = Column(Integer, default=0)
    stolen_bases = Column(Integer, default=0)
    caught_stealing = Column(Integer, default=0)
    sac_flies = Column(Integer, default=0)
    ingested_at = Column(DateTime, default=datetime.utcnow)


BOXSCORE_URL = 'https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore'
RATE_LIMIT_S = 0.1


def fetch_box(game_pk: int) -> dict | None:
    try:
        resp = requests.get(BOXSCORE_URL.format(game_pk=game_pk), timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f'  !! {game_pk}: {exc}')
        return None


def parse_box(box: dict) -> list[dict]:
    """Extract per-hitter runs/SB from a boxscore response."""
    rows: list[dict] = []
    for side in ('home', 'away'):
        team = box.get('teams', {}).get(side, {}) or {}
        for pid, player in (team.get('players') or {}).items():
            batting = (player.get('stats') or {}).get('batting') or {}
            if not batting:
                continue
            person_id = (player.get('person') or {}).get('id')
            if not person_id:
                continue
            rows.append({
                'hitter_id':   int(person_id),
                'runs':        int(batting.get('runs', 0) or 0),
                'stolen_bases': int(batting.get('stolenBases', 0) or 0),
                'caught_stealing': int(batting.get('caughtStealing', 0) or 0),
                'sac_flies':   int(batting.get('sacFlies', 0) or 0),
            })
    return rows


def ingest_runs(seasons: list[int], force: bool = False):
    engine = get_engine()
    Base.metadata.create_all(engine, tables=[HitterGameStats.__table__])

    session = get_session()
    try:
        # Already-ingested games (so we can skip on re-runs)
        if force:
            existing = set()
        else:
            existing = {
                int(r[0]) for r in session.execute(
                    text('SELECT DISTINCT game_pk FROM hitter_game_stats')
                )
            }

        # Games to process: all completed games in the given seasons
        seasons_tuple = tuple(seasons)
        games = session.execute(
            text(f"""
                SELECT game_pk FROM games
                WHERE season IN ({','.join(str(s) for s in seasons_tuple)})
                  AND status = 'final'
                ORDER BY game_date DESC, game_pk
            """),
        ).all()
        targets = [g[0] for g in games if g[0] not in existing]
        print(f'Boxscore fetch — {len(targets):,} of {len(games):,} games need data')

        for gpk in tqdm(targets, unit='game'):
            box = fetch_box(gpk)
            if box is None:
                continue
            rows = parse_box(box)
            for r in rows:
                r['game_pk'] = int(gpk)
                stmt = pg_insert(HitterGameStats).values(**r)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[HitterGameStats.hitter_id, HitterGameStats.game_pk],
                    set_={
                        'runs': stmt.excluded.runs,
                        'stolen_bases': stmt.excluded.stolen_bases,
                        'caught_stealing': stmt.excluded.caught_stealing,
                        'sac_flies': stmt.excluded.sac_flies,
                    },
                )
                session.execute(stmt)
            session.commit()
            time.sleep(RATE_LIMIT_S)
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', nargs='+', type=int, default=DEFAULT_SEASONS)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    ingest_runs(args.seasons, force=args.force)


if __name__ == '__main__':
    main()
