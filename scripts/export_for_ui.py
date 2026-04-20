"""
Export every Next.js query's output as static JSON snapshots.

Writes to `../sports-oracle-ui/public/data/` so the UI can serve Render
without ever touching a database. Run this after `run_daily.py` finishes.

Output layout (under public/data/):
  meta.json                    # { dates: [...], exported_at, today }
  calibration.json             # getCalibration() full-history
  slate/{date}.json            # getSlate()
  props/{date}.json            # getPropsSlate()
  nrfi/{date}.json             # getNrfiSlate()
  batters/{date}.json          # getBattersSlate() — players page
  pitchers/{date}.json         # getPitchersSlate()
  dfs/{date}.json              # getDfsPool() — FanDuel slate
  games/{gamePk}.json          # getGameDetail() — only for dates in range
  hitters/{id}.json            # getHitterDetail() — only hitters projected recently
  pitchers-detail/{id}.json    # getPitcherDetail()

Run:
  python -m scripts.export_for_ui                       # last 7 dates
  python -m scripts.export_for_ui --days 14             # last 14 dates
  python -m scripts.export_for_ui --no-details          # skip per-player files
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from sqlalchemy import text

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..'))

from config import get_engine


UI_ROOT = Path(_here).parent.parent / 'sports-oracle-ui'
DATA_DIR = UI_ROOT / 'public' / 'data'


# ══════════════════════ Serialization helpers ══════════════════════

def _encode(v):
    """JSON-safe conversion for Postgres/pandas-ish types."""
    if v is None:
        return None
    if isinstance(v, (str, int, bool)):
        return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [_encode(x) for x in v]
    if isinstance(v, dict):
        return {k: _encode(val) for k, val in v.items()}
    return str(v)


def _write_json(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_encode(obj), f, separators=(',', ':'))
    return path.stat().st_size


def _rows_to_dicts(conn, sql_text, params=None) -> list[dict]:
    res = conn.execute(text(sql_text), params or {})
    cols = list(res.keys())
    return [dict(zip(cols, row)) for row in res.fetchall()]


# ══════════════════════ Date listing ══════════════════════

def _recent_dates(conn, days: int) -> list[str]:
    rows = conn.execute(text("""
        SELECT DISTINCT game_date::text
        FROM projections
        ORDER BY game_date DESC
        LIMIT :n
    """), {'n': days}).fetchall()
    return [r[0] for r in rows]


# ══════════════════════ Page exporters ══════════════════════

def export_slate(conn, game_date: str) -> int:
    rows = _rows_to_dicts(conn, """
        WITH src AS (
          SELECT game_pk, side,
                 CASE WHEN BOOL_OR(lineup_source = 'confirmed') THEN 'confirmed' ELSE 'fallback' END AS lineup_source
          FROM projections WHERE game_date = :d
          GROUP BY game_pk, side
        )
        SELECT
          g.game_pk, g.game_date::text AS game_date, g.status,
          g.game_time_utc, g.weather,
          g.home_team_id, g.away_team_id, g.venue_id,
          t_home.abbrev AS home_abbrev, t_home.name AS home_name,
          t_away.abbrev AS away_abbrev, t_away.name AS away_name,
          v.name AS venue_name,
          np.nrfi_pct, np.yrfi_pct, np.nrfi_prob,
          np.home_pitcher, np.away_pitcher,
          s_home.lineup_source AS home_lineup_source,
          s_away.lineup_source AS away_lineup_source
        FROM games g
        LEFT JOIN teams  t_home ON t_home.team_id = g.home_team_id
        LEFT JOIN teams  t_away ON t_away.team_id = g.away_team_id
        LEFT JOIN venues v      ON v.venue_id    = g.venue_id
        LEFT JOIN nrfi_projections np ON np.game_pk = g.game_pk
        LEFT JOIN src s_home ON s_home.game_pk = g.game_pk AND s_home.side = 'home'
        LEFT JOIN src s_away ON s_away.game_pk = g.game_pk AND s_away.side = 'away'
        WHERE g.game_date = :d
        ORDER BY g.game_time_utc ASC
    """, {'d': game_date})

    # Last-7 NRFI trend per home team (excluding this date)
    trend = _rows_to_dicts(conn, """
        WITH ranked AS (
          SELECT g.home_team_id, g.game_date::text AS d, np.nrfi_pct,
                 ROW_NUMBER() OVER (PARTITION BY g.home_team_id ORDER BY g.game_date DESC) AS rn
          FROM nrfi_projections np
          JOIN games g ON g.game_pk = np.game_pk
          WHERE g.game_date < :d AND np.nrfi_pct IS NOT NULL
        )
        SELECT home_team_id, d, nrfi_pct FROM ranked
        WHERE rn <= 7 ORDER BY home_team_id, d ASC
    """, {'d': game_date})
    tmap: dict[int, list[dict]] = {}
    for r in trend:
        tmap.setdefault(r['home_team_id'], []).append({'d': r['d'], 'pct': r['nrfi_pct']})
    for r in rows:
        r['nrfiTrend'] = tmap.get(r['home_team_id'], [])

    return _write_json(DATA_DIR / 'slate' / f'{game_date}.json', rows)


def export_props(conn, game_date: str) -> int:
    season = int(game_date[:4])
    rows = _rows_to_dicts(conn, """
        WITH pitcher_hr9 AS (
          SELECT pitcher_id,
            (SUM(CASE WHEN event_type='home_run' THEN 1 ELSE 0 END)::float
              / NULLIF(SUM(CASE WHEN event_type IN ('strikeout','strikeout_double_play','walk','intent_walk','field_out','force_out','grounded_into_double_play','fielders_choice_out','single','double','triple','home_run','hit_by_pitch','sac_fly','sac_bunt','field_error') THEN 1 ELSE 0 END), 0)) * 38.0 AS hr9
          FROM at_bats WHERE season = :season
          GROUP BY pitcher_id
          HAVING SUM(CASE WHEN event_type IN ('strikeout','strikeout_double_play','walk','intent_walk','field_out','force_out','grounded_into_double_play','fielders_choice_out','single','double','triple','home_run','hit_by_pitch','sac_fly','sac_bunt','field_error') THEN 1 ELSE 0 END) >= 50
        ),
        hbb AS (
          SELECT hitter_id,
                 SUM(pa * COALESCE(barrel_pct, 0))::float    / NULLIF(SUM(CASE WHEN barrel_pct IS NOT NULL THEN pa END), 0) AS h_barrel_pct,
                 SUM(pa * COALESCE(fb_pct, 0))::float        / NULLIF(SUM(CASE WHEN fb_pct IS NOT NULL THEN pa END), 0)     AS h_fb_pct,
                 SUM(pa * COALESCE(hard_hit_pct, 0))::float  / NULLIF(SUM(CASE WHEN hard_hit_pct IS NOT NULL THEN pa END), 0) AS h_hard_hit_pct
          FROM hitter_pitch_splits WHERE season = :season GROUP BY hitter_id
        ),
        last15 AS (
          SELECT hitter_id, SUM(hr) AS last15_hr FROM (
            SELECT hitter_id, game_pk,
                   SUM(CASE WHEN event_type='home_run' THEN 1 ELSE 0 END) AS hr,
                   ROW_NUMBER() OVER (PARTITION BY hitter_id ORDER BY game_date DESC) AS rn
            FROM at_bats WHERE game_date < :d
            GROUP BY hitter_id, game_pk, game_date
          ) t WHERE rn <= 15 GROUP BY hitter_id
        )
        SELECT
          p.hitter_id, p.game_pk,
          pl.full_name AS hitter_name, p.hitter_hand,
          p.pitcher_id,
          pp.full_name AS pitcher_name, pp.pitch_hand AS pitcher_hand,
          p.lineup_slot, p.expected_pa, p.proj, p.side,
          g.home_team_id, g.away_team_id, g.weather, g.venue_id,
          v.name AS venue_name,
          pf.hr_factor AS park_hr_factor,
          hsp.pull_pct, hsp.avg_exit_velo, hsp.avg_launch_angle,
          hrf.form_ratio,
          phr.hr9 AS pitcher_hr9,
          pprj.fip AS pitcher_fip,
          hbb.h_barrel_pct  AS barrel_pct,
          hbb.h_hard_hit_pct AS hard_hit_pct,
          l15.last15_hr     AS last15_hr,
          t_own.abbrev AS own_abbrev,
          t_opp.abbrev AS opp_abbrev
        FROM projections p
        LEFT JOIN players pl        ON pl.player_id = p.hitter_id
        LEFT JOIN players pp        ON pp.player_id = p.pitcher_id
        LEFT JOIN games g           ON g.game_pk    = p.game_pk
        LEFT JOIN venues v          ON v.venue_id   = g.venue_id
        LEFT JOIN park_factors pf   ON pf.venue_id  = g.venue_id
        LEFT JOIN hitter_spray_profiles hsp ON hsp.hitter_id = p.hitter_id
        LEFT JOIN hitter_recent_form hrf    ON hrf.hitter_id = p.hitter_id
        LEFT JOIN pitcher_hr9 phr   ON phr.pitcher_id = p.pitcher_id
        LEFT JOIN pitcher_projections pprj ON pprj.pitcher_id = p.pitcher_id AND pprj.game_pk = p.game_pk
        LEFT JOIN hbb               ON hbb.hitter_id  = p.hitter_id
        LEFT JOIN last15 l15        ON l15.hitter_id  = p.hitter_id
        LEFT JOIN teams t_own ON t_own.team_id = CASE WHEN p.side='home' THEN g.home_team_id ELSE g.away_team_id END
        LEFT JOIN teams t_opp ON t_opp.team_id = CASE WHEN p.side='home' THEN g.away_team_id ELSE g.home_team_id END
        WHERE p.game_date = :d
    """, {'d': game_date, 'season': season})
    return _write_json(DATA_DIR / 'props' / f'{game_date}.json', rows)


def export_nrfi(conn, game_date: str) -> int:
    rows = _rows_to_dicts(conn, """
        SELECT
          np.game_pk, np.game_date::text AS game_date,
          np.nrfi_prob, np.nrfi_pct, np.yrfi_pct,
          np.home_xr, np.away_xr,
          np.home_p_scoreless, np.away_p_scoreless,
          np.home_p_score, np.away_p_score,
          np.top_threats, np.home_top_batters, np.away_top_batters,
          np.home_pitcher, np.away_pitcher,
          g.home_team_id, g.away_team_id,
          g.venue_id, v.name AS venue_name,
          g.game_time_utc, g.weather,
          t_home.abbrev AS home_abbrev, t_away.abbrev AS away_abbrev
        FROM nrfi_projections np
        LEFT JOIN games g        ON g.game_pk    = np.game_pk
        LEFT JOIN venues v       ON v.venue_id   = g.venue_id
        LEFT JOIN teams t_home   ON t_home.team_id = g.home_team_id
        LEFT JOIN teams t_away   ON t_away.team_id = g.away_team_id
        WHERE np.game_date = :d
        ORDER BY np.nrfi_pct DESC NULLS LAST
    """, {'d': game_date})
    return _write_json(DATA_DIR / 'nrfi' / f'{game_date}.json', rows)


def export_batters(conn, game_date: str) -> int:
    season = int(game_date[:4])
    rows = _rows_to_dicts(conn, """
        WITH season_avg AS (
          SELECT hitter_id, AVG(dk)::float AS season_dk_avg, AVG(fd)::float AS season_fd_avg,
                 COUNT(*)::int AS games
          FROM (
            SELECT hitter_id, game_pk,
              SUM(CASE event_type
                WHEN 'single' THEN 3 WHEN 'double' THEN 5 WHEN 'triple' THEN 8
                WHEN 'home_run' THEN 10 WHEN 'walk' THEN 2 WHEN 'intent_walk' THEN 2
                WHEN 'hit_by_pitch' THEN 2 WHEN 'strikeout' THEN -0.5
                WHEN 'strikeout_double_play' THEN -0.5 ELSE 0 END)
                + COALESCE(SUM(rbi), 0) * 2 AS dk,
              SUM(CASE event_type
                WHEN 'single' THEN 3 WHEN 'double' THEN 6 WHEN 'triple' THEN 9
                WHEN 'home_run' THEN 12 WHEN 'walk' THEN 3 WHEN 'intent_walk' THEN 3
                WHEN 'hit_by_pitch' THEN 3 ELSE 0 END)
                + COALESCE(SUM(rbi), 0) * 3.5 AS fd
            FROM at_bats WHERE season = :season
            GROUP BY hitter_id, game_pk
          ) pg GROUP BY hitter_id
        )
        SELECT
          p.hitter_id, p.game_pk, pl.full_name AS hitter_name, p.hitter_hand,
          p.pitcher_id, pp.full_name AS pitcher_name,
          p.side, p.lineup_slot,
          p.dk_pts, p.fd_pts, p.tuned_dk_pts, p.ml_dk_pts, p.ml_fd_pts,
          p.ml_delta, p.blend_dk_pts, p.blend_fd_pts,
          p.factors, p.factor_score, p.proj, p.expected_pa,
          g.home_team_id, g.away_team_id, t_home.abbrev AS home_abbrev,
          t_own.abbrev AS team, t_opp.abbrev AS opp,
          sa.season_dk_avg, sa.season_fd_avg, sa.games AS season_games
        FROM projections p
        LEFT JOIN players pl ON pl.player_id = p.hitter_id
        LEFT JOIN players pp ON pp.player_id = p.pitcher_id
        LEFT JOIN games g    ON g.game_pk    = p.game_pk
        LEFT JOIN teams t_home ON t_home.team_id = g.home_team_id
        LEFT JOIN teams t_own  ON t_own.team_id = CASE WHEN p.side='home' THEN g.home_team_id ELSE g.away_team_id END
        LEFT JOIN teams t_opp  ON t_opp.team_id = CASE WHEN p.side='home' THEN g.away_team_id ELSE g.home_team_id END
        LEFT JOIN season_avg sa ON sa.hitter_id = p.hitter_id
        WHERE p.game_date = :d
        ORDER BY p.dk_pts DESC
    """, {'d': game_date, 'season': season})
    return _write_json(DATA_DIR / 'batters' / f'{game_date}.json', rows)


def export_pitchers(conn, game_date: str) -> int:
    rows = _rows_to_dicts(conn, """
        SELECT
          pp.pitcher_id, pp.game_pk, pl.full_name AS pitcher_name, pl.pitch_hand AS pitcher_hand,
          pp.side, pp.dk_pts, pp.fd_pts, pp.ml_dk_pts, pp.ml_fd_pts, pp.ml_delta,
          pp.fip, pp.games_started, pp.proj,
          g.home_team_id, g.away_team_id,
          t_own.abbrev AS team, t_opp.abbrev AS opp
        FROM pitcher_projections pp
        LEFT JOIN players pl ON pl.player_id = pp.pitcher_id
        LEFT JOIN games g    ON g.game_pk    = pp.game_pk
        LEFT JOIN teams t_own ON t_own.team_id = CASE WHEN pp.side='home' THEN g.home_team_id ELSE g.away_team_id END
        LEFT JOIN teams t_opp ON t_opp.team_id = CASE WHEN pp.side='home' THEN g.away_team_id ELSE g.home_team_id END
        WHERE pp.game_date = :d
        ORDER BY pp.dk_pts DESC
    """, {'d': game_date})
    return _write_json(DATA_DIR / 'pitchers' / f'{game_date}.json', rows)


def export_dfs(conn, game_date: str) -> int:
    season = int(game_date[:4])
    rows = _rows_to_dicts(conn, """
        WITH latest_hitter AS (
          SELECT DISTINCT ON (hitter_id) hitter_id, blend_fd_pts, ml_fd_pts, fd_pts AS factor_fd_pts
          FROM projections WHERE game_date = :d
          ORDER BY hitter_id, game_pk
        ),
        latest_pitcher AS (
          SELECT DISTINCT ON (pitcher_id) pitcher_id, ml_fd_pts, fd_pts AS factor_fd_pts
          FROM pitcher_projections WHERE game_date = :d
          ORDER BY pitcher_id, game_pk
        ),
        season_avg AS (
          SELECT hitter_id, AVG(dk)::float AS season_fd_avg FROM (
            SELECT hitter_id, game_pk,
              SUM(CASE event_type
                WHEN 'single' THEN 3 WHEN 'double' THEN 6 WHEN 'triple' THEN 9
                WHEN 'home_run' THEN 12 WHEN 'walk' THEN 3 WHEN 'intent_walk' THEN 3
                WHEN 'hit_by_pitch' THEN 3 ELSE 0 END) + COALESCE(SUM(rbi), 0) * 3.5 AS dk
            FROM at_bats WHERE season = :season GROUP BY hitter_id, game_pk
          ) pg GROUP BY hitter_id
        )
        SELECT f.fd_player_id, f.fd_name, f.position, f.salary, f.fppg,
               f.team, f.opponent, f.game, f.batting_order, f.injury_indicator,
               f.matched_player_id,
               COALESCE(h.blend_fd_pts,  p.ml_fd_pts)    AS blend_fd_pts,
               COALESCE(h.ml_fd_pts,     p.ml_fd_pts)    AS ml_fd_pts,
               COALESCE(h.factor_fd_pts, p.factor_fd_pts) AS factor_fd_pts,
               s.season_fd_avg
        FROM fd_slate_prices f
        LEFT JOIN latest_hitter  h ON h.hitter_id  = f.matched_player_id
        LEFT JOIN latest_pitcher p ON p.pitcher_id = f.matched_player_id
        LEFT JOIN season_avg     s ON s.hitter_id  = f.matched_player_id
        WHERE f.slate_date = :d
    """, {'d': game_date, 'season': season})
    return _write_json(DATA_DIR / 'dfs' / f'{game_date}.json', rows)


def export_calibration(conn) -> int:
    # Simplified — read projection_actuals + matching projection columns, dump raw.
    rows = _rows_to_dicts(conn, """
        SELECT pa.game_date::text AS game_date, pa.actual_dk_pts,
               pa.proj_dk_pts AS factor_dk,
               p.tuned_dk_pts, p.ml_dk_pts, p.blend_dk_pts
        FROM projection_actuals pa
        JOIN projections p ON p.hitter_id = pa.hitter_id AND p.game_pk = pa.game_pk
        ORDER BY pa.game_date ASC
    """)
    return _write_json(DATA_DIR / 'calibration.json', {'rows': rows})


def export_game_detail(conn, game_pk: int) -> int:
    game = _rows_to_dicts(conn, """
        SELECT g.game_pk, g.game_date::text AS game_date, g.game_time_utc,
               g.status, g.home_score, g.away_score, g.weather,
               g.home_team_id, g.away_team_id, g.venue_id, v.name AS venue_name,
               t_home.abbrev AS home_abbrev, t_home.name AS home_name,
               t_away.abbrev AS away_abbrev, t_away.name AS away_name
        FROM games g
        LEFT JOIN venues v     ON v.venue_id = g.venue_id
        LEFT JOIN teams t_home ON t_home.team_id = g.home_team_id
        LEFT JOIN teams t_away ON t_away.team_id = g.away_team_id
        WHERE g.game_pk = :gpk
    """, {'gpk': game_pk})
    if not game:
        return 0

    hitters = _rows_to_dicts(conn, """
        SELECT p.hitter_id, pl.full_name AS hitter_name, p.hitter_hand,
               p.pitcher_id, p.lineup_slot, p.side,
               p.dk_pts, p.fd_pts, p.baseline_dk_pts, p.dk_delta,
               p.factors, p.factor_score, p.proj, p.expected_pa,
               p.tuned_dk_pts, p.ml_dk_pts, p.ml_fd_pts, p.ml_delta,
               p.blend_dk_pts, p.blend_fd_pts
        FROM projections p
        LEFT JOIN players pl ON pl.player_id = p.hitter_id
        WHERE p.game_pk = :gpk
        ORDER BY p.side ASC, p.lineup_slot ASC
    """, {'gpk': game_pk})

    pitchers = _rows_to_dicts(conn, """
        SELECT pp.pitcher_id, pl.full_name AS pitcher_name, pl.pitch_hand AS pitcher_hand,
               pp.side, pp.dk_pts, pp.fd_pts, pp.ml_dk_pts, pp.ml_fd_pts, pp.ml_delta,
               pp.proj, pp.fip, pp.games_started
        FROM pitcher_projections pp
        LEFT JOIN players pl ON pl.player_id = pp.pitcher_id
        WHERE pp.game_pk = :gpk
    """, {'gpk': game_pk})

    nrfi = _rows_to_dicts(conn, """
        SELECT * FROM nrfi_projections WHERE game_pk = :gpk
    """, {'gpk': game_pk})

    data = {'game': game[0], 'hitters': hitters, 'pitchers': pitchers, 'nrfi': nrfi[0] if nrfi else None}
    return _write_json(DATA_DIR / 'games' / f'{game_pk}.json', data)


def export_hitter_detail(conn, hitter_id: int) -> int:
    rows = _rows_to_dicts(conn, 'SELECT * FROM players WHERE player_id = :id', {'id': hitter_id})
    if not rows:
        return 0
    player = rows[0]

    form = _rows_to_dicts(conn, 'SELECT * FROM hitter_recent_form WHERE hitter_id = :id', {'id': hitter_id})
    spray = _rows_to_dicts(conn, 'SELECT * FROM hitter_spray_profiles WHERE hitter_id = :id', {'id': hitter_id})
    similar = _rows_to_dicts(conn, 'SELECT * FROM hitter_similar WHERE hitter_id = :id', {'id': hitter_id})
    splits = _rows_to_dicts(conn, """
        SELECT * FROM hitter_pitch_splits WHERE hitter_id = :id ORDER BY season DESC
    """, {'id': hitter_id})
    recent = _rows_to_dicts(conn, """
        SELECT game_date::text AS game_date, dk_pts, fd_pts, proj, factor_score, factors
        FROM projections WHERE hitter_id = :id
        ORDER BY game_date DESC LIMIT 30
    """, {'id': hitter_id})
    spray_hits = _rows_to_dicts(conn, """
        SELECT hit_coord_x, hit_coord_y, event_type, exit_velocity, launch_angle,
               game_date::text AS game_date
        FROM at_bats
        WHERE hitter_id = :id AND hit_coord_x IS NOT NULL AND hit_coord_y IS NOT NULL
        ORDER BY game_date DESC LIMIT 200
    """, {'id': hitter_id})

    data = {
        'player': player,
        'form': form[0] if form else None,
        'spray': spray[0] if spray else None,
        'similar': similar[0] if similar else None,
        'splits': splits,
        'recentProjections': recent,
        'sprayHits': spray_hits,
    }
    return _write_json(DATA_DIR / 'hitters' / f'{hitter_id}.json', data)


def export_pitcher_detail(conn, pitcher_id: int) -> int:
    rows = _rows_to_dicts(conn, 'SELECT * FROM players WHERE player_id = :id', {'id': pitcher_id})
    if not rows:
        return 0
    player = rows[0]
    seasons = _rows_to_dicts(conn, """
        SELECT * FROM pitcher_season_stats WHERE pitcher_id = :id ORDER BY season DESC
    """, {'id': pitcher_id})
    recent = _rows_to_dicts(conn, """
        SELECT * FROM pitcher_projections WHERE pitcher_id = :id
        ORDER BY game_date DESC LIMIT 15
    """, {'id': pitcher_id})
    reconciled = _rows_to_dicts(conn, """
        SELECT game_date::text AS game_date, proj_dk_pts, actual_dk_pts, dk_error, abs_dk_error, actual
        FROM pitcher_projection_actuals WHERE pitcher_id = :id
        ORDER BY game_date DESC LIMIT 15
    """, {'id': pitcher_id})
    data = {'player': player, 'seasons': seasons, 'recentProj': recent, 'reconciled': reconciled}
    return _write_json(DATA_DIR / 'pitchers-detail' / f'{pitcher_id}.json', data)


# ══════════════════════ Main ══════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=7, help='How many recent dates to export')
    ap.add_argument('--no-details', action='store_true', help='Skip per-game/player detail files')
    args = ap.parse_args()

    engine = get_engine()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with engine.connect() as conn:
        dates = _recent_dates(conn, args.days)
        if not dates:
            print('No projections to export. Run the pipeline first.')
            return
        print(f'Exporting {len(dates)} dates: {dates[0]} → {dates[-1]}')

        total_bytes = 0
        for d in dates:
            b = 0
            b += export_slate(conn, d)
            b += export_props(conn, d)
            b += export_nrfi(conn, d)
            b += export_batters(conn, d)
            b += export_pitchers(conn, d)
            try:
                b += export_dfs(conn, d)
            except Exception as exc:
                print(f'  (dfs export skipped for {d}: {exc})')
            total_bytes += b
            print(f'  {d}: {b/1024:.1f} KB')

        total_bytes += export_calibration(conn)

        if not args.no_details:
            today = dates[0]
            print(f'\nExporting detail pages for games on {today}…')
            game_pks = [r[0] for r in conn.execute(text(
                'SELECT DISTINCT game_pk FROM projections WHERE game_date = :d'
            ), {'d': today}).fetchall()]
            for gpk in game_pks:
                total_bytes += export_game_detail(conn, int(gpk))
            print(f'  {len(game_pks)} games')

            # Hitter & pitcher detail for anyone projected in the last N days
            id_rows = conn.execute(text("""
                SELECT DISTINCT hitter_id FROM projections
                WHERE game_date = ANY(:dates)
            """), {'dates': dates}).fetchall()
            hitter_ids = [r[0] for r in id_rows]
            for hid in hitter_ids:
                total_bytes += export_hitter_detail(conn, int(hid))
            print(f'  {len(hitter_ids)} hitters')

            pid_rows = conn.execute(text("""
                SELECT DISTINCT pitcher_id FROM pitcher_projections
                WHERE game_date = ANY(:dates)
            """), {'dates': dates}).fetchall()
            pitcher_ids = [r[0] for r in pid_rows]
            for pid in pitcher_ids:
                total_bytes += export_pitcher_detail(conn, int(pid))
            print(f'  {len(pitcher_ids)} pitchers')

        meta = {
            'dates': dates,
            'today': dates[0],
            'exported_at': datetime.now(timezone.utc).isoformat(),
        }
        _write_json(DATA_DIR / 'meta.json', meta)

        print(f'\nTotal: {total_bytes/1024/1024:.1f} MB → {DATA_DIR}')


if __name__ == '__main__':
    main()
