"""
Fetch and parse the feed/live endpoint for a single game.
Returns (game_update, at_bats, pitches) shaped for the Postgres schema.
"""
from __future__ import annotations

import requests
from datetime import date as _date

from config import MLB_API_BASE2


PITCH_FAMILY = {
    'FF': 'fastball', 'FA': 'fastball',
    'FT': 'sinker',   'SI': 'sinker',
    'FC': 'cutter',
    'SL': 'slider',   'ST': 'slider',   'SV': 'slider',
    'CU': 'curveball', 'KC': 'curveball', 'CS': 'curveball',
    'CH': 'changeup', 'FO': 'changeup', 'SC': 'changeup',
    'FS': 'splitter',
    'KN': 'knuckleball',
    'EP': 'eephus',
}


def fetch_game_feed(game_pk: int) -> dict | None:
    url = f'{MLB_API_BASE2}/game/{game_pk}/feed/live'
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def parse_game(feed: dict, game_meta: dict) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """
    Returns (game_update, at_bats, pitches, players).

    game_update — fields to merge into the `games` row (scores, weather JSON).
    at_bats — ready for bulk insert into at_bats.
    pitches — ready for bulk insert into pitches.
    players — one dict per unique hitter/pitcher seen in this game.
    """
    game_data = feed.get('gameData', {})
    live_data = feed.get('liveData', {})
    all_plays = live_data.get('plays', {}).get('allPlays', [])

    # Final scores
    linescore = live_data.get('linescore', {})
    home_score = linescore.get('teams', {}).get('home', {}).get('runs')
    away_score = linescore.get('teams', {}).get('away', {}).get('runs')

    weather = _parse_weather(game_data.get('weather', {}))

    game_update = {
        'game_pk':    game_meta['game_pk'],
        'home_score': home_score,
        'away_score': away_score,
        'weather':    weather,
    }

    game_pk = game_meta['game_pk']
    game_date: _date | None = game_meta.get('game_date')
    season = game_meta.get('season')

    at_bats:  list[dict] = []
    pitches:  list[dict] = []
    players:  dict[int, dict] = {}

    for play in all_plays:
        if not play.get('about', {}).get('isComplete', False):
            continue

        about   = play['about']
        matchup = play.get('matchup', {})
        result  = play.get('result', {})

        pitcher = matchup.get('pitcher', {}) or {}
        batter  = matchup.get('batter', {}) or {}
        pitcher_id   = pitcher.get('id')
        hitter_id    = batter.get('id')
        pitcher_name = pitcher.get('fullName')
        hitter_name  = batter.get('fullName')
        pitcher_hand = matchup.get('pitchHand', {}).get('code')
        hitter_side  = matchup.get('batSide', {}).get('code')
        ab_index     = about.get('atBatIndex', 0)
        inning       = about.get('inning', 0)
        half         = about.get('halfInning', '')

        if hitter_id and hitter_id not in players:
            players[hitter_id] = {
                'player_id': hitter_id,
                'full_name': hitter_name or '',
                'bat_side': hitter_side,
            }
        if pitcher_id and pitcher_id not in players:
            players[pitcher_id] = {
                'player_id': pitcher_id,
                'full_name': pitcher_name or '',
                'pitch_hand': pitcher_hand,
            }

        # hitData sits on the play-ending pitch event, not the play itself
        hit_data = {}
        for ev in play.get('playEvents', []):
            if ev.get('hitData'):
                hit_data = ev['hitData']
                break
        coords = hit_data.get('coordinates', {}) or {}

        at_bats.append({
            'game_pk':            game_pk,
            'at_bat_index':       ab_index,
            'inning':             inning,
            'half_inning':        half,
            'hitter_id':          hitter_id,
            'pitcher_id':         pitcher_id,
            'hitter_side':        hitter_side,
            'pitcher_hand':       pitcher_hand,
            'event':              result.get('event', '')[:80],
            'event_type':         result.get('eventType', '')[:80],
            'description':        result.get('description', ''),
            'rbi':                result.get('rbi', 0) or 0,
            'exit_velocity':      hit_data.get('launchSpeed'),
            'launch_angle':       hit_data.get('launchAngle'),
            'launch_speed_angle': hit_data.get('launchSpeedAngle'),
            'total_distance':     _to_float(hit_data.get('totalDistance')),
            'hit_coord_x':        _to_float(coords.get('coordX')),
            'hit_coord_y':        _to_float(coords.get('coordY')),
            'trajectory':         hit_data.get('trajectory'),
            'hardness':           hit_data.get('hardness'),
            'location':           str(hit_data.get('location')) if hit_data.get('location') is not None else None,
            'game_date':          game_date,
            'season':             season,
        })

        # Pitch records
        for event in play.get('playEvents', []):
            if event.get('type') != 'pitch':
                continue

            details    = event.get('details', {})
            pitch_data = event.get('pitchData', {})
            pcoords    = pitch_data.get('coordinates', {})
            count      = event.get('count', {})
            ptype_code = details.get('type', {}).get('code')

            breaks = pitch_data.get('breaks') or {}
            # MLB sometimes reports IVB under `breakVerticalInduced`; prefer
            # the more widely available `pfxZ` when present.
            pfx_z = pcoords.get('pfxZ')
            if pfx_z is None:
                pfx_z = breaks.get('breakVerticalInduced')
            pitches.append({
                'game_pk':       game_pk,
                'at_bat_index':  ab_index,
                'pitch_index':   event.get('index', 0),
                'hitter_id':     hitter_id,
                'pitcher_id':    pitcher_id,
                'pitch_type':    ptype_code,
                'pitch_family':  PITCH_FAMILY.get(ptype_code or ''),
                'start_speed':   pitch_data.get('startSpeed'),
                'end_speed':     pitch_data.get('endSpeed'),
                'spin_rate':     breaks.get('spinRate'),
                'spin_direction': breaks.get('spinDirection'),
                'px':            pcoords.get('pX'),
                'pz':            pcoords.get('pZ'),
                # Phase-4: movement + release geometry
                'pfx_x':         _to_float(pcoords.get('pfxX')),
                'pfx_z':         _to_float(pfx_z),
                'x0':            _to_float(pcoords.get('x0')),
                'z0':            _to_float(pcoords.get('z0')),
                'extension':     _to_float(pitch_data.get('extension')),
                'plate_time':    _to_float(pitch_data.get('plateTime')),
                'pitch_result':  details.get('call', {}).get('code') or details.get('description'),
                'zone':          pitch_data.get('zone'),
                'balls':         count.get('balls', 0),
                'strikes':       count.get('strikes', 0),
                'game_date':     game_date,
                'season':        season,
            })

    return game_update, at_bats, pitches, list(players.values())


def _parse_weather(w: dict) -> dict:
    """Parse MLB weather strings into structured JSON."""
    temp_str = w.get('temp', '')
    wind_str = w.get('wind', '')
    temp = None
    try:
        temp = int(temp_str)
    except (ValueError, TypeError):
        pass

    wind_speed = None
    wind_dir = ''
    if wind_str:
        parts = wind_str.split(',')
        try:
            wind_speed = int(parts[0].strip().split()[0])
        except (ValueError, IndexError):
            pass
        if len(parts) > 1:
            wind_dir = parts[1].strip()

    return {
        'condition':    w.get('condition', ''),
        'tempF':        temp,
        'windSpeedMph': wind_speed,
        'windDir':      wind_dir,
    }


def _to_float(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None
