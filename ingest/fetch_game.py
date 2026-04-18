"""
Fetch and parse the feed/live endpoint for a single game.
Extracts per-pitch records and per-at-bat records.
"""
from __future__ import annotations
import requests
from config import MLB_API_BASE2


def fetch_game_feed(game_pk: int) -> dict | None:
    url = f'{MLB_API_BASE2}/game/{game_pk}/feed/live'
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def parse_game(feed: dict, game_meta: dict) -> tuple[list[dict], list[dict]]:
    """
    Returns (pitches, at_bats) for a game feed.

    pitch doc fields:
        gamePk, season, gameDate, venueId, venueName,
        inning, halfInning, atBatIndex, pitchIndex,
        pitcherId, pitcherName, pitcherHand,
        hitterId, hitterName, hitterHand,
        pitchType, pitchTypeDesc,
        startSpeed, spinRate, px, pz,
        balls, strikes,
        pitchResult (called_strike, swinging_strike, ball, foul, hit_into_play, etc.)

    at_bat doc fields:
        gamePk, season, gameDate, venueId, venueName,
        inning, halfInning, atBatIndex,
        pitcherId, pitcherName, pitcherHand,
        hitterId, hitterName, hitterHand,
        event (Single, Double, Triple, Home Run, Strikeout, Walk, etc.),
        eventType,
        exitVelocity, launchAngle, totalDistance, trajectory, hardness,
        hitCoordX, hitCoordY,
        rbi, isOut

    NOTE: hitData is on the play-ending pitch event (playEvents[i] where isInPlay=True),
    NOT on the play object itself. This is a common MLB API gotcha.
    """
    game_data = feed.get('gameData', {})
    live_data = feed.get('liveData', {})
    all_plays  = live_data.get('plays', {}).get('allPlays', [])

    # Weather is in gameData.weather
    weather_raw = game_data.get('weather', {})
    weather = _parse_weather(weather_raw)

    common = {
        'gamePk':    game_meta['gamePk'],
        'season':    game_meta['season'],
        'gameDate':  game_meta['gameDate'],
        'venueId':   game_meta['venueId'],
        'venueName': game_meta['venueName'],
        'weather':   weather,
    }

    pitches  = []
    at_bats  = []

    for play in all_plays:
        if not play.get('about', {}).get('isComplete', False):
            continue

        about   = play['about']
        matchup = play.get('matchup', {})
        result  = play.get('result', {})

        pitcher_id   = matchup.get('pitcher', {}).get('id')
        pitcher_name = matchup.get('pitcher', {}).get('fullName', '')
        pitcher_hand = matchup.get('pitchHand', {}).get('code', '')
        hitter_id    = matchup.get('batter', {}).get('id')
        hitter_name  = matchup.get('batter', {}).get('fullName', '')
        hitter_hand  = matchup.get('batSide', {}).get('code', '')
        ab_index     = about.get('atBatIndex', 0)
        inning       = about.get('inning', 0)
        half         = about.get('halfInning', '')

        ab_base = {
            **common,
            'inning':       inning,
            'halfInning':   half,
            'atBatIndex':   ab_index,
            'pitcherId':    pitcher_id,
            'pitcherName':  pitcher_name,
            'pitcherHand':  pitcher_hand,
            'hitterId':     hitter_id,
            'hitterName':   hitter_name,
            'hitterHand':   hitter_hand,
        }

        # ── At-bat record ───────────────────────────────────────────
        # hitData is on the play-ending pitch event (isInPlay=True), not on the play.
        hit_data = {}
        for ev in play.get('playEvents', []):
            if ev.get('hitData'):
                hit_data = ev['hitData']
                break

        at_bats.append({
            **ab_base,
            'event':         result.get('event', ''),
            'eventType':     result.get('eventType', ''),
            'description':   result.get('description', ''),
            'rbi':           result.get('rbi', 0),
            'isOut':         result.get('isOut', False),
            'exitVelocity':  hit_data.get('launchSpeed'),
            'launchAngle':   hit_data.get('launchAngle'),
            'totalDistance': hit_data.get('totalDistance'),
            'trajectory':    hit_data.get('trajectory'),    # line_drive, fly_ball, ground_ball, popup
            'hardness':      hit_data.get('hardness'),      # hard, medium, soft
            'hitLocation':   hit_data.get('location'),      # field zone (1-9)
            'hitCoordX':     hit_data.get('coordinates', {}).get('coordX'),
            'hitCoordY':     hit_data.get('coordinates', {}).get('coordY'),
            'totalPitches':  len([e for e in play.get('playEvents', []) if e.get('type') == 'pitch']),
        })

        # ── Pitch records ────────────────────────────────────────────
        for event in play.get('playEvents', []):
            if event.get('type') != 'pitch':
                continue

            details   = event.get('details', {})
            pitch_data = event.get('pitchData', {})
            coords    = pitch_data.get('coordinates', {})
            count     = event.get('count', {})

            pitches.append({
                **ab_base,
                'pitchIndex':    event.get('index', 0),
                'pitchType':     details.get('type', {}).get('code', ''),
                'pitchTypeDesc': details.get('type', {}).get('description', ''),
                'pitchResult':   details.get('call', {}).get('code', ''),
                'pitchResultDesc': details.get('description', ''),
                'startSpeed':    pitch_data.get('startSpeed'),
                'spinRate':      pitch_data.get('breaks', {}).get('spinRate'),
                'px':            coords.get('pX'),
                'pz':            coords.get('pZ'),
                'balls':         count.get('balls', 0),
                'strikes':       count.get('strikes', 0),
                'isLastPitch':   event.get('isLastPitch', False),
            })

    return pitches, at_bats


def _parse_weather(w: dict) -> dict:
    """Parse MLB weather string into structured fields."""
    temp_str = w.get('temp', '')
    wind_str = w.get('wind', '')  # e.g. "5 mph, L to R"
    temp = None
    try:
        temp = int(temp_str)
    except (ValueError, TypeError):
        pass

    wind_speed = None
    wind_dir   = ''
    if wind_str:
        parts = wind_str.split(',')
        try:
            wind_speed = int(parts[0].strip().split()[0])
        except (ValueError, IndexError):
            pass
        if len(parts) > 1:
            wind_dir = parts[1].strip()

    return {
        'condition': w.get('condition', ''),
        'tempF':     temp,
        'windSpeedMph': wind_speed,
        'windDir':   wind_dir,
    }
