"""
Fetch regular-season gamePks for a given season from the MLB schedule API.
Returns one dict per game with the metadata needed to upsert the `games` table.
"""
from __future__ import annotations

import requests
from datetime import datetime

from config import MLB_API_BASE


STATUS_MAP = {
    'Final':        'final',
    'Live':         'in_progress',
    'Preview':      'scheduled',
    'Postponed':    'postponed',
}


def fetch_season_games(season: int, completed_only: bool = True) -> list[dict]:
    url = (
        f'{MLB_API_BASE}/schedule'
        f'?sportId=1&season={season}&gameType=R'
        f'&hydrate=team,venue'
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_entry in data.get('dates', []):
        for g in date_entry.get('games', []):
            abstract_state = g.get('status', {}).get('abstractGameState', '')
            detailed_state = g.get('status', {}).get('detailedState', '')
            status = STATUS_MAP.get(abstract_state, abstract_state.lower())
            if detailed_state == 'Postponed':
                status = 'postponed'

            if completed_only and status != 'final':
                continue

            # Parse ISO timestamp
            game_time = g.get('gameDate', '')
            try:
                game_time_utc = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                game_time_utc = None
            game_date = game_time_utc.date() if game_time_utc else None

            home_team = g['teams']['home']['team']
            away_team = g['teams']['away']['team']
            venue = g.get('venue', {}) or {}

            games.append({
                'game_pk':      g['gamePk'],
                'game_date':    game_date,
                'season':       season,
                'home_team_id': home_team['id'],
                'away_team_id': away_team['id'],
                'venue_id':     venue.get('id'),
                'status':       status,
                'double_header': g.get('doubleHeader', 'N'),
                'game_time_utc': game_time_utc,
                'home_team_name': home_team.get('name', ''),
                'away_team_name': away_team.get('name', ''),
                'venue_name':   venue.get('name', ''),
            })
    return games


if __name__ == '__main__':
    import sys
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    games = fetch_season_games(season)
    print(f'{season}: {len(games)} completed games')
