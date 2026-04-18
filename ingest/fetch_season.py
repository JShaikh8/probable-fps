"""
Fetch all regular-season gamePks for a given season from the MLB schedule API.
Returns list of { gamePk, gameDate, homeTeamId, awayTeamId, venueId, venueName }.
"""
from __future__ import annotations
import requests
from config import MLB_API_BASE


def fetch_season_games(season: int) -> list[dict]:
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
            # Only include completed games
            state = g.get('status', {}).get('abstractGameState', '')
            if state != 'Final':
                continue
            games.append({
                'gamePk':     g['gamePk'],
                'gameDate':   g.get('gameDate', ''),
                'homeTeamId': g['teams']['home']['team']['id'],
                'homeTeam':   g['teams']['home']['team'].get('name', ''),
                'awayTeamId': g['teams']['away']['team']['id'],
                'awayTeam':   g['teams']['away']['team'].get('name', ''),
                'venueId':    g.get('venue', {}).get('id'),
                'venueName':  g.get('venue', {}).get('name', ''),
                'season':     season,
            })
    return games


if __name__ == '__main__':
    import sys
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    games = fetch_season_games(season)
    print(f'{season}: {len(games)} completed games')
