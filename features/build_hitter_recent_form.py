"""
Compute rolling recent-form stats for hitters from mlb_at_bats.
Stores last-7, last-15, last-30 game averages and a hot/cold signal.

Output collection: mlb_hitter_recent_form
"""
from __future__ import annotations
import sys
import pandas as pd
sys.path.insert(0, '..')
from config import get_db
from pymongo import UpdateOne

HIT_EVENTS  = {'single', 'double', 'triple', 'home_run'}
WALK_EVENTS = {'walk', 'intent_walk'}
SAC_EVENTS  = {'sac_fly', 'sac_bunt', 'hit_by_pitch'}


def _game_line(grp: pd.DataFrame) -> dict:
    et   = grp['eventType'].fillna('')
    pa   = len(grp)
    hits = et.isin(HIT_EVENTS).sum()
    ab   = (~et.isin(WALK_EVENTS | SAC_EVENTS)).sum()
    bb   = et.isin(WALK_EVENTS).sum()
    k    = et.isin({'strikeout', 'strikeout_double_play'}).sum()
    hr   = (et == 'home_run').sum()
    slg_n = (
        (et == 'single').sum() +
        (et == 'double').sum() * 2 +
        (et == 'triple').sum() * 3 +
        hr * 4
    )
    return {
        'pa': int(pa), 'ab': int(ab), 'hits': int(hits),
        'hr': int(hr), 'bb': int(bb), 'k': int(k),
        'slg_n': int(slg_n),
    }


def _aggregate(rows: list[dict]) -> dict:
    pa    = sum(r['pa']    for r in rows)
    ab    = sum(r['ab']    for r in rows)
    hits  = sum(r['hits']  for r in rows)
    hr    = sum(r['hr']    for r in rows)
    bb    = sum(r['bb']    for r in rows)
    k     = sum(r['k']     for r in rows)
    slg_n = sum(r['slg_n'] for r in rows)
    return {
        'games':  len(rows),
        'pa':     pa,
        'ab':     ab,
        'hits':   hits,
        'hr':     hr,
        'bb':     bb,
        'k':      k,
        'avg':    round(hits / ab, 4)     if ab > 0  else 0,
        'slg':    round(slg_n / ab, 4)   if ab > 0  else 0,
        'obp':    round((hits + bb) / pa, 4) if pa > 0  else 0,
        'kRate':  round(k / pa, 4)        if pa > 0  else 0,
        'hrRate': round(hr / max(ab, 1), 4),
    }


def run():
    db = get_db()
    print('Loading at-bats for recent form...')
    cur = db.mlb_at_bats.find({}, {
        '_id': 0,
        'hitterId': 1, 'hitterName': 1,
        'gamePk': 1, 'gameDate': 1,
        'eventType': 1,
    })
    df = pd.DataFrame(list(cur))
    if df.empty:
        print('No data.')
        return

    print(f'  {len(df):,} at-bats loaded')
    df['gameDate'] = pd.to_datetime(df['gameDate'])

    # Per-game stats per hitter (apply _game_line to each hitter×game group)
    print('Computing per-game stats...')
    per_game = (
        df.groupby(['hitterId', 'gamePk', 'gameDate'])
        .apply(_game_line)
        .reset_index(name='line')
    )
    per_game = pd.concat(
        [per_game[['hitterId', 'gamePk', 'gameDate']],
         pd.json_normalize(per_game['line'])],
        axis=1,
    )

    print('Computing rolling windows...')
    records = []
    name_map = df.groupby('hitterId')['hitterName'].last().to_dict()

    for hitter_id, grp in per_game.groupby('hitterId'):
        grp  = grp.sort_values('gameDate')
        rows = grp.to_dict('records')
        if len(rows) < 3:
            continue

        last7  = _aggregate(rows[-7:])
        last15 = _aggregate(rows[-15:])
        last30 = _aggregate(rows[-30:])
        season = _aggregate(rows)       # full recent history (current + prev season)

        # Hot / cold vs season baseline
        baseline_avg = season['avg'] or 0.248
        recent_avg   = last7['avg']
        form_ratio   = recent_avg / baseline_avg if baseline_avg > 0 and last7['pa'] >= 15 else 1.0

        if form_ratio >= 1.15:
            form_signal = 'hot'
        elif form_ratio <= 0.78:
            form_signal = 'cold'
        else:
            form_signal = 'normal'

        records.append({
            'hitterId':    int(hitter_id),
            'hitterName':  name_map.get(hitter_id, ''),
            'last7':       last7,
            'last15':      last15,
            'last30':      last30,
            'season':      season,
            'formSignal':  form_signal,
            'formRatio':   round(float(form_ratio), 3),
            'lastGameDate': str(grp['gameDate'].max().date()),
        })

    ops = [
        UpdateOne({'hitterId': r['hitterId']}, {'$set': r}, upsert=True)
        for r in records
    ]
    if ops:
        db.mlb_hitter_recent_form.create_index('hitterId', unique=True)
        result = db.mlb_hitter_recent_form.bulk_write(ops, ordered=False)
        print(f'  recent form: {result.upserted_count} inserted, {result.modified_count} updated')
    print('Done.')


if __name__ == '__main__':
    run()
