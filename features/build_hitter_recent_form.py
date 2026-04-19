"""
Rolling recent-form stats per hitter — last 7/15/30 game aggregates + hot/cold signal.
Writes: hitter_recent_form.
"""
from __future__ import annotations

import pandas as pd

from config import get_engine, get_session
from db.models import HitterRecentForm
from db.io import bulk_upsert


HIT_EVENTS  = {'single', 'double', 'triple', 'home_run'}
WALK_EVENTS = {'walk', 'intent_walk'}
SAC_EVENTS  = {'sac_fly', 'sac_bunt', 'hit_by_pitch'}


def _game_line(grp: pd.DataFrame) -> dict:
    et = grp['event_type'].fillna('')
    pa = len(grp)
    hits = int(et.isin(HIT_EVENTS).sum())
    ab = int((~et.isin(WALK_EVENTS | SAC_EVENTS)).sum())
    bb = int(et.isin(WALK_EVENTS).sum())
    k  = int(et.isin({'strikeout', 'strikeout_double_play'}).sum())
    hr = int((et == 'home_run').sum())
    slg_n = int((et == 'single').sum()) + int((et == 'double').sum()) * 2 + int((et == 'triple').sum()) * 3 + hr * 4
    return {'pa': int(pa), 'ab': ab, 'hits': hits, 'hr': hr, 'bb': bb, 'k': k, 'slg_n': slg_n}


def _aggregate(rows: list[dict]) -> dict:
    pa = sum(r['pa'] for r in rows)
    ab = sum(r['ab'] for r in rows)
    hits = sum(r['hits'] for r in rows)
    hr = sum(r['hr'] for r in rows)
    bb = sum(r['bb'] for r in rows)
    k  = sum(r['k']  for r in rows)
    slg_n = sum(r['slg_n'] for r in rows)
    return {
        'games': len(rows),
        'pa': pa, 'ab': ab, 'hits': hits, 'hr': hr, 'bb': bb, 'k': k,
        'avg':    round(hits / ab, 4)      if ab else 0,
        'slg':    round(slg_n / ab, 4)     if ab else 0,
        'obp':    round((hits + bb) / pa, 4) if pa else 0,
        'kRate':  round(k / pa, 4)         if pa else 0,
        'hrRate': round(hr / max(ab, 1), 4),
    }


def run():
    engine = get_engine()

    print('Loading at-bats for recent form…')
    df = pd.read_sql_query(
        """
        SELECT hitter_id, game_pk, game_date, event_type
        FROM at_bats
        """,
        engine,
    )
    if df.empty:
        print('No data.')
        return
    print(f'  {len(df):,} at-bats')

    df['game_date'] = pd.to_datetime(df['game_date'])

    print('Computing per-game stats…')
    per_game = (
        df.groupby(['hitter_id', 'game_pk', 'game_date'])
          .apply(_game_line)
          .reset_index(name='line')
    )
    per_game = pd.concat(
        [per_game[['hitter_id', 'game_pk', 'game_date']],
         pd.json_normalize(per_game['line'])],
        axis=1,
    )

    print('Computing rolling windows…')
    records: list[dict] = []
    for hitter_id, grp in per_game.groupby('hitter_id'):
        grp  = grp.sort_values('game_date')
        rows = grp.to_dict('records')
        if len(rows) < 3:
            continue

        last7  = _aggregate(rows[-7:])
        last15 = _aggregate(rows[-15:])
        last30 = _aggregate(rows[-30:])
        season = _aggregate(rows)

        baseline_avg = season['avg'] or 0.248
        recent_avg   = last7['avg']
        form_ratio   = recent_avg / baseline_avg if baseline_avg > 0 and last7['pa'] >= 15 else 1.0

        if form_ratio >= 1.15:
            signal = 'hot'
        elif form_ratio <= 0.78:
            signal = 'cold'
        else:
            signal = 'normal'

        records.append({
            'hitter_id':   int(hitter_id),
            'form_signal': signal,
            'form_ratio':  round(float(form_ratio), 3),
            'last_7':      last7,
            'last_15':     last15,
            'last_30':     last30,
        })

    session = get_session()
    try:
        bulk_upsert(session, HitterRecentForm, records, pk_cols=['hitter_id'])
        session.commit()
        print(f'  recent form: {len(records)} rows')
    finally:
        session.close()


if __name__ == '__main__':
    run()
