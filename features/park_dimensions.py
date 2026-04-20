"""
Outfield fence distances for MLB parks. Mirrors the UI's park-dims.ts so
Python-side projection code and the frontend stadium diagram stay in sync.

Source: publicly posted outfield-wall distances ("the numbers painted on the
wall"). Values in feet at LF / CF / RF foul lines and deadcenter respectively.
"""
from __future__ import annotations


# Best-effort prefix matches against venues.name (case-insensitive).
PARK_DIMS: dict[str, dict[str, float]] = {
    'fenway':            {'lf': 310, 'cf': 390, 'rf': 302},
    'yankee':            {'lf': 318, 'cf': 408, 'rf': 314},
    'tropicana':         {'lf': 315, 'cf': 404, 'rf': 322},
    'rogers':            {'lf': 328, 'cf': 400, 'rf': 328},
    'camden':            {'lf': 333, 'cf': 410, 'rf': 318},
    'progressive':       {'lf': 325, 'cf': 410, 'rf': 325},
    'guaranteed':        {'lf': 330, 'cf': 400, 'rf': 335},
    'comerica':          {'lf': 345, 'cf': 420, 'rf': 330},
    'kauffman':          {'lf': 330, 'cf': 410, 'rf': 330},
    'target':            {'lf': 339, 'cf': 404, 'rf': 328},
    'daikin':            {'lf': 315, 'cf': 436, 'rf': 326},
    'minute maid':       {'lf': 315, 'cf': 436, 'rf': 326},
    'globe life':        {'lf': 329, 'cf': 407, 'rf': 326},
    'angel':             {'lf': 330, 'cf': 396, 'rf': 330},
    'oakland':           {'lf': 330, 'cf': 400, 'rf': 330},
    'sutter':            {'lf': 330, 'cf': 400, 'rf': 330},
    'ringcentral':       {'lf': 330, 'cf': 400, 'rf': 330},
    't-mobile':          {'lf': 331, 'cf': 401, 'rf': 326},
    'citi':              {'lf': 335, 'cf': 408, 'rf': 330},
    'citizens':          {'lf': 329, 'cf': 401, 'rf': 330},
    'nationals':         {'lf': 336, 'cf': 402, 'rf': 335},
    'trust':             {'lf': 335, 'cf': 400, 'rf': 325},
    'truist':            {'lf': 335, 'cf': 400, 'rf': 325},
    'loandepot':         {'lf': 344, 'cf': 407, 'rf': 335},
    'marlins':           {'lf': 344, 'cf': 407, 'rf': 335},
    'pnc':               {'lf': 325, 'cf': 399, 'rf': 320},
    'great american':    {'lf': 328, 'cf': 404, 'rf': 325},
    'wrigley':           {'lf': 355, 'cf': 400, 'rf': 353},
    'milwaukee':         {'lf': 344, 'cf': 400, 'rf': 345},
    'american family':   {'lf': 344, 'cf': 400, 'rf': 345},
    'busch':             {'lf': 336, 'cf': 400, 'rf': 335},
    'dodger':            {'lf': 330, 'cf': 395, 'rf': 330},
    'oracle':            {'lf': 339, 'cf': 399, 'rf': 309},
    'petco':             {'lf': 336, 'cf': 396, 'rf': 322},
    'chase':             {'lf': 330, 'cf': 407, 'rf': 335},
    'coors':             {'lf': 347, 'cf': 415, 'rf': 350},
}

DEFAULT_DIMS = {'lf': 330, 'cf': 400, 'rf': 330}

# League-mean fence distance per field — used as the denominator in
# distance-interacted HR multipliers (a shorter-than-league fence boosts HR).
LEAGUE_MEAN = {
    'lf': 332.7,
    'cf': 404.6,
    'rf': 328.9,
}


def dims_for_venue_name(name: str | None) -> dict[str, float]:
    if not name:
        return dict(DEFAULT_DIMS)
    lower = name.lower()
    for key, dims in PARK_DIMS.items():
        if key in lower:
            return dict(dims)
    return dict(DEFAULT_DIMS)


def populate_venues_table(engine) -> int:
    """Write dims into the `venues` table. Returns count updated."""
    import pandas as pd
    from sqlalchemy import text
    with engine.begin() as c:
        # Add columns if missing (idempotent).
        for col in ('lf_ft', 'cf_ft', 'rf_ft'):
            c.execute(text(f'ALTER TABLE venues ADD COLUMN IF NOT EXISTS {col} float'))
        venues = pd.read_sql_query('SELECT venue_id, name FROM venues', c)
    updates = []
    for _, r in venues.iterrows():
        dims = dims_for_venue_name(r['name'])
        updates.append({'vid': int(r['venue_id']), **dims})
    if not updates:
        return 0
    from sqlalchemy import text
    with engine.begin() as c:
        for u in updates:
            c.execute(text(
                'UPDATE venues SET lf_ft = :lf, cf_ft = :cf, rf_ft = :rf WHERE venue_id = :vid'
            ), u)
    return len(updates)


if __name__ == '__main__':
    from config import get_engine
    e = get_engine()
    n = populate_venues_table(e)
    print(f'populated dims for {n} venues')
