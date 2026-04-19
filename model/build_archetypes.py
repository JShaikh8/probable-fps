"""
Build hitter and pitcher archetypes via cosine similarity on split vectors.

Hitter features per pitch family (6 families × 8 features = 48 dim):
  avg, slg, whiff_pct, hard_hit_pct, high_velo_whiff_pct, high_spin_whiff_pct,
  avg_exit_velo, avg_launch_angle

Pitcher features per pitch family (6 families × 4 features = 24 dim):
  usagePct, avgSpeed, avgSpin, whiffPct

Writes hitter_vectors, hitter_similar, pitcher_vectors, pitcher_similar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from config import get_engine, get_session
from db.models import HitterSimilar, HitterVector, PitcherSimilar, PitcherVector
from db.io import bulk_upsert


PITCH_FAMILIES = ['fastball', 'sinker', 'cutter', 'slider', 'curveball', 'changeup']

HITTER_FEATURES = [
    'avg', 'slg', 'whiff_pct', 'hard_hit_pct',
    'high_velo_whiff_pct', 'high_spin_whiff_pct',
    'avg_exit_velo', 'avg_launch_angle',
]

PITCHER_FEATURES = ['usagePct', 'avgSpeed', 'avgSpin', 'whiffPct']

MIN_PA_PER_FAMILY = 10
MIN_FAMILIES = 3
PITCHER_MIN_PITCHES = 100
TOP_N_SIMILAR = 20

SEASON_WEIGHTS = {2026: 1.0, 2025: 0.90, 2024: 0.75, 2023: 0.60, 2022: 0.45, 2021: 0.30, 2020: 0.20}


def _weighted_agg(df: pd.DataFrame, group_cols: list[str],
                  metric_cols: list[str], weight_col: str, size_col: str) -> pd.DataFrame:
    """Null-safe weighted aggregate per group — nulls don't contaminate numerators."""
    df = df.copy()
    for c in metric_cols:
        has_val = df[c].notna().astype(float)
        df[f'_w_{c}']  = df[c].fillna(0) * df[weight_col] * df[size_col] * has_val
        df[f'_wp_{c}'] = df[weight_col] * df[size_col] * has_val

    agg_d = {'_total': (size_col, 'sum')}
    for c in metric_cols:
        agg_d[f'_w_{c}']  = (f'_w_{c}',  'sum')
        agg_d[f'_wp_{c}'] = (f'_wp_{c}', 'sum')

    out = df.groupby(group_cols).agg(**agg_d).reset_index()
    for c in metric_cols:
        out[c] = out[f'_w_{c}'] / out[f'_wp_{c}'].replace(0, np.nan)
        out.drop(columns=[f'_w_{c}', f'_wp_{c}'], inplace=True)
    return out.rename(columns={'_total': 'total_size'})


def run():
    engine = get_engine()

    # ── HITTER ARCHETYPES ──────────────────────────────────────────────
    print('Loading hitter pitch splits…')
    df = pd.read_sql_query(
        f"""
        SELECT s.hitter_id, p.full_name AS hitter_name, s.pitch_family, s.season,
               s.pa, s.avg, s.slg, s.whiff_pct, s.hard_hit_pct,
               s.high_velo_whiff_pct, s.high_spin_whiff_pct,
               s.avg_exit_velo, s.avg_launch_angle
        FROM hitter_pitch_splits s
        LEFT JOIN players p ON p.player_id = s.hitter_id
        """,
        engine,
    )
    if df.empty:
        print('No splits. Run build_pitch_splits first.')
        return

    print(f'  {df["hitter_id"].nunique()} hitters, {len(df)} rows')
    df['weight'] = df['season'].map(SEASON_WEIGHTS).fillna(0.25)

    agg = _weighted_agg(
        df, group_cols=['hitter_id', 'pitch_family'],
        metric_cols=HITTER_FEATURES, weight_col='weight', size_col='pa',
    )
    agg = agg[agg['total_size'] >= MIN_PA_PER_FAMILY]

    pivot = agg.pivot_table(
        index='hitter_id', columns='pitch_family', values=HITTER_FEATURES,
    )
    pivot.columns = [f'{v}_{f}' for v, f in pivot.columns]
    pivot = pivot.reset_index()

    fam_cov = pivot[[f'avg_{f}' for f in PITCH_FAMILIES if f'avg_{f}' in pivot.columns]].notna().sum(axis=1)
    pivot = pivot[fam_cov >= MIN_FAMILIES].copy()
    print(f'  {len(pivot)} hitters with ≥{MIN_FAMILIES} pitch-family coverage')

    if pivot.empty:
        print('  Not enough data for archetypes. Exiting.')
        return

    feature_cols = [
        f'{feat}_{fam}'
        for fam in PITCH_FAMILIES
        for feat in HITTER_FEATURES
        if f'{feat}_{fam}' in pivot.columns and pivot[f'{feat}_{fam}'].notna().any()
    ]
    if not feature_cols:
        print('  No valid features. Exiting.')
        return

    # League-mean impute
    for c in feature_cols:
        pivot[c] = pivot[c].fillna(pivot[c].mean())

    X = pivot[feature_cols].values.astype(float)
    X_scaled = StandardScaler().fit_transform(X)

    # Attach hitter names for display
    name_map = df.groupby('hitter_id')['hitter_name'].last().to_dict()
    pivot['hitter_name'] = pivot['hitter_id'].map(name_map).fillna('')

    print('Writing hitter_vectors…')
    vector_records = []
    for i, row in pivot.iterrows():
        idx = pivot.index.get_loc(i)
        vector_records.append({
            'hitter_id': int(row['hitter_id']),
            'vector': {c: (float(row[c]) if pd.notna(row[c]) else None) for c in feature_cols},
            'scaled_vector': X_scaled[idx].tolist(),
        })

    print('Computing cosine similarity…')
    sim = cosine_similarity(X_scaled)
    ids = pivot['hitter_id'].tolist()
    names = pivot['hitter_name'].tolist()
    similar_records = []
    for i, h in enumerate(ids):
        top = [j for j in np.argsort(sim[i])[::-1] if j != i][:TOP_N_SIMILAR]
        similar_records.append({
            'hitter_id': int(h),
            'similar_list': [
                {'hitter_id': int(ids[j]), 'hitter_name': names[j],
                 'similarity': round(float(sim[i][j]), 4)}
                for j in top
            ],
        })

    session = get_session()
    try:
        bulk_upsert(session, HitterVector, vector_records, pk_cols=['hitter_id'])
        bulk_upsert(session, HitterSimilar, similar_records, pk_cols=['hitter_id'])
        session.commit()
        print(f'  hitter_vectors: {len(vector_records)}  hitter_similar: {len(similar_records)}')
    finally:
        session.close()

    # ── PITCHER ARCHETYPES ────────────────────────────────────────────
    print('\nLoading pitcher profiles for archetypes…')
    pdf = pd.read_sql_query(
        f"""
        SELECT pp.pitcher_id, p.full_name AS pitcher_name, p.pitch_hand,
               pp.season, pp.total_pitches, pp.arsenal
        FROM pitcher_profiles pp
        LEFT JOIN players p ON p.player_id = pp.pitcher_id
        WHERE pp.total_pitches >= {PITCHER_MIN_PITCHES}
        """,
        engine,
    )
    if pdf.empty:
        print('  No pitcher profiles. Skipping.')
        return

    rows: list[dict] = []
    for _, doc in pdf.iterrows():
        arsenal = doc['arsenal'] or {}
        for fam, info in arsenal.items():
            rows.append({
                'pitcher_id':    int(doc['pitcher_id']),
                'pitcher_name':  doc['pitcher_name'] or '',
                'pitch_hand':    doc['pitch_hand'] or '',
                'season':        int(doc['season']),
                'pitch_family':  fam,
                'total_pitches': int(doc['total_pitches']),
                'usagePct':      info.get('usagePct'),
                'avgSpeed':      info.get('avgSpeed'),
                'avgSpin':       info.get('avgSpin'),
                'whiffPct':      info.get('whiffPct'),
            })

    parsed = pd.DataFrame(rows)
    if parsed.empty:
        print('  No pitcher arsenal data. Skipping.')
        return
    parsed['weight'] = parsed['season'].map(SEASON_WEIGHTS).fillna(0.25)

    pagg = _weighted_agg(
        parsed, group_cols=['pitcher_id', 'pitch_family'],
        metric_cols=PITCHER_FEATURES, weight_col='weight', size_col='total_pitches',
    )
    hand_map = parsed.groupby('pitcher_id')['pitch_hand'].last()

    ppivot = pagg.pivot_table(
        index='pitcher_id', columns='pitch_family', values=PITCHER_FEATURES,
    )
    ppivot.columns = [f'{v}_{f}' for v, f in ppivot.columns]
    ppivot = ppivot.reset_index()
    ppivot['pitch_hand'] = ppivot['pitcher_id'].map(hand_map).fillna('')

    pcov = ppivot[[f'usagePct_{f}' for f in PITCH_FAMILIES if f'usagePct_{f}' in ppivot.columns]].notna().sum(axis=1)
    ppivot = ppivot[pcov >= 2].copy()
    print(f'  {len(ppivot)} pitchers with sufficient coverage')
    if ppivot.empty:
        return

    pfeat_cols = [
        f'{feat}_{fam}'
        for fam in PITCH_FAMILIES
        for feat in PITCHER_FEATURES
        if f'{feat}_{fam}' in ppivot.columns
    ]
    for c in pfeat_cols:
        if ppivot[c].notna().any():
            ppivot[c] = ppivot[c].fillna(ppivot[c].mean())

    Xp = ppivot[pfeat_cols].values.astype(float)
    Xp_scaled = StandardScaler().fit_transform(Xp)
    pname_map = parsed.groupby('pitcher_id')['pitcher_name'].last().to_dict()
    ppivot['pitcher_name'] = ppivot['pitcher_id'].map(pname_map).fillna('')

    pvec_records = []
    for i, row in ppivot.iterrows():
        idx = ppivot.index.get_loc(i)
        pvec_records.append({
            'pitcher_id': int(row['pitcher_id']),
            'vector': {c: (float(row[c]) if pd.notna(row[c]) else None) for c in pfeat_cols},
            'scaled_vector': Xp_scaled[idx].tolist(),
        })

    psim = cosine_similarity(Xp_scaled)
    pids = ppivot['pitcher_id'].tolist()
    pnames = ppivot['pitcher_name'].tolist()
    psim_records = []
    for i, p in enumerate(pids):
        top = [j for j in np.argsort(psim[i])[::-1] if j != i][:TOP_N_SIMILAR]
        psim_records.append({
            'pitcher_id': int(p),
            'similar_list': [
                {'pitcher_id': int(pids[j]), 'pitcher_name': pnames[j],
                 'similarity': round(float(psim[i][j]), 4)}
                for j in top
            ],
        })

    session = get_session()
    try:
        bulk_upsert(session, PitcherVector, pvec_records, pk_cols=['pitcher_id'])
        bulk_upsert(session, PitcherSimilar, psim_records, pk_cols=['pitcher_id'])
        session.commit()
        print(f'  pitcher_vectors: {len(pvec_records)}  pitcher_similar: {len(psim_records)}')
    finally:
        session.close()

    print('\nDone.')


if __name__ == '__main__':
    run()
