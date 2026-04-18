from __future__ import annotations
"""
Build hitter AND pitcher archetypes using cosine similarity on split vectors.

Hitter feature vector — per pitch family (6 families × 8 features = 48 dimensions):
  avg, slg, whiffPct, hardHitPct, highVeloWhiffPct, highSpinWhiffPct,
  avgExitVelo, avgLaunchAngle

  avgExitVelo / avgLaunchAngle per pitch family capture contact quality profile —
  a hitter may have great avg/slg against fastballs but with a flat launch angle
  (groundball tendency) vs a high-launch-angle line-drive hitter.

  highVeloWhiffPct / highSpinWhiffPct capture whether a hitter struggles against
  elite velocity or elite spin.

Pitcher feature vector — per pitch family (6 families × 4 features = 24 dimensions):
  usagePct, avgSpeed, avgSpin, whiffPct

Output collections:
  mlb_hitter_vectors, mlb_hitter_similar
  mlb_pitcher_vectors, mlb_pitcher_similar
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, '..')
from config import get_db
from pymongo import UpdateOne

PITCH_FAMILIES      = ['fastball', 'sinker', 'cutter', 'slider', 'curveball', 'changeup']
HITTER_FEATURES     = ['avg', 'slg', 'whiffPct', 'hardHitPct', 'highVeloWhiffPct', 'highSpinWhiffPct',
                       'avgExitVelo', 'avgLaunchAngle']
PITCHER_FEATURES    = ['usagePct', 'avgSpeed', 'avgSpin', 'whiffPct']
MIN_PA_PER_FAMILY   = 10
MIN_FAMILIES        = 3
PITCHER_MIN_PITCHES = 100
TOP_N_SIMILAR       = 20

SEASON_WEIGHTS = {2025: 1.0, 2024: 0.85, 2023: 0.70, 2022: 0.55, 2021: 0.40, 2020: 0.30}


def _null_safe_weighted_agg(df, group_cols, name_col, metric_cols, weight_col, size_col):
    """
    Weighted average per group.  Nulls do NOT contribute to numerator or
    denominator, so a metric with no data comes back as NaN — not 0.
    This fixes the hardHitPct = 0 bug where pandas .sum(skipna=True) was
    treating NaN numerators as 0 but keeping the full denominator.
    """
    df = df.copy()
    for col in metric_cols:
        has_val = df[col].notna().astype(float)
        df[f'_w_{col}']  = df[col].fillna(0) * df[weight_col] * df[size_col] * has_val
        df[f'_wp_{col}'] = df[weight_col] * df[size_col] * has_val

    agg_dict = {name_col: (name_col, 'last'), '_total': (size_col, 'sum')}
    for col in metric_cols:
        agg_dict[f'_w_{col}']  = (f'_w_{col}',  'sum')
        agg_dict[f'_wp_{col}'] = (f'_wp_{col}', 'sum')

    agg = df.groupby(group_cols).agg(**agg_dict).reset_index()

    for col in metric_cols:
        agg[col] = agg[f'_w_{col}'] / agg[f'_wp_{col}'].replace(0, np.nan)
        agg.drop(columns=[f'_w_{col}', f'_wp_{col}'], inplace=True)

    return agg.rename(columns={'_total': 'total_size'})


def run():
    db = get_db()

    # ── HITTER ARCHETYPES ────────────────────────────────────────────────────
    print('Loading hitter pitch splits...')
    cur = db.mlb_hitter_pitch_splits.find({}, {
        '_id': 0, 'hitterId': 1, 'hitterName': 1, 'pitchFamily': 1, 'season': 1,
        'pa': 1,
        'avg': 1, 'slg': 1, 'whiffPct': 1, 'hardHitPct': 1,
        'highVeloWhiffPct': 1, 'highSpinWhiffPct': 1,
        'avgExitVelo': 1, 'avgLaunchAngle': 1,
    })
    df = pd.DataFrame(list(cur))

    if df.empty:
        print('No splits data. Run build_pitch_splits.py first.')
        return

    print(f'  {df["hitterId"].nunique()} hitters, {len(df)} split rows')

    df['weight'] = df['season'].map(SEASON_WEIGHTS).fillna(0.25)

    agg = _null_safe_weighted_agg(
        df,
        group_cols=['hitterId', 'pitchFamily'],
        name_col='hitterName',
        metric_cols=HITTER_FEATURES,
        weight_col='weight',
        size_col='pa',
    )
    agg = agg[agg['total_size'] >= MIN_PA_PER_FAMILY]

    pivot = agg.pivot_table(
        index=['hitterId', 'hitterName'],
        columns='pitchFamily',
        values=HITTER_FEATURES,
    )
    pivot.columns = [f'{v}_{f}' for v, f in pivot.columns]
    pivot = pivot.reset_index()

    # Require coverage across >= MIN_FAMILIES pitch families
    family_cov = pivot[
        [f'avg_{fam}' for fam in PITCH_FAMILIES if f'avg_{fam}' in pivot.columns]
    ].notna().sum(axis=1)
    pivot = pivot[family_cov >= MIN_FAMILIES].copy()
    print(f'  {len(pivot)} hitters with sufficient pitch-family coverage')

    feature_cols = [
        f'{feat}_{fam}'
        for fam  in PITCH_FAMILIES
        for feat in HITTER_FEATURES
        if f'{feat}_{fam}' in pivot.columns
           and pivot[f'{feat}_{fam}'].notna().any()   # skip features with zero data
    ]

    if not feature_cols:
        print('  No valid feature columns. Exiting.')
        return

    print(f'  Feature dimensions: {len(feature_cols)} ({len(HITTER_FEATURES)} per family × {len(PITCH_FAMILIES)} families, some may be excluded if data unavailable)')

    # Impute missing values with league-mean per feature
    for col in feature_cols:
        pivot[col] = pivot[col].fillna(pivot[col].mean())

    X        = pivot[feature_cols].values.astype(float)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print('Storing hitter vectors...')
    ops = []
    for i, row in pivot.iterrows():
        vec = {col: (float(row[col]) if pd.notna(row[col]) else None) for col in feature_cols}
        ops.append(UpdateOne(
            {'hitterId': int(row['hitterId'])},
            {'$set': {
                'hitterId':     int(row['hitterId']),
                'hitterName':   row['hitterName'],
                'vector':       vec,
                'scaledVector': X_scaled[pivot.index.get_loc(i)].tolist(),
            }},
            upsert=True,
        ))
    db.mlb_hitter_vectors.create_index('hitterId', unique=True)
    if ops:
        db.mlb_hitter_vectors.bulk_write(ops, ordered=False)

    print('Computing hitter cosine similarity...')
    sim   = cosine_similarity(X_scaled)
    ids   = pivot['hitterId'].tolist()
    names = pivot['hitterName'].tolist()
    ops   = []
    for i, h_id in enumerate(ids):
        top_idx = [j for j in np.argsort(sim[i])[::-1] if j != i][:TOP_N_SIMILAR]
        similar = [
            {'hitterId': ids[j], 'hitterName': names[j], 'similarity': round(float(sim[i][j]), 4)}
            for j in top_idx
        ]
        ops.append(UpdateOne(
            {'hitterId': h_id},
            {'$set': {'hitterId': h_id, 'hitterName': names[i], 'similar': similar}},
            upsert=True,
        ))
    db.mlb_hitter_similar.create_index('hitterId', unique=True)
    if ops:
        result = db.mlb_hitter_similar.bulk_write(ops, ordered=False)
        print(f'  hitter similarity: {result.upserted_count} inserted, {result.modified_count} updated')

    # ── PITCHER ARCHETYPES ───────────────────────────────────────────────────
    print('\nLoading pitcher profiles for archetypes...')
    pcur = db.mlb_pitcher_profiles.find(
        {'totalPitches': {'$gte': PITCHER_MIN_PITCHES}},
        {'_id': 0, 'pitcherId': 1, 'pitcherName': 1, 'pitcherHand': 1,
         'season': 1, 'totalPitches': 1, 'arsenal': 1}
    )
    pitcher_docs = list(pcur)

    if not pitcher_docs:
        print('  No pitcher profiles found. Skipping pitcher archetypes.')
        print('\nDone.')
        return

    rows = []
    for doc in pitcher_docs:
        for fam, info in (doc.get('arsenal') or {}).items():
            rows.append({
                'pitcherId':    doc['pitcherId'],
                'pitcherName':  doc.get('pitcherName', ''),
                'pitcherHand':  doc.get('pitcherHand', ''),
                'season':       doc['season'],
                'pitchFamily':  fam,
                'totalPitches': doc['totalPitches'],
                'usagePct':     info.get('usagePct'),
                'avgSpeed':     info.get('avgSpeed'),
                'avgSpin':      info.get('avgSpin'),
                'whiffPct':     info.get('whiffPct'),
            })

    pdf = pd.DataFrame(rows)
    pdf['weight'] = pdf['season'].map(SEASON_WEIGHTS).fillna(0.25)

    pagg = _null_safe_weighted_agg(
        pdf,
        group_cols=['pitcherId', 'pitchFamily'],
        name_col='pitcherName',
        metric_cols=PITCHER_FEATURES,
        weight_col='weight',
        size_col='totalPitches',
    )
    hand_map = pdf.groupby('pitcherId')['pitcherHand'].last()
    pagg['pitcherHand'] = pagg['pitcherId'].map(hand_map)

    ppivot = pagg.pivot_table(
        index=['pitcherId', 'pitcherName', 'pitcherHand'],
        columns='pitchFamily',
        values=PITCHER_FEATURES,
    )
    ppivot.columns = [f'{v}_{f}' for v, f in ppivot.columns]
    ppivot = ppivot.reset_index()

    pcov = ppivot[
        [f'usagePct_{fam}' for fam in PITCH_FAMILIES if f'usagePct_{fam}' in ppivot.columns]
    ].notna().sum(axis=1)
    ppivot = ppivot[pcov >= 2].copy()
    print(f'  {len(ppivot)} pitchers with sufficient pitch-family coverage')

    pfeat_cols = [
        f'{feat}_{fam}'
        for fam  in PITCH_FAMILIES
        for feat in PITCHER_FEATURES
        if f'{feat}_{fam}' in ppivot.columns
    ]
    for col in pfeat_cols:
        if ppivot[col].notna().any():
            ppivot[col] = ppivot[col].fillna(ppivot[col].mean())

    Xp        = ppivot[pfeat_cols].values.astype(float)
    pscaler   = StandardScaler()
    Xp_scaled = pscaler.fit_transform(Xp)

    print('Storing pitcher vectors...')
    ops = []
    for i, row in ppivot.iterrows():
        vec = {col: (float(row[col]) if pd.notna(row[col]) else None) for col in pfeat_cols}
        ops.append(UpdateOne(
            {'pitcherId': int(row['pitcherId'])},
            {'$set': {
                'pitcherId':    int(row['pitcherId']),
                'pitcherName':  row['pitcherName'],
                'pitcherHand':  row['pitcherHand'],
                'vector':       vec,
                'scaledVector': Xp_scaled[ppivot.index.get_loc(i)].tolist(),
            }},
            upsert=True,
        ))
    db.mlb_pitcher_vectors.create_index('pitcherId', unique=True)
    if ops:
        db.mlb_pitcher_vectors.bulk_write(ops, ordered=False)

    print('Computing pitcher cosine similarity...')
    psim   = cosine_similarity(Xp_scaled)
    pids   = ppivot['pitcherId'].tolist()
    pnames = ppivot['pitcherName'].tolist()
    ops    = []
    for i, p_id in enumerate(pids):
        top_idx = [j for j in np.argsort(psim[i])[::-1] if j != i][:TOP_N_SIMILAR]
        similar = [
            {'pitcherId': pids[j], 'pitcherName': pnames[j], 'similarity': round(float(psim[i][j]), 4)}
            for j in top_idx
        ]
        ops.append(UpdateOne(
            {'pitcherId': p_id},
            {'$set': {'pitcherId': p_id, 'pitcherName': pnames[i], 'similar': similar}},
            upsert=True,
        ))
    db.mlb_pitcher_similar.create_index('pitcherId', unique=True)
    if ops:
        result = db.mlb_pitcher_similar.bulk_write(ops, ordered=False)
        print(f'  pitcher similarity: {result.upserted_count} inserted, {result.modified_count} updated')

    print('\nDone.')


if __name__ == '__main__':
    run()
