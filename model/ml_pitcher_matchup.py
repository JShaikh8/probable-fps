"""
Per-AB matchup classifier for pitchers — mirror of the hitter classifier.

Target classes from the pitcher's perspective (DK pitcher scoring relevant):
    strikeout   → +2 DK pts
    walk        → -0.6 (BB allowed)
    hit         → -0.6 (H allowed, but home_run adds ER separately)
    home_run    → hit + ER bomb
    out         → neutral

Per-start aggregation:
    expected_pa = features come from how many batters the pitcher typically faces
                 → derived from pitcher_season_stats.avg_ip × 4.3
    E[K]   = Σ_i P(K|i)     per AB
    E[BB]  = Σ_i P(BB|i)    per AB
    E[H]   = Σ_i P(H|i)     per AB
    E[HR]  = Σ_i P(HR|i)    per AB
    E[IP]  ≈ avg_ip from history

    DK pitcher: IP×2.25 + K×2 + ER×-2 + H×-0.6 + BB×-0.6
    FD pitcher: outs×1 + K×3 + ER×-3 + H×-0.6 + BB×-0.6

ER is hardest — approximated as HR × 1.3 (solo shots plus some extra runs).

Subcommands:
    python -m model.ml_pitcher_matchup train
    python -m model.ml_pitcher_matchup predict [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sqlalchemy import text

from config import get_engine, get_session


ARTIFACTS = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
os.makedirs(ARTIFACTS, exist_ok=True)


def _canonicalize(evt: str | None) -> str:
    """Collapse event_type into pitcher-relevant classes."""
    if evt is None:
        return 'out'
    if evt in ('strikeout', 'strikeout_double_play'):
        return 'strikeout'
    if evt in ('walk', 'intent_walk'):
        return 'walk'
    if evt == 'home_run':
        return 'home_run'
    if evt in ('single', 'double', 'triple'):
        return 'hit'
    if evt == 'hit_by_pitch':
        return 'walk'   # same neg DK pts
    return 'out'


CLASSES = ['strikeout', 'walk', 'hit', 'home_run', 'out']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# DK pitcher scoring contribution per AB outcome (pitcher's side)
# IP is amortized separately; we add per-AB effects for K/BB/H/HR:
DK_AB: dict[str, float] = {
    'strikeout':  2.0,
    'walk':      -0.6,
    'hit':       -0.6,
    'home_run':  -0.6 + (-2.0 * 1.3),   # hit + ~1.3 ER per HR
    'out':        0.0,
}
FD_AB: dict[str, float] = {
    'strikeout':  3.0,
    'walk':      -0.6,
    'hit':       -0.6,
    'home_run':  -0.6 + (-3.0 * 1.3),
    'out':        0.0,
}

DK_VEC = np.array([DK_AB[c] for c in CLASSES])
FD_VEC = np.array([FD_AB[c] for c in CLASSES])


FEATURES = [
    'h_prior_avg', 'h_prior_slg', 'h_prior_hr_rate',
    'h_prior_k_rate', 'h_prior_bb_rate', 'h_prior_pa',
    'p_prior_k_pct', 'p_prior_bb_pct', 'p_prior_hr_per9', 'p_prior_fip',
    'p_prior_games_started',
    'hr_factor', 'hit_factor', 'k_factor',
    'temp_f', 'wind_speed_mph',
    'bats_L', 'bats_R', 'throws_L', 'throws_R', 'platoon_opp',
    'pk_fastball_usage', 'pk_primary_velo',
]


# ═══════════════════════════════════════════════════════════════════

def build_training_frame() -> pd.DataFrame:
    engine = get_engine()

    print('Loading at-bats…')
    abs_ = pd.read_sql_query(
        """
        SELECT ab.id, ab.hitter_id, ab.pitcher_id, ab.game_pk, ab.season,
               ab.hitter_side, ab.pitcher_hand, ab.event_type
        FROM at_bats ab
        WHERE ab.pitcher_id IS NOT NULL AND ab.hitter_id IS NOT NULL
        """,
        engine,
    )
    abs_['target'] = abs_['event_type'].apply(_canonicalize)
    abs_['y'] = abs_['target'].map(CLASS_TO_IDX)

    # Reuse the feature-build SQL from the hitter classifier
    from model.ml_matchup import build_training_frame as _hitter_build
    print('(reusing hitter-classifier feature build for common features)…')
    hf = _hitter_build()
    # Keep only id + features; drop the hitter-class target so we can re-label
    feat_cols = ['id'] + FEATURES
    hf = hf[feat_cols]
    df = abs_.merge(hf, on='id', how='inner')
    print(f'  merged frame: {len(df):,} × {df.shape[1]}')
    return df


def train():
    df = build_training_frame()
    latest = int(df['season'].max())
    train_df = df[df['season'] < latest]
    test_df  = df[df['season'] == latest]
    print(f'  train: {len(train_df):,}  test: {len(test_df):,} ({latest} holdout)')

    X_tr = train_df[FEATURES].astype(float).values
    y_tr = train_df['y'].astype(int).values
    X_te = test_df[FEATURES].astype(float).values
    y_te = test_df['y'].astype(int).values

    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=6, min_samples_leaf=50,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        random_state=42,
    )
    print('  fitting…')
    model.fit(X_tr, y_tr)

    probs = model.predict_proba(X_te)
    acc = accuracy_score(y_te, probs.argmax(axis=1))
    ll = log_loss(y_te, probs, labels=list(range(len(CLASSES))))
    base_freq = np.bincount(y_tr, minlength=len(CLASSES)) / len(y_tr)
    ll_base = log_loss(y_te, np.tile(base_freq, (len(y_te), 1)),
                       labels=list(range(len(CLASSES))))
    print(f'\n  accuracy: {acc:.3f}')
    print(f'  log-loss: {ll:.4f}  (baseline {ll_base:.4f}, Δ {ll_base - ll:+.4f})')

    joblib.dump({'model': model, 'features': FEATURES, 'classes': CLASSES},
                os.path.join(ARTIFACTS, 'pitcher_matchup_clf.pkl'))
    print('    → saved artifacts/pitcher_matchup_clf.pkl')


# ═══════════════════════════════════════════════════════════════════

def predict(game_date: str | None = None):
    if game_date is None:
        game_date = date.today().isoformat()

    path = os.path.join(ARTIFACTS, 'pitcher_matchup_clf.pkl')
    if not os.path.exists(path):
        print(f'  No pitcher classifier at {path}. Run `train` first.')
        return
    bundle = joblib.load(path)
    model = bundle['model']
    features = bundle['features']

    engine = get_engine()
    # For each pitcher projected today, pull the HITTERS they're facing (the
    # opposing lineup from the projections table). Aggregate per-AB outcome
    # probabilities across that lineup, weighted by each hitter's expected PA.
    df = pd.read_sql_query(
        f"""
        WITH hitter_prior AS (
          SELECT hitter_id, season + 1 AS season,
                 COUNT(*) AS h_prior_pa,
                 SUM(CASE WHEN event_type IN ('single','double','triple','home_run') THEN 1 ELSE 0 END)::float
                   / NULLIF(SUM(CASE WHEN event_type NOT IN ('walk','intent_walk','hit_by_pitch','sac_fly','sac_bunt') THEN 1 ELSE 0 END), 0) AS h_prior_avg,
                 SUM(CASE event_type WHEN 'single' THEN 1 WHEN 'double' THEN 2 WHEN 'triple' THEN 3 WHEN 'home_run' THEN 4 ELSE 0 END)::float
                   / NULLIF(SUM(CASE WHEN event_type NOT IN ('walk','intent_walk','hit_by_pitch','sac_fly','sac_bunt') THEN 1 ELSE 0 END), 0) AS h_prior_slg,
                 SUM(CASE WHEN event_type = 'home_run' THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS h_prior_hr_rate,
                 SUM(CASE WHEN event_type IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS h_prior_k_rate,
                 SUM(CASE WHEN event_type IN ('walk','intent_walk') THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS h_prior_bb_rate
          FROM at_bats
          GROUP BY hitter_id, season
          HAVING COUNT(*) >= 30
        )
        SELECT p.hitter_id, p.pitcher_id, p.game_pk, p.expected_pa,
               p.hitter_hand, p.pitcher_hand, g.season, g.weather,
               pf.hr_factor, pf.hit_factor, pf.k_factor,
               pss.fip AS p_prior_fip, pss.games_started AS p_prior_games_started,
               pss.avg_k, pss.avg_bb, pss.avg_hr, pss.avg_ip,
               pp.arsenal,
               hp.h_prior_pa, hp.h_prior_avg, hp.h_prior_slg,
               hp.h_prior_hr_rate, hp.h_prior_k_rate, hp.h_prior_bb_rate
        FROM projections p
        JOIN games g ON g.game_pk = p.game_pk
        LEFT JOIN park_factors pf ON pf.venue_id = g.venue_id
        LEFT JOIN hitter_prior hp ON hp.hitter_id = p.hitter_id AND hp.season = g.season
        LEFT JOIN LATERAL (
          SELECT * FROM pitcher_season_stats WHERE pitcher_id = p.pitcher_id
          ORDER BY season DESC LIMIT 1
        ) pss ON TRUE
        LEFT JOIN LATERAL (
          SELECT arsenal FROM pitcher_profiles WHERE pitcher_id = p.pitcher_id
          ORDER BY season DESC LIMIT 1
        ) pp ON TRUE
        WHERE p.game_date = '{game_date}'
        """,
        engine,
    )
    if df.empty:
        print(f'  No projections for {game_date}')
        return

    # Same feature extraction as hitter classifier
    def _w(r, k):
        w = r.get('weather') or {}
        v = w.get(k)
        try:
            return float(v) if v is not None else np.nan
        except (TypeError, ValueError):
            return np.nan

    def _arsenal(arsenal: dict | None) -> pd.Series:
        if not arsenal:
            return pd.Series({'pk_fastball_usage': np.nan, 'pk_primary_velo': np.nan})
        fb = (arsenal.get('fastball') or {}).get('usagePct')
        primary = max(arsenal.items(), key=lambda kv: (kv[1] or {}).get('usagePct', 0))
        return pd.Series({
            'pk_fastball_usage': fb,
            'pk_primary_velo': (primary[1] or {}).get('avgSpeed'),
        })

    df['temp_f'] = df.apply(lambda r: _w(r, 'tempF'), axis=1)
    df['wind_speed_mph'] = df.apply(lambda r: _w(r, 'windSpeedMph'), axis=1)
    ars = df['arsenal'].apply(_arsenal)
    df = pd.concat([df, ars], axis=1)
    df['p_prior_k_pct']  = df['avg_k']  / df['avg_ip'].replace(0, np.nan) / 4.3
    df['p_prior_bb_pct'] = df['avg_bb'] / df['avg_ip'].replace(0, np.nan) / 4.3
    df['p_prior_hr_per9'] = df['avg_hr'] / df['avg_ip'].replace(0, np.nan) * 9

    df['bats_L']      = (df['hitter_hand'] == 'L').astype(int)
    df['bats_R']      = (df['hitter_hand'] == 'R').astype(int)
    df['throws_L']    = (df['pitcher_hand'] == 'L').astype(int)
    df['throws_R']    = (df['pitcher_hand'] == 'R').astype(int)
    df['platoon_opp'] = (
        ((df['hitter_hand'] == 'L') & (df['pitcher_hand'] == 'R')) |
        ((df['hitter_hand'] == 'R') & (df['pitcher_hand'] == 'L'))
    ).astype(int)

    defaults = {
        'h_prior_avg': 0.248, 'h_prior_slg': 0.405,
        'h_prior_hr_rate': 0.030, 'h_prior_k_rate': 0.22, 'h_prior_bb_rate': 0.085,
        'h_prior_pa': 300,
        'p_prior_k_pct': 0.22, 'p_prior_bb_pct': 0.085, 'p_prior_hr_per9': 1.1,
        'p_prior_fip': 4.20, 'p_prior_games_started': 10,
        'pk_fastball_usage': 0.55, 'pk_primary_velo': 93.0,
        'hr_factor': 1.0, 'hit_factor': 1.0, 'k_factor': 1.0,
        'temp_f': 72, 'wind_speed_mph': 0,
    }
    for c, v in defaults.items():
        df[c] = df[c].fillna(v)

    X = df[features].astype(float).values
    probs = model.predict_proba(X)
    df['dk_per_ab']  = (probs * DK_VEC).sum(axis=1)
    df['fd_per_ab']  = (probs * FD_VEC).sum(axis=1)
    df['_expected_pa'] = df['expected_pa'].fillna(4.0)

    # Aggregate per pitcher: sum across all hitters they'll face, weighted by PA.
    pitcher_ml = df.groupby(['pitcher_id', 'game_pk']).apply(
        lambda g: pd.Series({
            'dk_from_abs': (g['dk_per_ab'] * g['_expected_pa']).sum(),
            'fd_from_abs': (g['fd_per_ab'] * g['_expected_pa']).sum(),
        })
    ).reset_index()

    # Load base pitcher projections for avg_ip + factor dk_pts
    pp = pd.read_sql_query(
        f"""
        SELECT pp.pitcher_id, pp.game_pk, pp.dk_pts, pp.proj
        FROM pitcher_projections pp
        WHERE pp.game_date = '{game_date}'
        """,
        engine,
    )
    pitcher_ml = pitcher_ml.merge(pp, on=['pitcher_id', 'game_pk'], how='left')

    # IP contribution: DK pitcher gets 2.25/IP, FD gets 1/out. Use projected IP from factor.
    pitcher_ml['avg_ip'] = pitcher_ml['proj'].apply(lambda p: (p or {}).get('ip', 5.5))
    pitcher_ml['ml_dk_pts'] = pitcher_ml['dk_from_abs'] + pitcher_ml['avg_ip'] * 2.25
    pitcher_ml['ml_fd_pts'] = pitcher_ml['fd_from_abs'] + (pitcher_ml['avg_ip'] * 3) * 1.0
    pitcher_ml['ml_delta'] = pitcher_ml['ml_dk_pts'] - pitcher_ml['dk_pts']

    session = get_session()
    try:
        for _, row in pitcher_ml.iterrows():
            session.execute(
                text("""
                UPDATE pitcher_projections
                SET ml_dk_pts = :ml_dk,
                    ml_fd_pts = :ml_fd,
                    ml_delta  = :delta
                WHERE pitcher_id = :pid AND game_pk = :gpk
                """),
                {
                    'ml_dk': float(row['ml_dk_pts']),
                    'ml_fd': float(row['ml_fd_pts']),
                    'delta': float(row['ml_delta']),
                    'pid':   int(row['pitcher_id']),
                    'gpk':   int(row['game_pk']),
                },
            )
        session.commit()
    finally:
        session.close()

    print(f'  wrote {len(pitcher_ml)} pitcher matchup predictions. '
          f'mean DK={pitcher_ml["ml_dk_pts"].mean():.2f}  FD={pitcher_ml["ml_fd_pts"].mean():.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'predict', 'all'])
    parser.add_argument('--date', default=None)
    args = parser.parse_args()
    if args.command in ('train', 'all'):
        train()
    if args.command in ('predict', 'all'):
        predict(args.date)


if __name__ == '__main__':
    main()
