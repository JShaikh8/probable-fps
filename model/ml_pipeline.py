"""
Two-headed ML layer on top of the factor projections.

Head A — factor-weight tuner (linear regression).
    target  = actual DK pts - baseline DK pts
    features = the 7 factor signals and their products with baseline
    output  = `tuned_dk_pts` per projection

Head B — LightGBM predictor (gradient-boosted trees).
    target  = actual DK pts per hitter-game
    features = baseline, factor signals, expected PA, slot, park factors, weather,
               handedness, pitcher FIP
    output  = `ml_dk_pts` per projection

Both run as subcommands:
    python -m model.ml_pipeline build      # build training set from historical at-bats
    python -m model.ml_pipeline train      # fit both models, save to artifacts/
    python -m model.ml_pipeline predict    # write tuned + ml columns into today's projections
    python -m model.ml_pipeline all        # build → train → predict
"""
from __future__ import annotations

import argparse
import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sqlalchemy import text

from config import get_engine, get_session
from db.models import Projection


ARTIFACTS = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
os.makedirs(ARTIFACTS, exist_ok=True)

FACTOR_KEYS = [
    'matchup', 'platoon', 'stuffQuality', 'recentForm',
    'park', 'battingOrder', 'weather',
]

NUMERIC_FEATURES = [
    'baseline_dk_pts',
    'expected_pa',
    'lineup_slot',
    'factor_score',
    'hr_factor',
    'hit_factor',
    'hard_hit_factor',
    'k_factor',
    'temp_f',
    'wind_speed_mph',
    'pitcher_fip',
] + [f'f_{k}' for k in FACTOR_KEYS]


# ═══════════════════════════════════════════════════════════════════
# BUILD — derive (features, target) rows from historical at-bats
# ═══════════════════════════════════════════════════════════════════

def build_training_set() -> pd.DataFrame:
    """
    For every hitter-game in at_bats, compute actual DK pts + join the
    same features today's projection engine uses. Returns a DataFrame.
    """
    engine = get_engine()
    print('Aggregating actual DK pts per hitter-game…')
    actuals = pd.read_sql_query(
        """
        WITH per_ab AS (
          SELECT
            ab.hitter_id, ab.game_pk, ab.game_date, ab.season,
            ab.pitcher_id, ab.hitter_side, ab.pitcher_hand,
            CASE ab.event_type
              WHEN 'single'    THEN 3
              WHEN 'double'    THEN 5
              WHEN 'triple'    THEN 8
              WHEN 'home_run'  THEN 10
              WHEN 'walk'      THEN 2
              WHEN 'intent_walk' THEN 2
              WHEN 'hit_by_pitch' THEN 2
              WHEN 'strikeout' THEN -0.5
              WHEN 'strikeout_double_play' THEN -0.5
              ELSE 0
            END AS pts,
            COALESCE(ab.rbi, 0) * 2 AS rbi_pts
          FROM at_bats ab
        )
        SELECT
          hitter_id, game_pk, game_date, season,
          MAX(pitcher_id) AS pitcher_id,   -- most frequent starter; good enough
          MAX(hitter_side) AS hitter_side,
          MAX(pitcher_hand) AS pitcher_hand,
          COUNT(*) AS pa,
          SUM(pts + rbi_pts) AS actual_dk_pts
        FROM per_ab
        GROUP BY hitter_id, game_pk, game_date, season
        HAVING COUNT(*) >= 3
        """,
        engine,
    )
    print(f'  {len(actuals):,} hitter-game rows')

    print('Joining game context (venue, weather)…')
    games_ctx = pd.read_sql_query(
        """
        SELECT g.game_pk, g.venue_id, g.weather,
               pf.hr_factor, pf.hit_factor, pf.hard_hit_factor, pf.k_factor
        FROM games g
        LEFT JOIN park_factors pf ON pf.venue_id = g.venue_id
        """,
        engine,
    )
    df = actuals.merge(games_ctx, on='game_pk', how='left')

    # Weather JSONB → temp, wind
    def _w(r, k):
        w = r.get('weather') or {}
        v = w.get(k)
        try:
            return float(v) if v is not None else np.nan
        except (TypeError, ValueError):
            return np.nan
    df['temp_f'] = df.apply(lambda r: _w(r, 'tempF'), axis=1)
    df['wind_speed_mph'] = df.apply(lambda r: _w(r, 'windSpeedMph'), axis=1)

    print('Joining pitcher FIP…')
    fip = pd.read_sql_query(
        """
        SELECT DISTINCT ON (pitcher_id, season) pitcher_id, season, fip
        FROM pitcher_season_stats
        ORDER BY pitcher_id, season, fip
        """,
        engine,
    )
    df = df.merge(fip, on=['pitcher_id', 'season'], how='left')

    df.rename(columns={'fip': 'pitcher_fip'}, inplace=True)
    df['pitcher_fip'] = df['pitcher_fip'].fillna(4.20)

    # Fill park factors with 1.0 defaults
    for c in ['hr_factor', 'hit_factor', 'hard_hit_factor', 'k_factor']:
        df[c] = df[c].fillna(1.0)

    print('Joining hitter prior-season stats (identity features)…')
    hitter_stats = pd.read_sql_query(
        """
        SELECT hitter_id, season,
               COUNT(*) AS pa,
               SUM(CASE WHEN event_type IN ('single','double','triple','home_run') THEN 1 ELSE 0 END)::float
                 / NULLIF(SUM(CASE WHEN event_type NOT IN ('walk','intent_walk','hit_by_pitch','sac_fly','sac_bunt') THEN 1 ELSE 0 END), 0) AS avg_,
               SUM(CASE event_type WHEN 'single' THEN 1 WHEN 'double' THEN 2 WHEN 'triple' THEN 3 WHEN 'home_run' THEN 4 ELSE 0 END)::float
                 / NULLIF(SUM(CASE WHEN event_type NOT IN ('walk','intent_walk','hit_by_pitch','sac_fly','sac_bunt') THEN 1 ELSE 0 END), 0) AS slg_,
               SUM(CASE WHEN event_type = 'home_run' THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS hr_rate,
               SUM(CASE WHEN event_type IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS k_rate,
               SUM(CASE WHEN event_type IN ('walk','intent_walk') THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS bb_rate
        FROM at_bats
        GROUP BY hitter_id, season
        HAVING COUNT(*) >= 30
        """,
        engine,
    )
    # Shift season +1 → this is the prior-season baseline for (hitter_id, season)
    hitter_stats['season_apply'] = hitter_stats['season'] + 1
    hitter_stats = hitter_stats.drop(columns=['season']).rename(columns={
        'season_apply': 'season',
        'avg_': 'h_prior_avg',
        'slg_': 'h_prior_slg',
        'hr_rate': 'h_prior_hr_rate',
        'k_rate': 'h_prior_k_rate',
        'bb_rate': 'h_prior_bb_rate',
        'pa': 'h_prior_pa',
    })
    df = df.merge(hitter_stats, on=['hitter_id', 'season'], how='left')

    # League means for cold-start hitters
    for col, default in [
        ('h_prior_avg', 0.248),
        ('h_prior_slg', 0.405),
        ('h_prior_hr_rate', 0.030),
        ('h_prior_k_rate', 0.22),
        ('h_prior_bb_rate', 0.085),
        ('h_prior_pa', 300),
    ]:
        df[col] = df[col].fillna(default)

    print(f'  merged: {len(df):,} rows × {df.shape[1]} cols')
    return df


# ═══════════════════════════════════════════════════════════════════
# TRAIN — fit both models
# ═══════════════════════════════════════════════════════════════════

def train(df: pd.DataFrame):
    """
    Temporal split: train on all seasons except the latest; validate on latest.
    Save Ridge (for baseline correction) and LightGBM to artifacts/.
    """
    latest = int(df['season'].max())
    train_df = df[df['season'] < latest].copy()
    test_df  = df[df['season'] == latest].copy()

    if len(train_df) < 500:
        print(f'  Not enough training data ({len(train_df)} rows). Skipping ML training.')
        return

    print(f'  train: {len(train_df):,} rows (seasons < {latest})')
    print(f'  test:  {len(test_df):,} rows (season {latest})')

    # The feature set for training uses a simple proxy for baseline_dk_pts and
    # factor signals since we don't have point-in-time projection records for
    # historical games. For LightGBM we use the game-context features only;
    # the factor tuner trains against reconciliation data (projection_actuals),
    # which is real post-hoc feedback.

    base_features = [c for c in [
        'pa', 'pitcher_fip',
        'hr_factor', 'hit_factor', 'hard_hit_factor', 'k_factor',
        'temp_f', 'wind_speed_mph',
        'h_prior_avg', 'h_prior_slg', 'h_prior_hr_rate',
        'h_prior_k_rate', 'h_prior_bb_rate', 'h_prior_pa',
    ] if c in df.columns]

    def _prep(d):
        X = d[base_features].copy()
        X['temp_f'] = X['temp_f'].fillna(72)
        X['wind_speed_mph'] = X['wind_speed_mph'].fillna(0)
        # Handedness as one-hot
        X['bats_L'] = (d['hitter_side'] == 'L').astype(int)
        X['bats_R'] = (d['hitter_side'] == 'R').astype(int)
        X['throws_L'] = (d['pitcher_hand'] == 'L').astype(int)
        X['throws_R'] = (d['pitcher_hand'] == 'R').astype(int)
        X['platoon_opp'] = (
            ((d['hitter_side'] == 'L') & (d['pitcher_hand'] == 'R')) |
            ((d['hitter_side'] == 'R') & (d['pitcher_hand'] == 'L'))
        ).astype(int)
        return X.astype(float)

    X_tr, y_tr = _prep(train_df), train_df['actual_dk_pts'].values
    X_te, y_te = _prep(test_df),  test_df['actual_dk_pts'].values

    print('\n  — HistGradientBoosting —')
    gb_model = HistGradientBoostingRegressor(
        loss='absolute_error',        # MAE objective
        max_iter=400,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42,
    )
    gb_model.fit(X_tr, y_tr)

    pred_te = gb_model.predict(X_te)
    mae = mean_absolute_error(y_te, pred_te)
    baseline_mae = mean_absolute_error(y_te, [y_tr.mean()] * len(y_te))
    print(f'    MAE test: {mae:.3f}  (baseline mean={baseline_mae:.3f})')
    print(f'    improvement: {(1 - mae / baseline_mae) * 100:.1f}% over naive mean')

    joblib.dump(
        {'model': gb_model, 'features': list(X_tr.columns), 'mae_test': mae},
        os.path.join(ARTIFACTS, 'gb_hitter.pkl'),
    )
    print(f'    → saved to artifacts/gb_hitter.pkl')

    # ── Factor-weight tuner (Ridge on residual) ────────────────────
    # Uses reconciliation data: actual_dk_pts - proj_dk_pts = α + Σ β_k × signal_k
    print('\n  — Factor-weight tuner —')
    engine = get_engine()
    rec = pd.read_sql_query(
        """
        SELECT pa.proj_dk_pts, pa.actual_dk_pts, p.factors, p.factor_score
        FROM projection_actuals pa
        JOIN projections p
          ON p.hitter_id = pa.hitter_id AND p.game_pk = pa.game_pk
        WHERE pa.actual_dk_pts IS NOT NULL
        """,
        engine,
    )
    if len(rec) < 100:
        print(f'    Only {len(rec)} reconciled rows; skipping tuner (need 100+). '
              f'Will train automatically once more data lands.')
    else:
        factor_cols = [f'f_{k}' for k in FACTOR_KEYS]
        for k, col in zip(FACTOR_KEYS, factor_cols):
            rec[col] = rec['factors'].apply(lambda d: (d or {}).get(k, 0) or 0)
        X_f = rec[factor_cols + ['factor_score']].astype(float).values
        y_f = (rec['actual_dk_pts'] - rec['proj_dk_pts']).values
        ridge = Ridge(alpha=1.0).fit(X_f, y_f)
        print(f'    trained on {len(rec)} reconciled rows')
        for k, β in zip(factor_cols + ['factor_score'], ridge.coef_):
            print(f'      {k:<20} {β:+.3f}')
        joblib.dump(
            {'model': ridge, 'features': factor_cols + ['factor_score']},
            os.path.join(ARTIFACTS, 'factor_tuner.pkl'),
        )
        print('    → saved to artifacts/factor_tuner.pkl')


# ═══════════════════════════════════════════════════════════════════
# PREDICT — apply both models to today's projection rows
# ═══════════════════════════════════════════════════════════════════

def predict(game_date: str | None = None):
    if game_date is None:
        game_date = date.today().isoformat()

    engine = get_engine()
    projections = pd.read_sql_query(
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
        SELECT p.hitter_id, p.game_pk, p.pitcher_id, p.dk_pts, p.baseline_dk_pts,
               p.factors, p.factor_score, p.expected_pa, p.lineup_slot,
               p.hitter_hand, p.pitcher_hand,
               g.season, pf.hr_factor, pf.hit_factor, pf.hard_hit_factor,
               pf.k_factor, g.weather,
               pss.fip AS pitcher_fip,
               hp.h_prior_avg, hp.h_prior_slg, hp.h_prior_hr_rate,
               hp.h_prior_k_rate, hp.h_prior_bb_rate, hp.h_prior_pa
        FROM projections p
        JOIN games g ON g.game_pk = p.game_pk
        LEFT JOIN park_factors pf ON pf.venue_id = g.venue_id
        LEFT JOIN hitter_prior hp ON hp.hitter_id = p.hitter_id AND hp.season = g.season
        LEFT JOIN LATERAL (
            SELECT fip FROM pitcher_season_stats ps
            WHERE ps.pitcher_id = p.pitcher_id
            ORDER BY ps.season DESC LIMIT 1
        ) pss ON TRUE
        WHERE p.game_date = '{game_date}'
        """,
        engine,
    )
    if projections.empty:
        print(f'  No projections for {game_date}')
        return

    print(f'  applying ML to {len(projections)} projections…')

    # Factor tuner
    tuner_path = os.path.join(ARTIFACTS, 'factor_tuner.pkl')
    if os.path.exists(tuner_path):
        bundle = joblib.load(tuner_path)
        factor_cols = [f'f_{k}' for k in FACTOR_KEYS]
        for k, col in zip(FACTOR_KEYS, factor_cols):
            projections[col] = projections['factors'].apply(lambda d: (d or {}).get(k, 0) or 0)
        X = projections[factor_cols + ['factor_score']].astype(float).values
        correction = bundle['model'].predict(X)
        projections['tuned_dk_pts'] = projections['dk_pts'] + correction
    else:
        projections['tuned_dk_pts'] = None

    # Gradient boosting
    gb_path = os.path.join(ARTIFACTS, 'gb_hitter.pkl')
    if os.path.exists(gb_path):
        bundle = joblib.load(gb_path)
        model = bundle['model']
        feats = bundle['features']

        def _w(r, k):
            w = r.get('weather') or {}
            v = w.get(k)
            try:
                return float(v) if v is not None else np.nan
            except (TypeError, ValueError):
                return np.nan
        projections['temp_f'] = projections.apply(lambda r: _w(r, 'tempF'), axis=1).fillna(72)
        projections['wind_speed_mph'] = projections.apply(lambda r: _w(r, 'windSpeedMph'), axis=1).fillna(0)
        projections['pitcher_fip'] = projections['pitcher_fip'].fillna(4.20)
        for c in ['hr_factor', 'hit_factor', 'hard_hit_factor', 'k_factor']:
            projections[c] = projections[c].fillna(1.0)
        projections['pa'] = projections['expected_pa'].fillna(4.0)
        for col, default in [
            ('h_prior_avg', 0.248),
            ('h_prior_slg', 0.405),
            ('h_prior_hr_rate', 0.030),
            ('h_prior_k_rate', 0.22),
            ('h_prior_bb_rate', 0.085),
            ('h_prior_pa', 300),
        ]:
            projections[col] = projections[col].fillna(default)
        projections['bats_L'] = (projections['hitter_hand'] == 'L').astype(int)
        projections['bats_R'] = (projections['hitter_hand'] == 'R').astype(int)
        projections['throws_L'] = (projections['pitcher_hand'] == 'L').astype(int)
        projections['throws_R'] = (projections['pitcher_hand'] == 'R').astype(int)
        projections['platoon_opp'] = (
            ((projections['hitter_hand'] == 'L') & (projections['pitcher_hand'] == 'R')) |
            ((projections['hitter_hand'] == 'R') & (projections['pitcher_hand'] == 'L'))
        ).astype(int)

        # Ensure feature order matches training
        for f in feats:
            if f not in projections.columns:
                projections[f] = 0.0
        X = projections[feats].astype(float).values
        projections['ml_dk_pts'] = model.predict(X)
        projections['ml_delta']  = projections['ml_dk_pts'] - projections['dk_pts']
    else:
        projections['ml_dk_pts'] = None
        projections['ml_delta']  = None

    # Write back to DB (this module owns tuned; ml_matchup owns ml + blend).
    session = get_session()
    try:
        for _, row in projections.iterrows():
            session.execute(
                text("""
                UPDATE projections
                SET tuned_dk_pts = :tuned,
                    blend_dk_pts = CASE
                      WHEN ml_dk_pts IS NOT NULL AND :tuned IS NOT NULL
                        THEN (:tuned + ml_dk_pts) / 2.0
                      WHEN :tuned IS NOT NULL
                        THEN :tuned
                      ELSE blend_dk_pts
                    END
                WHERE hitter_id = :hid AND game_pk = :gpk
                """),
                {
                    'tuned': float(row['tuned_dk_pts']) if pd.notna(row.get('tuned_dk_pts')) else None,
                    'hid':   int(row['hitter_id']),
                    'gpk':   int(row['game_pk']),
                },
            )
        session.commit()
    finally:
        session.close()

    tuned_n = projections['tuned_dk_pts'].notna().sum()
    ml_n    = projections['ml_dk_pts'].notna().sum()
    print(f'  wrote: tuned={tuned_n}  ml={ml_n}')


# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['build', 'train', 'predict', 'all'])
    parser.add_argument('--date', type=str, default=None)
    args = parser.parse_args()

    if args.command in ('build', 'train', 'all'):
        df = build_training_set()
        if args.command in ('train', 'all'):
            train(df)

    if args.command in ('predict', 'all'):
        predict(args.date)


if __name__ == '__main__':
    main()
