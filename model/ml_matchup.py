"""
Per-AB matchup classifier — the real ML layer.

Instead of predicting per-game DK pts directly, this trains a multiclass
classifier on every historical at-bat:

    features = (hitter_prior, pitcher_prior, park, weather, handedness/platoon)
    target   = event outcome class ∈ {single, double, triple, home_run,
                                        walk, hit_by_pitch, strikeout, out}

At projection time, for each matchup (hitter × pitcher × game) we get
P(outcome) per AB. Expected DK pts for the hitter-game is:

    E[DK] = expected_PA × Σ_outcome P(outcome) × dk_pts(outcome)

This is how pro DFS shops do it. Generalizes cleanly to pitcher projections
and NRFI (simulate first-inning PAs with the same probabilities).

Subcommands:
    python -m model.ml_matchup train
    python -m model.ml_matchup predict [--date YYYY-MM-DD]
    python -m model.ml_matchup all
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


# DK scoring per outcome
DK_PTS: dict[str, float] = {
    'single':       3.0,
    'double':       5.0,
    'triple':       8.0,
    'home_run':    10.0,
    'walk':         2.0,
    'hit_by_pitch': 2.0,
    'strikeout':   -0.5,
    'out':          0.0,
}

# FanDuel scoring per outcome (no K penalty; HR/1B/2B/3B different values)
FD_PTS: dict[str, float] = {
    'single':       3.0,
    'double':       6.0,
    'triple':       9.0,
    'home_run':    12.0,
    'walk':         3.0,
    'hit_by_pitch': 3.0,
    'strikeout':    0.0,
    'out':          0.0,
}

# Map raw MLBAM event_type → our class set
def _canonicalize(evt: str | None) -> str:
    if evt is None:
        return 'out'
    if evt in ('single', 'double', 'triple', 'home_run',
               'walk', 'hit_by_pitch'):
        return evt
    if evt in ('intent_walk',):
        return 'walk'
    if evt in ('strikeout', 'strikeout_double_play'):
        return 'strikeout'
    return 'out'


CLASSES = ['single', 'double', 'triple', 'home_run',
           'walk', 'hit_by_pitch', 'strikeout', 'out']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DK_VEC = np.array([DK_PTS[c] for c in CLASSES])
FD_VEC = np.array([FD_PTS[c] for c in CLASSES])


# ═══════════════════════════════════════════════════════════════════
# BUILD — features per at-bat
# ═══════════════════════════════════════════════════════════════════

FEATURES = [
    'h_prior_avg', 'h_prior_slg', 'h_prior_hr_rate',
    'h_prior_k_rate', 'h_prior_bb_rate', 'h_prior_pa',
    'p_prior_k_pct', 'p_prior_bb_pct', 'p_prior_hr_per9', 'p_prior_fip',
    'p_prior_games_started',
    'hr_factor', 'hit_factor', 'k_factor',
    'temp_f', 'wind_speed_mph',
    'bats_L', 'bats_R', 'throws_L', 'throws_R', 'platoon_opp',
    'pk_fastball_usage', 'pk_primary_velo',
    'h_avg_exit_velo', 'h_avg_launch_angle', 'h_pull_pct',
    'h_avg_vs_primary', 'h_whiff_vs_primary', 'h_hard_hit_vs_primary',
    'h_form_ratio',
    'lineup_slot',
    # NEW: point-in-time rolling hitter stats (trailing N games before THIS game)
    'h_roll30_avg', 'h_roll30_slg', 'h_roll30_hr_rate',
    'h_roll30_k_rate', 'h_roll30_bb_rate', 'h_roll30_games',
    'h_roll10_avg', 'h_roll10_slg', 'h_roll10_pa',
    # Pitcher rolling (trailing 5 starts)
    'p_roll5_k_rate', 'p_roll5_bb_rate', 'p_roll5_hr_rate',
    'p_roll5_baa', 'p_roll5_starts',
]


def build_training_frame() -> pd.DataFrame:
    engine = get_engine()

    print('Loading at-bats (targets)…')
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
    print(f'  {len(abs_):,} ABs')

    print('Hitter prior-season features…')
    h_prior = pd.read_sql_query(
        """
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
        """,
        engine,
    )
    df = abs_.merge(h_prior, on=['hitter_id', 'season'], how='left')

    print('Pitcher prior-season features…')
    p_prior = pd.read_sql_query(
        """
        SELECT pss.pitcher_id, pss.season + 1 AS season,
               pss.fip AS p_prior_fip,
               pss.games_started AS p_prior_games_started,
               ROUND(CAST(pss.avg_k / NULLIF(pss.avg_ip * 4.3, 0) AS NUMERIC), 4)::float AS p_prior_k_pct,
               ROUND(CAST(pss.avg_bb / NULLIF(pss.avg_ip * 4.3, 0) AS NUMERIC), 4)::float AS p_prior_bb_pct,
               ROUND(CAST(pss.avg_hr / NULLIF(pss.avg_ip, 0) * 9 AS NUMERIC), 3)::float AS p_prior_hr_per9
        FROM pitcher_season_stats pss
        """,
        engine,
    )
    df = df.merge(p_prior, on=['pitcher_id', 'season'], how='left')

    print('Pitcher arsenal primary + velocity…')
    p_arsenal = pd.read_sql_query(
        """
        SELECT pp.pitcher_id, pp.season + 1 AS season, pp.arsenal
        FROM pitcher_profiles pp
        """,
        engine,
    )
    # Extract primary-pitch usage and avg velocity
    def _arsenal_features(arsenal: dict | None) -> pd.Series:
        if not arsenal:
            return pd.Series({'pk_fastball_usage': np.nan, 'pk_primary_velo': np.nan})
        fb_usage = (arsenal.get('fastball') or {}).get('usagePct')
        primary = max(arsenal.items(), key=lambda kv: (kv[1] or {}).get('usagePct', 0))
        primary_velo = (primary[1] or {}).get('avgSpeed')
        return pd.Series({
            'pk_fastball_usage': fb_usage,
            'pk_primary_velo': primary_velo,
        })

    arsenal_ext = p_arsenal['arsenal'].apply(_arsenal_features)
    p_arsenal = pd.concat([p_arsenal[['pitcher_id', 'season']], arsenal_ext], axis=1)
    df = df.merge(p_arsenal, on=['pitcher_id', 'season'], how='left')

    print('Park factors + weather…')
    game_ctx = pd.read_sql_query(
        """
        SELECT g.game_pk, g.weather,
               pf.hr_factor, pf.hit_factor, pf.k_factor
        FROM games g
        LEFT JOIN park_factors pf ON pf.venue_id = g.venue_id
        """,
        engine,
    )
    df = df.merge(game_ctx, on='game_pk', how='left')

    # Weather JSONB → numeric
    def _w(r, k):
        w = r.get('weather') or {}
        v = w.get(k)
        try:
            return float(v) if v is not None else np.nan
        except (TypeError, ValueError):
            return np.nan
    df['temp_f'] = df.apply(lambda r: _w(r, 'tempF'), axis=1)
    df['wind_speed_mph'] = df.apply(lambda r: _w(r, 'windSpeedMph'), axis=1)

    # Handedness features
    df['bats_L']      = (df['hitter_side'] == 'L').astype(int)
    df['bats_R']      = (df['hitter_side'] == 'R').astype(int)
    df['throws_L']    = (df['pitcher_hand'] == 'L').astype(int)
    df['throws_R']    = (df['pitcher_hand'] == 'R').astype(int)
    df['platoon_opp'] = (
        ((df['hitter_side'] == 'L') & (df['pitcher_hand'] == 'R')) |
        ((df['hitter_side'] == 'R') & (df['pitcher_hand'] == 'L'))
    ).astype(int)

    print('Hitter trailing 30 / 10 game rolling stats (point-in-time)…')
    # Per-game hitter totals — then window by (hitter_id, game_date) to get trailing stats.
    roll = pd.read_sql_query(
        """
        WITH per_game AS (
          SELECT hitter_id, game_pk, game_date,
                 COUNT(*) AS pa,
                 SUM(CASE WHEN event_type IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) AS hits,
                 SUM(CASE event_type WHEN 'single' THEN 1 WHEN 'double' THEN 2 WHEN 'triple' THEN 3 WHEN 'home_run' THEN 4 ELSE 0 END) AS tb,
                 SUM(CASE WHEN event_type = 'home_run' THEN 1 ELSE 0 END) AS hr,
                 SUM(CASE WHEN event_type IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS k,
                 SUM(CASE WHEN event_type IN ('walk','intent_walk') THEN 1 ELSE 0 END) AS bb,
                 SUM(CASE WHEN event_type NOT IN ('walk','intent_walk','hit_by_pitch','sac_fly','sac_bunt') THEN 1 ELSE 0 END) AS ab
          FROM at_bats
          GROUP BY hitter_id, game_pk, game_date
        ),
        rolled AS (
          SELECT hitter_id, game_pk,
                 -- Trailing 30-game windows, EXCLUDING current game
                 SUM(hits) OVER w30 - hits AS h30_hits,
                 SUM(ab)   OVER w30 - ab   AS h30_ab,
                 SUM(tb)   OVER w30 - tb   AS h30_tb,
                 SUM(hr)   OVER w30 - hr   AS h30_hr,
                 SUM(k)    OVER w30 - k    AS h30_k,
                 SUM(bb)   OVER w30 - bb   AS h30_bb,
                 SUM(pa)   OVER w30 - pa   AS h30_pa,
                 COUNT(*)  OVER w30 - 1    AS h30_games,
                 SUM(hits) OVER w10 - hits AS h10_hits,
                 SUM(ab)   OVER w10 - ab   AS h10_ab,
                 SUM(tb)   OVER w10 - tb   AS h10_tb,
                 SUM(pa)   OVER w10 - pa   AS h10_pa
          FROM per_game
          WINDOW
            w30 AS (PARTITION BY hitter_id ORDER BY game_date, game_pk
                    ROWS BETWEEN 30 PRECEDING AND CURRENT ROW),
            w10 AS (PARTITION BY hitter_id ORDER BY game_date, game_pk
                    ROWS BETWEEN 10 PRECEDING AND CURRENT ROW)
        )
        SELECT hitter_id, game_pk,
               CASE WHEN h30_ab > 0 THEN h30_hits::float / h30_ab ELSE NULL END AS h_roll30_avg,
               CASE WHEN h30_ab > 0 THEN h30_tb::float   / h30_ab ELSE NULL END AS h_roll30_slg,
               CASE WHEN h30_pa > 0 THEN h30_hr::float   / h30_pa ELSE NULL END AS h_roll30_hr_rate,
               CASE WHEN h30_pa > 0 THEN h30_k::float    / h30_pa ELSE NULL END AS h_roll30_k_rate,
               CASE WHEN h30_pa > 0 THEN h30_bb::float   / h30_pa ELSE NULL END AS h_roll30_bb_rate,
               h30_games::int AS h_roll30_games,
               CASE WHEN h10_ab > 0 THEN h10_hits::float / h10_ab ELSE NULL END AS h_roll10_avg,
               CASE WHEN h10_ab > 0 THEN h10_tb::float   / h10_ab ELSE NULL END AS h_roll10_slg,
               h10_pa::int AS h_roll10_pa
        FROM rolled
        """,
        engine,
    )
    df = df.merge(roll, on=['hitter_id', 'game_pk'], how='left')

    print('Pitcher trailing 5-start rolling stats (point-in-time)…')
    proll = pd.read_sql_query(
        """
        WITH per_start AS (
          SELECT pitcher_id, game_pk, game_date,
                 COUNT(*) AS bf,
                 SUM(CASE WHEN event_type IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) AS h,
                 SUM(CASE WHEN event_type = 'home_run' THEN 1 ELSE 0 END) AS hr,
                 SUM(CASE WHEN event_type IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS k,
                 SUM(CASE WHEN event_type IN ('walk','intent_walk') THEN 1 ELSE 0 END) AS bb,
                 SUM(CASE WHEN event_type NOT IN ('walk','intent_walk','hit_by_pitch','sac_fly','sac_bunt') THEN 1 ELSE 0 END) AS ab
          FROM at_bats
          GROUP BY pitcher_id, game_pk, game_date
          HAVING COUNT(*) >= 12   -- likely a start
        )
        SELECT pitcher_id, game_pk,
               SUM(k)  OVER w - k  AS r5_k,
               SUM(bb) OVER w - bb AS r5_bb,
               SUM(hr) OVER w - hr AS r5_hr,
               SUM(bf) OVER w - bf AS r5_bf,
               SUM(h)  OVER w - h  AS r5_h,
               SUM(ab) OVER w - ab AS r5_ab,
               COUNT(*) OVER w - 1 AS r5_starts
        FROM per_start
        WINDOW w AS (PARTITION BY pitcher_id ORDER BY game_date, game_pk
                     ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)
        """,
        engine,
    )
    proll['p_roll5_k_rate']  = proll['r5_k']  / proll['r5_bf'].where(proll['r5_bf'] > 0)
    proll['p_roll5_bb_rate'] = proll['r5_bb'] / proll['r5_bf'].where(proll['r5_bf'] > 0)
    proll['p_roll5_hr_rate'] = proll['r5_hr'] / proll['r5_bf'].where(proll['r5_bf'] > 0)
    proll['p_roll5_baa']     = proll['r5_h']  / proll['r5_ab'].where(proll['r5_ab'] > 0)
    proll['p_roll5_starts']  = proll['r5_starts']
    proll = proll[['pitcher_id', 'game_pk',
                   'p_roll5_k_rate', 'p_roll5_bb_rate', 'p_roll5_hr_rate',
                   'p_roll5_baa', 'p_roll5_starts']]
    df = df.merge(proll, on=['pitcher_id', 'game_pk'], how='left')

    print('Hitter spray + contact-quality features…')
    spray = pd.read_sql_query(
        """
        SELECT hitter_id,
               avg_exit_velo AS h_avg_exit_velo,
               avg_launch_angle AS h_avg_launch_angle,
               pull_pct AS h_pull_pct
        FROM hitter_spray_profiles
        """,
        engine,
    )
    df = df.merge(spray, on='hitter_id', how='left')

    print('Hitter recent form…')
    form = pd.read_sql_query(
        """
        SELECT hitter_id, form_ratio AS h_form_ratio
        FROM hitter_recent_form
        """,
        engine,
    )
    df = df.merge(form, on='hitter_id', how='left')

    print('Hitter splits vs pitcher primary pitch family…')
    # For each pitcher season, find their primary pitch family (highest usagePct in arsenal)
    primary_per_pitcher = pd.read_sql_query(
        """
        SELECT pp.pitcher_id, pp.season + 1 AS season, pp.primary_pitch AS primary_family
        FROM pitcher_profiles pp
        """,
        engine,
    )
    df = df.merge(primary_per_pitcher, on=['pitcher_id', 'season'], how='left')
    # Then look up the hitter's split vs that family
    splits = pd.read_sql_query(
        """
        SELECT hitter_id, pitch_family AS primary_family, season,
               avg AS h_avg_vs_primary,
               whiff_pct AS h_whiff_vs_primary,
               hard_hit_pct AS h_hard_hit_vs_primary
        FROM hitter_pitch_splits
        """,
        engine,
    )
    df = df.merge(splits, on=['hitter_id', 'primary_family', 'season'], how='left')

    # Lineup slot is in `projections` table only for projected games — at historical
    # at-bat time we don't have it. We leave NaN (HistGB handles it natively).

    # League-mean defaults
    defaults = {
        'h_prior_avg': 0.248, 'h_prior_slg': 0.405,
        'h_prior_hr_rate': 0.030, 'h_prior_k_rate': 0.22, 'h_prior_bb_rate': 0.085,
        'h_prior_pa': 300,
        'p_prior_k_pct': 0.22, 'p_prior_bb_pct': 0.085, 'p_prior_hr_per9': 1.1,
        'p_prior_fip': 4.20, 'p_prior_games_started': 10,
        'pk_fastball_usage': 0.55, 'pk_primary_velo': 93.0,
        'hr_factor': 1.0, 'hit_factor': 1.0, 'k_factor': 1.0,
        'temp_f': 72, 'wind_speed_mph': 0,
        'h_avg_exit_velo': 88.0, 'h_avg_launch_angle': 12.0, 'h_pull_pct': 0.37,
        'h_avg_vs_primary': 0.248, 'h_whiff_vs_primary': 0.24, 'h_hard_hit_vs_primary': 0.35,
        'h_form_ratio': 1.0,
        'lineup_slot': 5,
        'h_roll30_avg': 0.248, 'h_roll30_slg': 0.405, 'h_roll30_hr_rate': 0.030,
        'h_roll30_k_rate': 0.22, 'h_roll30_bb_rate': 0.085, 'h_roll30_games': 0,
        'h_roll10_avg': 0.248, 'h_roll10_slg': 0.405, 'h_roll10_pa': 0,
        'p_roll5_k_rate': 0.22, 'p_roll5_bb_rate': 0.085,
        'p_roll5_hr_rate': 0.028, 'p_roll5_baa': 0.248, 'p_roll5_starts': 0,
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
        else:
            df[c] = df[c].fillna(v)

    df['y'] = df['target'].map(CLASS_TO_IDX)
    print(f'  merged frame: {len(df):,} × {df.shape[1]} cols')
    return df


# ═══════════════════════════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════════════════════════

def train():
    df = build_training_frame()
    latest = int(df['season'].max())
    train_df = df[df['season'] < latest]
    test_df  = df[df['season'] == latest]
    if len(train_df) < 10_000:
        print('  Not enough training ABs.')
        return
    print(f'  train: {len(train_df):,} ABs   test: {len(test_df):,} ABs   ({latest} holdout)')

    X_tr = train_df[FEATURES].astype(float).values
    y_tr = train_df['y'].astype(int).values
    X_te = test_df[FEATURES].astype(float).values
    y_te = test_df['y'].astype(int).values

    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=50,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
    )
    print('  fitting…')
    model.fit(X_tr, y_tr)

    print('\n  — holdout metrics —')
    probs = model.predict_proba(X_te)
    acc = accuracy_score(y_te, probs.argmax(axis=1))

    # Baseline: class frequencies from training
    class_freq = np.bincount(y_tr, minlength=len(CLASSES)) / len(y_tr)
    baseline_probs = np.tile(class_freq, (len(y_te), 1))

    ll_model = log_loss(y_te, probs, labels=list(range(len(CLASSES))))
    ll_base  = log_loss(y_te, baseline_probs, labels=list(range(len(CLASSES))))

    # Expected DK per AB (comparing predicted vs actual realized)
    pred_dk = (probs * DK_VEC).sum(axis=1)
    actual_dk = np.array([DK_PTS[CLASSES[y]] for y in y_te])
    ab_mae = np.abs(pred_dk - actual_dk).mean()

    print(f'    accuracy:      {acc:.3f}')
    print(f'    log-loss:      {ll_model:.4f}  (baseline {ll_base:.4f}, Δ {ll_base - ll_model:+.4f})')
    print(f'    DK pts/AB MAE: {ab_mae:.3f}')
    print(f'    class dist (predicted): ' + ' '.join(
        f'{c}={probs.mean(axis=0)[CLASS_TO_IDX[c]]:.2f}' for c in CLASSES[:4]))
    print(f'    class dist (actual):    ' + ' '.join(
        f'{c}={(y_te == CLASS_TO_IDX[c]).mean():.2f}' for c in CLASSES[:4]))

    joblib.dump({'model': model, 'features': FEATURES, 'classes': CLASSES},
                os.path.join(ARTIFACTS, 'matchup_clf.pkl'))
    print('    → saved artifacts/matchup_clf.pkl')


# ═══════════════════════════════════════════════════════════════════
# PREDICT — apply to today's projections
# ═══════════════════════════════════════════════════════════════════

def predict(game_date: str | None = None):
    if game_date is None:
        game_date = date.today().isoformat()

    path = os.path.join(ARTIFACTS, 'matchup_clf.pkl')
    if not os.path.exists(path):
        print(f'  No matchup classifier at {path}. Run `train` first.')
        return
    bundle = joblib.load(path)
    model = bundle['model']
    features = bundle['features']

    engine = get_engine()
    df = pd.read_sql_query(
        f"""
        WITH hitter_rolling AS (
          WITH per_game AS (
            SELECT hitter_id, game_pk, game_date,
                   COUNT(*) AS pa,
                   SUM(CASE WHEN event_type IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) AS hits,
                   SUM(CASE event_type WHEN 'single' THEN 1 WHEN 'double' THEN 2 WHEN 'triple' THEN 3 WHEN 'home_run' THEN 4 ELSE 0 END) AS tb,
                   SUM(CASE WHEN event_type = 'home_run' THEN 1 ELSE 0 END) AS hr,
                   SUM(CASE WHEN event_type IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS k,
                   SUM(CASE WHEN event_type IN ('walk','intent_walk') THEN 1 ELSE 0 END) AS bb,
                   SUM(CASE WHEN event_type NOT IN ('walk','intent_walk','hit_by_pitch','sac_fly','sac_bunt') THEN 1 ELSE 0 END) AS ab
            FROM at_bats
            WHERE game_date < '{game_date}'
            GROUP BY hitter_id, game_pk, game_date
          ),
          ranked AS (
            SELECT hitter_id,
                   ROW_NUMBER() OVER (PARTITION BY hitter_id ORDER BY game_date DESC) AS rn,
                   pa, hits, tb, hr, k, bb, ab
            FROM per_game
          )
          SELECT hitter_id,
                 SUM(hits) FILTER (WHERE rn <= 30)::float / NULLIF(SUM(ab) FILTER (WHERE rn <= 30), 0) AS h_roll30_avg,
                 SUM(tb)   FILTER (WHERE rn <= 30)::float / NULLIF(SUM(ab) FILTER (WHERE rn <= 30), 0) AS h_roll30_slg,
                 SUM(hr)   FILTER (WHERE rn <= 30)::float / NULLIF(SUM(pa) FILTER (WHERE rn <= 30), 0) AS h_roll30_hr_rate,
                 SUM(k)    FILTER (WHERE rn <= 30)::float / NULLIF(SUM(pa) FILTER (WHERE rn <= 30), 0) AS h_roll30_k_rate,
                 SUM(bb)   FILTER (WHERE rn <= 30)::float / NULLIF(SUM(pa) FILTER (WHERE rn <= 30), 0) AS h_roll30_bb_rate,
                 COUNT(*)  FILTER (WHERE rn <= 30)::int AS h_roll30_games,
                 SUM(hits) FILTER (WHERE rn <= 10)::float / NULLIF(SUM(ab) FILTER (WHERE rn <= 10), 0) AS h_roll10_avg,
                 SUM(tb)   FILTER (WHERE rn <= 10)::float / NULLIF(SUM(ab) FILTER (WHERE rn <= 10), 0) AS h_roll10_slg,
                 SUM(pa)   FILTER (WHERE rn <= 10)::int AS h_roll10_pa
          FROM ranked
          GROUP BY hitter_id
        ),
        pitcher_rolling AS (
          WITH per_start AS (
            SELECT pitcher_id, game_pk, game_date,
                   COUNT(*) AS bf,
                   SUM(CASE WHEN event_type IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) AS h,
                   SUM(CASE WHEN event_type = 'home_run' THEN 1 ELSE 0 END) AS hr,
                   SUM(CASE WHEN event_type IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS k,
                   SUM(CASE WHEN event_type IN ('walk','intent_walk') THEN 1 ELSE 0 END) AS bb,
                   SUM(CASE WHEN event_type NOT IN ('walk','intent_walk','hit_by_pitch','sac_fly','sac_bunt') THEN 1 ELSE 0 END) AS ab
            FROM at_bats
            WHERE game_date < '{game_date}'
            GROUP BY pitcher_id, game_pk, game_date
            HAVING COUNT(*) >= 12
          ),
          ranked AS (
            SELECT pitcher_id,
                   ROW_NUMBER() OVER (PARTITION BY pitcher_id ORDER BY game_date DESC) AS rn,
                   bf, h, hr, k, bb, ab
            FROM per_start
          )
          SELECT pitcher_id,
                 SUM(k)  FILTER (WHERE rn <= 5)::float / NULLIF(SUM(bf) FILTER (WHERE rn <= 5), 0) AS p_roll5_k_rate,
                 SUM(bb) FILTER (WHERE rn <= 5)::float / NULLIF(SUM(bf) FILTER (WHERE rn <= 5), 0) AS p_roll5_bb_rate,
                 SUM(hr) FILTER (WHERE rn <= 5)::float / NULLIF(SUM(bf) FILTER (WHERE rn <= 5), 0) AS p_roll5_hr_rate,
                 SUM(h)  FILTER (WHERE rn <= 5)::float / NULLIF(SUM(ab) FILTER (WHERE rn <= 5), 0) AS p_roll5_baa,
                 COUNT(*) FILTER (WHERE rn <= 5)::int AS p_roll5_starts
          FROM ranked
          GROUP BY pitcher_id
        ),
        hitter_prior AS (
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
        SELECT p.hitter_id, p.pitcher_id, p.game_pk, p.dk_pts, p.expected_pa, p.lineup_slot,
               p.hitter_hand, p.pitcher_hand, g.season, g.weather,
               pf.hr_factor, pf.hit_factor, pf.k_factor,
               pss.fip AS p_prior_fip, pss.games_started AS p_prior_games_started,
               pss.avg_k, pss.avg_bb, pss.avg_hr, pss.avg_ip,
               pp.arsenal, pp.primary_pitch AS primary_family,
               hp.h_prior_pa, hp.h_prior_avg, hp.h_prior_slg,
               hp.h_prior_hr_rate, hp.h_prior_k_rate, hp.h_prior_bb_rate,
               sp.avg_exit_velo AS h_avg_exit_velo,
               sp.avg_launch_angle AS h_avg_launch_angle,
               sp.pull_pct AS h_pull_pct,
               rf.form_ratio AS h_form_ratio,
               hs.avg AS h_avg_vs_primary,
               hs.whiff_pct AS h_whiff_vs_primary,
               hs.hard_hit_pct AS h_hard_hit_vs_primary,
               hr.h_roll30_avg, hr.h_roll30_slg, hr.h_roll30_hr_rate,
               hr.h_roll30_k_rate, hr.h_roll30_bb_rate, hr.h_roll30_games,
               hr.h_roll10_avg, hr.h_roll10_slg, hr.h_roll10_pa,
               pr.p_roll5_k_rate, pr.p_roll5_bb_rate, pr.p_roll5_hr_rate,
               pr.p_roll5_baa, pr.p_roll5_starts
        FROM projections p
        JOIN games g ON g.game_pk = p.game_pk
        LEFT JOIN park_factors pf ON pf.venue_id = g.venue_id
        LEFT JOIN hitter_prior hp ON hp.hitter_id = p.hitter_id AND hp.season = g.season
        LEFT JOIN hitter_rolling hr ON hr.hitter_id = p.hitter_id
        LEFT JOIN pitcher_rolling pr ON pr.pitcher_id = p.pitcher_id
        LEFT JOIN hitter_spray_profiles sp ON sp.hitter_id = p.hitter_id
        LEFT JOIN hitter_recent_form rf ON rf.hitter_id = p.hitter_id
        LEFT JOIN LATERAL (
          SELECT * FROM pitcher_season_stats WHERE pitcher_id = p.pitcher_id
          ORDER BY season DESC LIMIT 1
        ) pss ON TRUE
        LEFT JOIN LATERAL (
          SELECT arsenal, primary_pitch FROM pitcher_profiles WHERE pitcher_id = p.pitcher_id
          ORDER BY season DESC LIMIT 1
        ) pp ON TRUE
        LEFT JOIN LATERAL (
          SELECT avg, whiff_pct, hard_hit_pct
          FROM hitter_pitch_splits hs
          WHERE hs.hitter_id = p.hitter_id AND hs.pitch_family = pp.primary_pitch
          ORDER BY hs.season DESC LIMIT 1
        ) hs ON TRUE
        WHERE p.game_date = '{game_date}'
        """,
        engine,
    )
    if df.empty:
        print(f'  No projections for {game_date}')
        return

    # Compute derived features
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
    arsenal_ext = df['arsenal'].apply(_arsenal)
    df = pd.concat([df, arsenal_ext], axis=1)

    # Pitcher rates from raw season stats (same formula as training)
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
        'h_avg_exit_velo': 88.0, 'h_avg_launch_angle': 12.0, 'h_pull_pct': 0.37,
        'h_avg_vs_primary': 0.248, 'h_whiff_vs_primary': 0.24, 'h_hard_hit_vs_primary': 0.35,
        'h_form_ratio': 1.0,
        'lineup_slot': 5,
        'h_roll30_avg': 0.248, 'h_roll30_slg': 0.405, 'h_roll30_hr_rate': 0.030,
        'h_roll30_k_rate': 0.22, 'h_roll30_bb_rate': 0.085, 'h_roll30_games': 0,
        'h_roll10_avg': 0.248, 'h_roll10_slg': 0.405, 'h_roll10_pa': 0,
        'p_roll5_k_rate': 0.22, 'p_roll5_bb_rate': 0.085,
        'p_roll5_hr_rate': 0.028, 'p_roll5_baa': 0.248, 'p_roll5_starts': 0,
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
        else:
            df[c] = df[c].fillna(v)

    # Ensure every feature column exists
    for f in features:
        if f not in df.columns:
            df[f] = 0.0

    X = df[features].astype(float).values
    probs = model.predict_proba(X)
    expected_dk_per_pa = (probs * DK_VEC).sum(axis=1)
    expected_fd_per_pa = (probs * FD_VEC).sum(axis=1)
    expected_pa = df['expected_pa'].fillna(4.0).values
    df['ml_dk_pts'] = expected_dk_per_pa * expected_pa
    df['ml_fd_pts'] = expected_fd_per_pa * expected_pa

    # The per-AB classifier doesn't see runs-scored or stolen bases — those
    # are separate post-AB events. Pull each hitter's career rate from
    # hitter_game_stats and add their DK/FD contribution.
    run_sb = pd.read_sql_query(
        """
        SELECT hitter_id,
               AVG(runs)::float         AS r_per_game,
               AVG(stolen_bases)::float AS sb_per_game
        FROM hitter_game_stats
        GROUP BY hitter_id
        """,
        get_engine(),
    )
    df = df.merge(run_sb, left_on='hitter_id', right_on='hitter_id', how='left')
    df['r_per_game']  = df['r_per_game'].fillna(0.40)   # league avg ~0.41
    df['sb_per_game'] = df['sb_per_game'].fillna(0.06)

    # DK: +2/run, +5/SB;  FD: +3.2/run, +6/SB
    df['ml_dk_pts'] = df['ml_dk_pts'] + df['r_per_game'] * 2.0 + df['sb_per_game'] * 5.0
    df['ml_fd_pts'] = df['ml_fd_pts'] + df['r_per_game'] * 3.2 + df['sb_per_game'] * 6.0

    df['ml_delta'] = df['ml_dk_pts'] - df['dk_pts']

    # Persist per-outcome probabilities — the NRFI simulator consumes these.
    prob_blobs = []
    for i in range(len(df)):
        prob_blobs.append({cls: float(probs[i, j]) for j, cls in enumerate(CLASSES)})
    df['_prob_blob'] = prob_blobs

    # Write back
    import json
    session = get_session()
    try:
        for _, row in df.iterrows():
            session.execute(
                text("""
                UPDATE projections
                SET ml_dk_pts       = :ml_dk,
                    ml_fd_pts       = :ml_fd,
                    ml_delta        = :delta,
                    ml_outcome_probs = CAST(:probs AS jsonb),
                    blend_dk_pts = CASE
                      WHEN tuned_dk_pts IS NOT NULL THEN (tuned_dk_pts + :ml_dk) / 2.0
                      ELSE :ml_dk
                    END,
                    blend_fd_pts = :ml_fd
                WHERE hitter_id = :hid AND game_pk = :gpk
                """),
                {
                    'ml_dk': float(row['ml_dk_pts']),
                    'ml_fd': float(row['ml_fd_pts']),
                    'delta': float(row['ml_delta']),
                    'probs': json.dumps(row['_prob_blob']),
                    'hid':   int(row['hitter_id']),
                    'gpk':   int(row['game_pk']),
                },
            )
        session.commit()
    finally:
        session.close()

    print(f'  wrote {len(df)} matchup predictions. '
          f'mean DK={df["ml_dk_pts"].mean():.2f}  FD={df["ml_fd_pts"].mean():.2f}  '
          f'(factor DK={df["dk_pts"].mean():.2f})')


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
