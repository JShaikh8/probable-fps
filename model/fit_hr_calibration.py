"""
Isotonic calibration for HR and Hit probabilities.

Problem:
  The factor model produces `proj.hr` (expected HRs per game), and the Props UI
  computes P(HR) = 1 - exp(-proj.hr). This turned out to be miscalibrated —
  observed HR rates in the low-P(HR) buckets were 4-6× higher than predicted,
  and the top bucket was slightly over-predicted.

Fix:
  Fit an isotonic regression from predicted P(X) → empirical rate on reconciled
  data. Save two monotone lookups: hr_calibration and hit_calibration. Apply
  in the UI query layer (`queries.ts`) via a small lookup table.

Run:
  python -m model.fit_hr_calibration
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from config import get_engine


ARTIFACTS = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
os.makedirs(ARTIFACTS, exist_ok=True)


def _p_from_expected(x: float | None) -> float:
    """Poisson P(≥1 event) = 1 - exp(-λ) for expected count λ."""
    if x is None:
        return 0.0
    return 1.0 - math.exp(-max(0.0, float(x)))


def fit_and_save() -> None:
    engine = get_engine()
    print('Loading reconciled projections…')
    df = pd.read_sql_query(
        """
        SELECT p.proj, pa.actual
        FROM projection_actuals pa
        JOIN projections p
          ON p.hitter_id = pa.hitter_id AND p.game_pk = pa.game_pk
        WHERE pa.actual IS NOT NULL AND p.proj IS NOT NULL
        """,
        engine,
    )
    if len(df) < 500:
        print(f'  Only {len(df)} reconciled rows. Need 500+.')
        return
    print(f'  {len(df):,} reconciled hitter-games')

    df['proj_hr_exp']  = df['proj'].apply(lambda d: (d or {}).get('hr'))
    df['proj_h_exp']   = df['proj'].apply(lambda d: (d or {}).get('h'))
    df['actual_hr']    = df['actual'].apply(lambda d: 1 if ((d or {}).get('hr') or 0) > 0 else 0)
    df['actual_hit']   = df['actual'].apply(lambda d: 1 if ((d or {}).get('h') or 0) > 0 else 0)

    df['pred_hr']  = df['proj_hr_exp'].apply(_p_from_expected)
    df['pred_hit'] = df['proj_h_exp'].apply(_p_from_expected)

    for name, pred_col, actual_col in [
        ('hr',  'pred_hr',  'actual_hr'),
        ('hit', 'pred_hit', 'actual_hit'),
    ]:
        sub = df[[pred_col, actual_col]].dropna()
        x = sub[pred_col].astype(float).values
        y = sub[actual_col].astype(int).values

        # Pre-cal metrics
        br_pre = brier_score_loss(y, x)
        ll_pre = log_loss(y, np.clip(x, 1e-6, 1 - 1e-6))

        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(x, y)
        calibrated = iso.predict(x)

        br_post = brier_score_loss(y, calibrated)
        ll_post = log_loss(y, np.clip(calibrated, 1e-6, 1 - 1e-6))

        print(f'\n  [{name}] n={len(sub):,}')
        print(f'    Brier  pre={br_pre:.5f}  post={br_post:.5f}  Δ={br_post - br_pre:+.5f}')
        print(f'    logL   pre={ll_pre:.4f}  post={ll_post:.4f}  Δ={ll_post - ll_pre:+.4f}')

        # Serialize as a 1024-point lookup table — UI can apply without Python.
        xs = np.linspace(0.0, 1.0, 1024)
        ys = iso.predict(xs)
        table = {
            'name': name,
            'n_train': int(len(sub)),
            'brier_pre': float(br_pre),
            'brier_post': float(br_post),
            'xs': xs.tolist(),
            'ys': ys.tolist(),
        }
        path = os.path.join(ARTIFACTS, f'calibration_{name}.json')
        with open(path, 'w') as f:
            json.dump(table, f)
        print(f'    → saved {path}')


if __name__ == '__main__':
    fit_and_save()
