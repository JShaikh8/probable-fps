#!/bin/bash
# Backfill-and-reconcile loop for model evaluation.
# For each date: build_projections → ml_matchup predict → ml_pipeline predict → reconcile.
set -e
cd "$(dirname "$0")/.."

PY=.venv/bin/python
dates=""
# 30 consecutive dates in August 2025
for d in $(seq 1 30); do
  dates+=" 2025-08-$(printf '%02d' $d)"
done

for d in $dates; do
  echo "════════════════════════════════════════"
  echo "  $d"
  echo "════════════════════════════════════════"
  $PY -m model.build_projections   --date $d 2>&1 | tail -1
  $PY -m model.ml_matchup predict  --date $d 2>&1 | tail -1
  $PY -m model.ml_pipeline predict --date $d 2>&1 | tail -1
  $PY -m model.build_reconciliation --date $d 2>&1 | grep -E 'hitters:|pitchers:' | head -2
done
echo "Done."
