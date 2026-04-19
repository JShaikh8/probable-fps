# sports-oracle-python

Local MLB data pipeline + projection engine. Ingests StatsAPI game feeds into
Postgres, builds feature tables, trains hitter/pitcher archetypes, and writes
daily hitter + pitcher + NRFI projections. Reconciles against final box scores
nightly so the UI can show calibration metrics.

Frontend lives at [../sports-oracle-ui](../sports-oracle-ui).

## Setup

```bash
# start Postgres (+ pgAdmin on :5050)
docker compose up -d

# install deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# create all tables
python -m db.init_schema
```

`.env.local` holds `DATABASE_URL` (defaults to the Docker Compose Postgres).

## Run

```bash
python run_daily.py                    # full pipeline for today
python run_daily.py --date 2024-07-15  # specific date
python run_daily.py --only-projections # projections + NRFI only
python run_daily.py --skip-reconcile   # skip yesterday's reconciliation
python run_daily.py --skip-ingest      # features + projections + NRFI
```

Backfill by running `ingest_runner` directly:

```bash
python -m ingest.ingest_runner --seasons 2020 2021 2022 2023 2024 2025 2026
```

At ~0.15s per game × ~2,430 games per season × 7 seasons, plan for ~1 hour of
cold ingest. StatsAPI rate limit: comfortable at 6-7 req/s.

## Pipeline order

0. **Reconciliation** — yesterday's projections vs final box scores
1. **Ingest** — completed games → `games`, `at_bats`, `pitches`
2. **Pitch splits** — hitter × pitch-family × season aggregates, pitcher arsenal
3. **Park factors** — raw ratios per venue (HR, hit, hard-hit, K, BB + 3×3 hit grid)
4. **Pitcher season stats** — per-start FIP, IP, K/BB/H/HR averages
5. **Hitter recent form + spray profiles**
6. **Archetypes** — cosine-similar hitters + pitchers on feature vectors
7. **Projections** — hitter (DK/FD pts + 7 factor signals) + pitcher
8. **NRFI** — Poisson on first-inning λ from linear-weight run contributions

## Schema

SQLAlchemy models in [db/models.py](db/models.py). Drizzle mirror for the UI in
[../sports-oracle-ui/src/lib/db/schema.ts](../sports-oracle-ui/src/lib/db/schema.ts).

Reconciliation writes into `projection_actuals`, `pitcher_projection_actuals`,
and `nrfi_actuals` — these power the Calibration screen in the UI.

## Deployment

Not deployed. Runs locally. Scheduling is easiest via `cron` or `launchd`:

```cron
# 6am CT nightly (after all MLB games final)
0 11 * * * cd ~/projects/sports-oracle-python && .venv/bin/python run_daily.py
```
