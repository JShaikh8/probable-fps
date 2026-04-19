# Deploying to Render

This guide deploys **both** the Python pipeline and the Next.js UI to
Render, sharing a single managed Postgres database.

## Architecture

```
  ┌─────────────────┐       ┌──────────────────────────┐
  │ Render Cron     │       │ Render Web Service       │
  │ sports-oracle-  │       │ sports-oracle-ui         │
  │ pipeline        │       │ (Next.js)                │
  │ runs 6am CT     │       │                          │
  └────────┬────────┘       └───────────┬──────────────┘
           │                            │
           ▼                            ▼
           ┌─────────────────────────────────┐
           │ Render Managed Postgres         │
           │ sports-oracle-db                │
           └─────────────────────────────────┘
```

## Prerequisites

- GitHub account with both repos pushed:
  - `sports-oracle-python` (this repo)
  - `sports-oracle-ui` (sibling repo at `../sports-oracle-ui`)
- Render account (free tier works for evaluation)
- Local Postgres already populated (run the pipeline locally first so there's
  data to migrate — see the main `README.md`)

## Step 1 — Push both repos to GitHub

```bash
cd ~/projects/sports-oracle-python
git init -b main                    # if not already a repo
git add .
git commit -m "initial"
gh repo create sports-oracle-python --public --source . --push

cd ~/projects/sports-oracle-ui
git init -b main
git add .
git commit -m "initial"
gh repo create sports-oracle-ui --public --source . --push
```

**Important:** `.env.local` is in `.gitignore` for both repos — your
credentials won't be pushed.

## Step 2 — Deploy the Python pipeline + database

1. In Render: **New → Blueprint**.
2. Connect your `sports-oracle-python` repo.
3. Render reads [`render.yaml`](./render.yaml) and provisions:
   - A Postgres database named **sports-oracle-db**.
   - A nightly cron job named **sports-oracle-pipeline**.

The cron builds its own Python env during deploy (`pip install`) and
connects to the DB via `DATABASE_URL` (injected automatically from the
blueprint).

## Step 3 — Seed the database

You have two options. Pick **(A) pg_dump migration** if you have good
local data (much faster), or **(B) fresh ingest** if you're starting from
scratch.

### Option A: Migrate your local data (≈ 5-10 min)

Export from local Postgres:

```bash
# Dump everything. Use Render's recommended --no-owner / --no-acl flags.
pg_dump \
  --no-owner --no-acl --format=custom \
  "postgresql://sportsoracle:sportsoracle@localhost:5432/sports_oracle" \
  > sports_oracle.dump
```

Find your Render DB's external connection string in the Render dashboard
(DB → Connect → External Database URL), then restore:

```bash
pg_restore --no-owner --no-acl --clean --if-exists --dbname \
  "postgres://user:pass@dpg-XXXX.oregon-postgres.render.com/sports_oracle" \
  sports_oracle.dump
```

That's it. The full history lands in Render's DB in minutes.

### Option B: Cold ingest on Render

Open the Python service's **Shell** tab in Render and run:

```bash
python -m db.init_schema
python -m ingest.ingest_runner --seasons 2024 2025 2026   # ~30 min on starter plan
python -m ingest.fetch_runs                               # ~15 min
python run_daily.py --retrain-ml                          # features + models + today's projections
```

Expect ingest to take 1-2× longer on Render's free tier than locally.
Move to the Starter plan ($7/month) if the free cron runs out of time.

## Step 4 — Deploy the Next.js UI

1. In Render: **New → Blueprint** again.
2. Connect your `sports-oracle-ui` repo.
3. The blueprint provisions a web service named **sports-oracle-ui**.
4. After first deploy, open the service's **Environment** tab.
5. Paste the same `DATABASE_URL` you used for the Python pipeline into
   the `DATABASE_URL` slot (marked `sync: false` in the blueprint).
6. Trigger a **Manual Deploy**.

Your UI is live at `https://sports-oracle-ui.onrender.com/slate`.

## Step 5 — Verify

```bash
curl https://sports-oracle-ui.onrender.com/slate -sI | head -1
# HTTP/2 200
```

Inside the Render dashboard, click the cron service → **Trigger Run**
once to confirm the nightly job succeeds against the hosted DB.

## Cost breakdown (as of 2026)

| Service           | Plan       | Monthly |
| ----------------- | ---------- | ------: |
| Postgres          | Basic 256M |  $6     |
| Python cron       | Starter    |  $7     |
| Next.js web       | Starter    |  $7     |
| **Total**         |            | **~$20** |

The free tiers work for evaluation but:
- Free Postgres sleeps after 30 days idle.
- Free web service spins down after 15 min inactivity (first request slow).
- Free cron has a 750 min/month cap (our job runs ≈ 5-10 min/day).

## Ongoing operation

- **Pipeline**: runs nightly at 11:00 UTC (6am CT). Logs in Render → cron service.
- **UI**: auto-deploys on push to `main` in the UI repo.
- **DB backups**: Render's paid plans include daily backups. For free tier,
  run `pg_dump` locally on a schedule.
- **ML model retraining**: the cron runs with `--retrain-ml` every night,
  so the factor tuner and matchup classifier always reflect the latest
  reconciliation data.

## Troubleshooting

**"DATABASE_URL is not set"** — confirm the env var is pasted in the
service's Environment tab; restart the service.

**"could not translate host name"** — using the internal DB URL from an
external machine. Use the External Database URL for pg_dump / pg_restore.

**Cron timeout** — upgrade the cron's plan; ingest + retrain can exceed
free-tier limits on first run.

**Slow cold starts on UI** — expected on free tier. Upgrade to Starter
or higher.
