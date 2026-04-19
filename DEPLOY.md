# Deploying (free tier)

This guide walks through **hosting everything for $0/month** using:

- **Render** — managed Postgres (free) + Next.js UI (free)
- **GitHub Actions** — nightly pipeline (free, 2,000 min/month)

```
  GitHub Actions cron            Render free web service
  ┌──────────────────┐           ┌──────────────────────┐
  │ nightly.yml      │           │ sports-oracle-ui     │
  │ runs daily 6am CT│           │ (Next.js, sleeps)    │
  └────────┬─────────┘           └───────────┬──────────┘
           │                                 │
           ▼                                 ▼
           ┌────────────────────────────────────┐
           │ Render Free Postgres (1 GB, 90-day) │
           │ sports-oracle-db                    │
           └────────────────────────────────────┘
```

## Free tier caveats (read before committing)

1. **Render Free Postgres expires 90 days after creation.** Before then
   you either upgrade ($6/mo) or dump + recreate. Our DB is ~600MB so
   moving is fast.
2. **Render Free Web Service spins down after 15 min of inactivity.**
   First request after idle takes ~30s to wake. Fine for personal use.
3. **GitHub Actions** has a 2,000 min/month free cap on private repos.
   Our job uses ~10 min/night = ~300 min/month. If your repo is public,
   Actions are **unlimited free**.

## Step 1 — Create the Render database

1. In Render: **New → Blueprint** → connect `JShaikh8/probable-fps`.
2. Render reads [`render.yaml`](./render.yaml) and creates **sports-oracle-db**
   on the free tier.
3. Open the new database service → **Connect** tab → copy the
   **External Database URL**. You'll need it twice.

## Step 2 — Seed the database

### Option A: Migrate your local data (fastest, recommended)

From your local machine:

```bash
pg_dump \
  --no-owner --no-acl --format=custom \
  "postgresql://sportsoracle:sportsoracle@localhost:5432/sports_oracle" \
  > sports_oracle.dump

pg_restore --no-owner --no-acl --clean --if-exists --dbname \
  "<EXTERNAL URL FROM STEP 1>" \
  sports_oracle.dump
```

Takes about 5-10 minutes over the public internet.

### Option B: Cold ingest via GitHub Actions

Skip the pg_dump and let the nightly job do it. Set `DATABASE_URL` as a
repo secret (step 3), then manually trigger `nightly.yml` via the Actions
tab. First run will take ~30 min on a fresh DB (init schema + ingest).

## Step 3 — Set up the GitHub Actions cron (free)

1. Go to your `probable-fps` repo on GitHub.
2. **Settings → Secrets and variables → Actions → New repository secret**.
3. Name: `DATABASE_URL`, value: the External URL from step 1.
4. The workflow at [`.github/workflows/nightly.yml`](./.github/workflows/nightly.yml)
   runs automatically at 11:00 UTC daily, or manually from the
   **Actions tab → nightly pipeline → Run workflow**.

## Step 4 — Deploy the UI

1. In Render: **New → Blueprint** → connect `JShaikh8/stunning-goggles`.
2. The blueprint creates **sports-oracle-ui** on the free tier.
3. Open the new web service → **Environment** tab.
4. Add `DATABASE_URL` with the same External URL from step 1.
5. Trigger a **Manual Deploy**.

Your UI is live at `https://sports-oracle-ui.onrender.com/slate` (or
similar — Render gives you the exact URL).

## Step 5 — Verify

```bash
curl -sI https://sports-oracle-ui.onrender.com/slate | head -1
# HTTP/2 200

# Trigger a one-off pipeline run to confirm Postgres connectivity:
gh workflow run nightly.yml --repo JShaikh8/probable-fps
```

Then watch the run in the Actions tab.

## Ongoing costs & limits

| Resource | Limit | What happens when exceeded |
|---|---|---|
| Postgres free tier | 1 GB, 90-day expiry | Service suspends; pg_dump + recreate |
| UI free web | 512 MB RAM, idle spin-down | First request after idle = 30s cold start |
| Actions (public repo) | Unlimited | — |
| Actions (private repo) | 2,000 min/month | Jobs skipped until next month |

## Upgrade path if you outgrow free

- **Render Starter** ($7/mo each): no spin-down on web, Postgres doesn't
  expire, can host the cron there ($7/mo for the cron service) — uncomment
  the `services:` block in [`render.yaml`](./render.yaml) and push.
- **Neon** (alternative): free 0.5GB Postgres with no expiration. Swap
  the `DATABASE_URL`.
