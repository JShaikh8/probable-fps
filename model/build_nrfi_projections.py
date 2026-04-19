"""
NRFI (No Runs First Inning) projection engine — Markov chain version.

For each game:
  1. Load the first 9 projected hitters per team with their per-PA outcome
     probability distribution (from ML classifier, or synthesized from factor
     projection if ML probs aren't available).
  2. Run a Markov-chain simulation of the first inning from state (empty
     bases, 0 outs, 0 runs), evolving through the lineup until 3 outs.
  3. Compute P(team scores) per team.
  4. P(NRFI) = P(home scoreless) × P(away scoreless).

Output fields per game:
  nrfi_prob, nrfi_pct, yrfi_pct
  home_xr (expected first-inning runs), away_xr
  home_p_scoreless, away_p_scoreless
  home_p_score, away_p_score  (= 1 - p_scoreless)
  top_threats (ordered by per-hitter score contribution)
  home_top_batters / away_top_batters  (with per-hitter p_on_base / p_score)
  home_pitcher / away_pitcher summaries
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

import pandas as pd

from config import get_engine, get_session
from db.io import bulk_upsert
from db.models import NrfiProjection
from model.nrfi_simulator import (
    simulate_first_inning, summarize, probs_from_proj,
)


LEAGUE_ERA = 4.20


def run(game_date: str | None = None):
    if game_date is None:
        game_date = date.today().isoformat()

    engine = get_engine()
    print(f'Building NRFI projections for {game_date}…')

    hitters = pd.read_sql_query(
        f"""
        SELECT pr.hitter_id, pr.pitcher_id, pr.game_pk, pr.game_date, pr.side,
               pr.proj, pr.dk_pts, pr.expected_pa, pr.lineup_slot,
               pr.weather, pr.ml_outcome_probs,
               pl.full_name AS hitter_name,
               g.venue_id,
               ht.abbrev AS home_abbrev, at_.abbrev AS away_abbrev,
               v.name AS venue_name
        FROM projections pr
        LEFT JOIN players pl ON pl.player_id = pr.hitter_id
        LEFT JOIN games g    ON g.game_pk    = pr.game_pk
        LEFT JOIN teams ht   ON ht.team_id   = g.home_team_id
        LEFT JOIN teams at_  ON at_.team_id  = g.away_team_id
        LEFT JOIN venues v   ON v.venue_id   = g.venue_id
        WHERE pr.game_date = '{game_date}'
        """,
        engine,
    )
    if hitters.empty:
        print('  No hitter projections for this date. Run build_projections.py first.')
        return

    pitchers = pd.read_sql_query(
        f"""
        SELECT pp.pitcher_id, pp.game_pk, pp.proj, pp.dk_pts, pp.fip,
               pp.stuff_signal, pp.side, pl.full_name AS pitcher_name
        FROM pitcher_projections pp
        LEFT JOIN players pl ON pl.player_id = pp.pitcher_id
        WHERE pp.game_date = '{game_date}'
        """,
        engine,
    )
    pitcher_by_game_side = {
        (int(r['game_pk']), r['side']): r for _, r in pitchers.iterrows() if r['side']
    }

    # Group hitters by game × side, sorted by lineup slot
    games: dict[int, dict] = defaultdict(lambda: {'home': [], 'away': [], 'meta': {}})
    for _, r in hitters.iterrows():
        gp = int(r['game_pk'])
        side = r['side'] or 'home'
        # Use ML outcome probs if present, else synthesize from factor projection
        probs = r['ml_outcome_probs']
        if not isinstance(probs, dict) or not probs:
            probs = probs_from_proj(r['proj'] or {}, float(r['expected_pa'] or 4.0))

        games[gp][side].append({
            'hitter_id':   int(r['hitter_id']),
            'hitter_name': r['hitter_name'] or '',
            'pitcher_id':  int(r['pitcher_id']) if r['pitcher_id'] is not None else None,
            'proj':        r['proj'] or {},
            'dk_pts':      float(r['dk_pts'] or 0.0),
            'lineup_slot': int(r['lineup_slot']) if r['lineup_slot'] is not None else 9,
            'expected_pa': float(r['expected_pa'] or 4.0),
            'side':        side,
            'abbrev':      r['home_abbrev'] if side == 'home' else r['away_abbrev'],
            'probs':       probs,
        })
        if not games[gp]['meta']:
            games[gp]['meta'] = {
                'venue_id':    int(r['venue_id']) if r['venue_id'] is not None else None,
                'venue_name':  r['venue_name'] or '',
                'home_abbrev': r['home_abbrev'] or '',
                'away_abbrev': r['away_abbrev'] or '',
                'weather':     r['weather'] or {},
            }

    records: list[dict] = []
    print_lines: list[tuple] = []

    for game_pk, g in games.items():
        home_bats = sorted(g['home'], key=lambda x: x['lineup_slot'])[:9]
        away_bats = sorted(g['away'], key=lambda x: x['lineup_slot'])[:9]
        meta = g['meta']
        if not home_bats or not away_bats:
            continue

        away_pitcher = pitcher_by_game_side.get((game_pk, 'away'), None)
        home_pitcher = pitcher_by_game_side.get((game_pk, 'home'), None)
        away_pitcher_id = int(away_pitcher['pitcher_id']) if away_pitcher is not None else None
        home_pitcher_id = int(home_pitcher['pitcher_id']) if home_pitcher is not None else None

        # Pitcher-quality scalar: FIP better-than-league → dampen outcome probs;
        # worse-than-league → amplify. Applied gently to avoid double-counting
        # pitcher features already present in the classifier outputs.
        home_adj = _pitcher_adjustment(_fget(away_pitcher, 'fip'))
        away_adj = _pitcher_adjustment(_fget(home_pitcher, 'fip'))

        home_dist = simulate_first_inning(
            [_apply_adj(b['probs'], home_adj) for b in home_bats],
        )
        away_dist = simulate_first_inning(
            [_apply_adj(b['probs'], away_adj) for b in away_bats],
        )
        home_s = summarize(home_dist)
        away_s = summarize(away_dist)

        nrfi = home_s['p_scoreless'] * away_s['p_scoreless']

        # Per-batter contribution to team scoring probability
        home_threats = _rank_batter_threats(home_bats, home_adj)
        away_threats = _rank_batter_threats(away_bats, away_adj)
        threats = sorted(home_threats + away_threats,
                         key=lambda x: x['fi_run_contrib'], reverse=True)[:6]

        records.append({
            'game_pk':   int(game_pk),
            'game_date': game_date,
            'nrfi_prob': round(nrfi, 4),
            'nrfi_pct':  round(nrfi * 100, 1),
            'yrfi_pct':  round((1 - nrfi) * 100, 1),
            'home_xr':   round(home_s['expected_runs'], 4),
            'away_xr':   round(away_s['expected_runs'], 4),
            'home_p_scoreless': round(home_s['p_scoreless'], 4),
            'away_p_scoreless': round(away_s['p_scoreless'], 4),
            'home_p_score':     round(home_s['p_score'], 4),
            'away_p_score':     round(away_s['p_score'], 4),
            'top_threats': threats,
            'home_top_batters': [_batter_summary(b, home_adj) for b in home_bats[:5]],
            'away_top_batters': [_batter_summary(b, away_adj) for b in away_bats[:5]],
            'home_pitcher': _pitcher_summary(home_pitcher, home_pitcher_id),
            'away_pitcher': _pitcher_summary(away_pitcher, away_pitcher_id),
        })
        print_lines.append((
            meta['away_abbrev'], meta['home_abbrev'],
            nrfi * 100, (1 - nrfi) * 100,
            home_s['expected_runs'], away_s['expected_runs'],
        ))

    session = get_session()
    try:
        bulk_upsert(session, NrfiProjection, records, pk_cols=['game_pk'])
        session.commit()
        print(f'  {len(records)} NRFI projections written')
    finally:
        session.close()

    print_lines.sort(key=lambda x: x[2], reverse=True)
    print(f'\n  {"Game":<12} {"NRFI%":>7} {"YRFI%":>7}  {"Home xR":>8}  {"Away xR":>8}')
    print(f'  {"-" * 50}')
    for away, home, nrfi_pct, yrfi_pct, hxr, axr in print_lines:
        print(f'  {away}@{home:<8} {nrfi_pct:>6.1f}%  {yrfi_pct:>6.1f}%  {hxr:>8.3f}  {axr:>8.3f}')
    print('\nDone.')


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _pitcher_adjustment(fip) -> float:
    """
    Gentle multiplier on outcome probabilities from pitcher quality.
    Elite pitcher (FIP ≤ 3.5) → downweight offensive outcomes ~15%.
    Bad pitcher (FIP ≥ 5.0) → amplify offensive outcomes ~10%.
    """
    if fip is None:
        return 1.0
    delta = (fip - LEAGUE_ERA) / LEAGUE_ERA    # +ve = worse pitcher
    adj = 1.0 + delta * 0.30
    return max(0.80, min(1.15, adj))


def _apply_adj(probs: dict, adj: float) -> dict:
    """
    Multiply offensive outcome probs by `adj`, rebalancing K/out to keep total = 1.
    """
    if adj == 1.0:
        return probs
    off_keys = ('single', 'double', 'triple', 'home_run', 'walk', 'hit_by_pitch')
    off_sum = sum(probs.get(k, 0.0) for k in off_keys)
    new_off_sum = off_sum * adj
    # Redistribute remainder to K + out in proportion to their current share
    rem = 1.0 - new_off_sum
    rem = max(0.02, rem)  # don't let out+K shrink to zero
    k_share = probs.get('strikeout', 0.2)
    o_share = probs.get('out', 0.45)
    defence_total = k_share + o_share
    if defence_total <= 0:
        defence_total = 1.0
    new_k = rem * (k_share / defence_total)
    new_o = rem * (o_share / defence_total)
    out = dict(probs)
    for k in off_keys:
        out[k] = probs.get(k, 0.0) * adj
    out['strikeout'] = new_k
    out['out'] = new_o
    return out


def _rank_batter_threats(batters: list[dict], adj: float) -> list[dict]:
    """
    Per-batter contribution: simulate inning where this batter leads off
    alone (approximation of their individual threat). Sort by expected runs.
    """
    threats = []
    for b in batters:
        dist = simulate_first_inning([_apply_adj(b['probs'], adj)])
        s = summarize(dist)
        threats.append({
            'hitter_id':     b['hitter_id'],
            'hitter_name':   b['hitter_name'],
            'team':          b['abbrev'],
            'side':          b['side'],
            'lineup_slot':   b['lineup_slot'],
            'dk_pts':        b['dk_pts'],
            'fi_run_contrib': round(s['expected_runs'], 4),
            'p_on_base':     round(_p_on_base(b['probs']), 4),
        })
    return threats


def _batter_summary(b: dict, adj: float) -> dict:
    probs = _apply_adj(b['probs'], adj)
    return {
        'hitter_id':   b['hitter_id'],
        'hitter_name': b['hitter_name'],
        'lineup_slot': b['lineup_slot'],
        'dk_pts':      b['dk_pts'],
        'proj':        b['proj'],
        'fi_weight':   SLOT_FI_WEIGHTS.get(b['lineup_slot'], 0.0),
        'p_on_base':   round(_p_on_base(probs), 4),
        'probs':       {k: round(v, 4) for k, v in probs.items()},
    }


SLOT_FI_WEIGHTS = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.82, 5: 0.62}


def _p_on_base(probs: dict) -> float:
    return (probs.get('single', 0) + probs.get('double', 0) +
            probs.get('triple', 0) + probs.get('home_run', 0) +
            probs.get('walk', 0) + probs.get('hit_by_pitch', 0))


def _pitcher_summary(row, pitcher_id) -> dict:
    if row is None:
        return {'pitcher_id': pitcher_id}
    pid = _fget(row, 'pitcher_id')
    return {
        'pitcher_id':   int(pid) if pid is not None else pitcher_id,
        'pitcher_name': _fget(row, 'pitcher_name') or '',
        'fip':          _fget(row, 'fip'),
        'stuff_signal': _fget(row, 'stuff_signal'),
        'dk_pts':       _fget(row, 'dk_pts'),
    }


def _fget(rowlike, key):
    """Safely pull a field from a pandas Series / None. NaN → None."""
    if rowlike is None:
        return None
    try:
        v = rowlike[key]
    except (KeyError, IndexError, TypeError):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str)
    args = parser.parse_args()
    run(game_date=args.date)
