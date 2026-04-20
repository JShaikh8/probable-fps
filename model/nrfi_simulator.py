"""
Markov-chain first-inning simulator.

Given a sequence of per-PA outcome probability distributions for each batter
(from the ML matchup classifier), compute the exact probability that the first
inning ends with 0 runs, 1 run, 2 runs, etc.

The state is (bases, outs, runs_so_far). We evolve the probability distribution
over all reachable states by applying each batter's outcome transition.

Outcomes handled:
  single, double, triple, home_run, walk, hit_by_pitch, strikeout, out

Base advancement conventions (slightly simplified but reasonable):
  - 1B: all runners advance exactly 1 base; runner on 3B scores
  - 2B: all runners advance 2 bases; runners on 2B and 3B score; batter to 2B
  - 3B: all runners score; batter to 3B
  - HR: all runners + batter score; bases clear
  - BB / HBP: force play — runners advance only if forced; bases loaded walks score 1
  - K / out: no runner movement; outs += 1
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable


OUTCOMES = ('single', 'double', 'triple', 'home_run',
            'walk', 'hit_by_pitch', 'strikeout', 'out')


def _advance(bases: tuple[bool, bool, bool], outcome: str) -> tuple[tuple[bool, bool, bool], int, int]:
    """
    Apply one batter's outcome to the bases state.
    Returns (new_bases, runs_scored_on_this_play, outs_added_on_this_play).
    """
    on1, on2, on3 = bases

    if outcome == 'home_run':
        runs = 1 + int(on1) + int(on2) + int(on3)
        return (False, False, False), runs, 0

    if outcome == 'triple':
        runs = int(on1) + int(on2) + int(on3)
        return (False, False, True), runs, 0

    if outcome == 'double':
        runs = int(on2) + int(on3)
        # Runner on 1B advances to 3B (~convention); batter to 2B
        return (False, True, on1), runs, 0

    if outcome == 'single':
        runs = int(on3)
        # Runner on 2B → 3B, runner on 1B → 2B, batter → 1B
        return (True, on1, on2), runs, 0

    if outcome in ('walk', 'hit_by_pitch'):
        # Force-play logic: only forced runners advance.
        # Runner on 1B is always forced. 2B is forced iff 1B was occupied.
        # 3B is forced iff both 1B and 2B were occupied.
        runs = 1 if (on1 and on2 and on3) else 0
        new_on2 = on1
        new_on3 = (on1 and on2) or (on3 and not (on1 and on2 and on3))
        return (True, new_on2, new_on3), runs, 0

    if outcome in ('strikeout', 'out'):
        return bases, 0, 1

    # Unknown outcome — treat as out (safest fallback)
    return bases, 0, 1


def simulate_first_inning(
    batter_probs: list[dict[str, float]],
    max_batters: int = 12,
) -> dict[int, float]:
    """
    Evolve the state distribution through the lineup, stopping when inning ends.
    Returns a dict mapping `runs_scored` → probability.
    """
    # State: (bases_tuple, outs, runs_so_far) → probability
    start = ((False, False, False), 0, 0)
    states: dict[tuple, float] = {start: 1.0}
    final: dict[int, float] = defaultdict(float)

    for i, probs in enumerate(batter_probs[:max_batters]):
        # Normalize (guard against rounding)
        total = sum(probs.get(o, 0.0) for o in OUTCOMES)
        if total <= 0:
            break
        probs = {o: probs.get(o, 0.0) / total for o in OUTCOMES}

        new_states: dict[tuple, float] = defaultdict(float)
        for (bases, outs, runs), p in states.items():
            if outs >= 3:
                final[runs] += p
                continue
            for outcome, op in probs.items():
                if op <= 0:
                    continue
                nb, dr, do_ = _advance(bases, outcome)
                new_outs = outs + do_
                new_runs = runs + dr
                if new_outs >= 3:
                    final[new_runs] += p * op
                else:
                    new_states[(nb, new_outs, new_runs)] += p * op
        states = new_states
        if not states:
            break

    # Any states that never reached 3 outs fold back in (rare with 12-batter cap)
    for (_, _, runs), p in states.items():
        final[runs] += p

    return dict(final)


def summarize(run_dist: dict[int, float]) -> dict[str, float]:
    p0 = run_dist.get(0, 0.0)
    mean_runs = sum(k * v for k, v in run_dist.items())
    return {
        'p_scoreless': p0,
        'p_score':     1.0 - p0,
        'expected_runs': mean_runs,
        'p_1':  run_dist.get(1, 0.0),
        'p_2_plus': sum(v for k, v in run_dist.items() if k >= 2),
    }


# ═══════════════════════════════════════════════════════════════════
# Fallback: synthesize per-outcome probs from factor projection blob.
# Used for hitters where the ML classifier hasn't written probs yet.
# ═══════════════════════════════════════════════════════════════════

def probs_from_proj(proj: dict, expected_pa: float) -> dict[str, float]:
    """
    Approximate per-PA outcome probabilities from a factor projection dict.
    Split total hits into 1B/2B/3B using league-average distribution.
    """
    pa = max(expected_pa, 1)
    h = proj.get('h', 0) or 0
    hr = proj.get('hr', 0) or 0
    bb = proj.get('bb', 0) or 0
    k = proj.get('k', 0) or 0

    # Per-PA rates
    hr_rate = hr / pa
    bb_rate = bb / pa
    k_rate  = k  / pa

    # Non-HR hits split by league-typical 1B/2B/3B ratio
    non_hr_hits = max(0, h - hr)
    single_rate = (non_hr_hits * 0.78) / pa
    double_rate = (non_hr_hits * 0.20) / pa
    triple_rate = (non_hr_hits * 0.02) / pa

    # Out in play = 1 - everything else (clipped)
    rest = 1.0 - (single_rate + double_rate + triple_rate + hr_rate + bb_rate + k_rate)
    out_rate = max(0.0, rest)

    return {
        'single':       single_rate,
        'double':       double_rate,
        'triple':       triple_rate,
        'home_run':     hr_rate,
        'walk':         bb_rate,
        'hit_by_pitch': 0.0,
        'strikeout':    k_rate,
        'out':          out_rate,
    }


# ═══════════════════════════════════════════════════════════════════
# Phase-1: Park + weather modifier applied to per-batter outcome probs
# ═══════════════════════════════════════════════════════════════════

WIND_BOOST_DIRS = {'out', 'out to cf', 'out to lf', 'out to rf',
                   'out to left', 'out to right', 'out to center'}
WIND_SUPPRESS   = {'in', 'in from cf', 'in from lf', 'in from rf',
                   'in from left', 'in from right', 'in from center'}


def env_multiplier(park: dict | None, weather: dict | None) -> tuple[float, float]:
    """
    Compute (hit_mult, hr_mult) from park factors + game-time weather.
    Both multipliers are clipped; applied per-outcome inside apply_env_to_probs.
    """
    hit_mult = 1.0
    hr_mult  = 1.0

    if park:
        hit_mult *= max(0.9, min(park.get('hrFactor', 1.0) * 0.5 + park.get('hitFactor', 1.0) * 0.5, 1.1))
        hr_mult  *= max(0.75, min(park.get('hrFactor', 1.0), 1.25))

    if weather:
        temp = weather.get('tempF')
        if isinstance(temp, (int, float)):
            # Cold (<55°F) suppresses offense ~5-8%; warm (>85°F) +3%
            if temp < 55:
                hit_mult *= max(0.92, 1 - (55 - temp) * 0.006)
                hr_mult  *= max(0.85, 1 - (55 - temp) * 0.010)
            elif temp > 80:
                hit_mult *= min(1.04, 1 + (temp - 80) * 0.003)
                hr_mult  *= min(1.12, 1 + (temp - 80) * 0.006)
        ws = weather.get('windSpeedMph')
        wd = (weather.get('windDir') or '').lower()
        if isinstance(ws, (int, float)) and ws > 0:
            if wd in WIND_BOOST_DIRS:
                hr_mult *= min(1.25, 1 + ws * 0.012)
            elif wd in WIND_SUPPRESS:
                hr_mult *= max(0.80, 1 - ws * 0.010)

    return hit_mult, hr_mult


def apply_env_to_probs(probs: dict[str, float],
                       park: dict | None = None,
                       weather: dict | None = None) -> dict[str, float]:
    """
    Rescale outcome probs for park + weather, keeping sum = 1.
    HR mult applied to home_run; hit mult applied to 1B/2B/3B. Walks/Ks held.
    """
    if not park and not weather:
        return probs
    hit_mult, hr_mult = env_multiplier(park, weather)
    out = dict(probs)
    out['home_run'] = probs.get('home_run', 0.0) * hr_mult
    for k in ('single', 'double', 'triple'):
        out[k] = probs.get(k, 0.0) * hit_mult
    # Rebalance so total sums to 1 — absorb deltas into 'out' outcome
    defense_keys = ('walk', 'hit_by_pitch', 'strikeout', 'out')
    off_sum = sum(out.get(k, 0.0) for k in ('single', 'double', 'triple', 'home_run'))
    def_sum = sum(probs.get(k, 0.0) for k in defense_keys)
    target_def = max(0.02, 1.0 - off_sum)
    scale = target_def / def_sum if def_sum > 0 else 1.0
    for k in defense_keys:
        out[k] = probs.get(k, 0.0) * scale
    return out
