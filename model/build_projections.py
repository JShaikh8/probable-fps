from __future__ import annotations
"""
Daily projection engine for hitters AND starting pitchers.

Hitter model factors:
  1. Pitch-type split rates (avg/slg/k/bb/hr per pitch family)
  2. Archetype blending (similar-hitter data when split is sparse)
  3. Platoon advantage/disadvantage (+5% / -5%)
  4. Pitcher stuff quality (velocity + spin vs league avg → multiplier on rates)
  5. Recent form (hot +10%, cold -8%)
  6. Park factors (HR, hit)
  7. Weather (temp, wind direction/speed → HR multiplier)
  8. Lineup position (expected PA proxy)

Factor signals stored per projection:
  park | weather | platoon | stuffQuality | recentForm | battingOrder | matchup
  Each is -1 (negative) / 0 (neutral) / 1 (positive) for the hitter.
  factorScore: weighted composite [-1.0, +1.0] using FACTOR_WEIGHTS
    (matchup 25%, platoon 20%, stuffQuality 20%, recentForm 15%, park 10%, order 5%, weather 5%)

Pitcher model:
  Historical avgIP, avgK, avgBB, avgH, avgHR, FIP → DK/FD pts.
  Park factor (inverse direction), weather adjustments.

Output collections: mlb_projections, mlb_pitcher_projections
"""
from datetime import date, datetime
import requests
import pandas as pd
import numpy as np
from sqlalchemy import select, text

from config import get_engine, get_session, MLB_API_BASE
from db.io import bulk_upsert
from db.models import (
    AtBat, HitterPitchSplit, HitterRecentForm, HitterSimilar, HitterSprayProfile,
    ParkFactor, PitcherProfile, PitcherSeasonStats, Player, PitcherProjection,
    Projection,
)

# ── Lineup PA expectations ──────────────────────────────────────────
PA_BY_SLOT = {1: 4.7, 2: 4.6, 3: 4.5, 4: 4.4, 5: 4.2,
              6: 4.1, 7: 3.9, 8: 3.7, 9: 3.6}
DEFAULT_PA   = 4.0
MIN_PA_BLEND = 20
SIMILAR_TOP  = 5

PITCH_FAMILIES = ['fastball', 'sinker', 'cutter', 'slider', 'curveball', 'changeup']

# League-average arsenal weights (for baseline hitter computation)
LEAGUE_ARSENAL = {
    'fastball': 0.35, 'sinker': 0.15, 'cutter': 0.10,
    'slider': 0.20, 'curveball': 0.10, 'changeup': 0.10,
}

# League-average pitch velocity and spin per family (for stuff quality)
LEAGUE_AVG_SPEED = {
    'fastball': 93.5, 'sinker': 92.5, 'cutter': 88.5,
    'slider': 85.5,  'curveball': 79.5, 'changeup': 84.5,
}
LEAGUE_AVG_SPIN = {
    'fastball': 2300, 'sinker': 2150, 'cutter': 2400,
    'slider': 2400,  'curveball': 2600, 'changeup': 1800,
}

# ── Weather adjustments ─────────────────────────────────────────────
TEMP_BASELINE   = 72
TEMP_HR_PER_DEG = 0.004
WIND_BOOST_DIRS = {'out', 'out to cf', 'out to lf', 'out to rf',
                   'out to left', 'out to right', 'out to center'}
WIND_SUPPRESS   = {'in', 'in from cf', 'in from lf', 'in from rf',
                   'in from left', 'in from right', 'in from center'}
WIND_HR_PER_MPH = 0.015

# ── Multipliers ─────────────────────────────────────────────────────
PLATOON_OPP  = 1.05    # opposite hand → +5%
PLATOON_SAME = 0.95    # same hand → -5%
FORM_HOT     = 1.10    # hot streak → +10%
FORM_COLD    = 0.92    # cold streak → -8%

# ── DraftKings pitcher scoring ──────────────────────────────────────
# IP=2.25, K=2, ER=-2, H=-0.6, BB=-0.6  (W=4 omitted — too uncertain)
DK_P_IP = 2.25; DK_P_K = 2.0; DK_P_ER = -2.0; DK_P_H = -0.6; DK_P_BB = -0.6

# ── FanDuel pitcher scoring ─────────────────────────────────────────
# Outs=1 (IP×3), K=3, ER=-3, H=-0.6, BB=-0.6
FD_P_OUT = 1.0; FD_P_K = 3.0; FD_P_ER = -3.0; FD_P_H = -0.6; FD_P_BB = -0.6

LEAGUE_ERA = 4.20

# ── Factor weights (predictive importance) ──────────────────────────
# Weights sum to 1.0; reflect effect size and specificity
# factorScore = sum(signal × weight) → range [-1.0, +1.0]
FACTOR_WEIGHTS = {
    'matchup':      0.25,  # pitch-type whiff matchup — most specific
    'platoon':      0.20,  # handedness — well-established, consistent
    'stuffQuality': 0.20,  # pitcher velo+spin vs league avg
    'recentForm':   0.15,  # last 7 games — real but noisy
    'park':         0.10,  # park HR/hit factor
    'battingOrder': 0.05,  # lineup slot PA exposure
    'weather':      0.05,  # temp+wind HR adjustment
}


# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def run(game_date: str | None = None):
    if game_date is None:
        game_date = date.today().isoformat()

    print(f'Building projections for {game_date}…')

    matchups = fetch_lineups(game_date)
    if not matchups:
        print('No lineups or probable pitchers found for this date.')
        return
    print(f'  {len(matchups)} hitter/pitcher matchups')

    hitter_ids  = [m['hitterId']  for m in matchups]
    pitcher_ids = list({m['pitcherId'] for m in matchups if m.get('pitcherId')})
    venue_ids   = list({m['venueId']   for m in matchups if m.get('venueId')})

    pitcher_profiles     = _load_pitcher_profiles(pitcher_ids)
    hitter_splits        = _load_hitter_splits(hitter_ids)
    hitter_similar       = _load_similar(hitter_ids)
    park_factors         = _load_park_factors(venue_ids)
    recent_form          = _load_recent_form(hitter_ids)
    pitcher_season_stats = _load_pitcher_season_stats(pitcher_ids)
    hitter_hands         = _load_hitter_hands(hitter_ids)
    spray_profiles       = _load_spray_profiles(hitter_ids)

    for m in matchups:
        if not m.get('hitterHand'):
            m['hitterHand'] = hitter_hands.get(m['hitterId'], '')
        if not m.get('pitcherHand'):
            prof = pitcher_profiles.get(m['pitcherId'])
            if prof and prof.get('pitcherHand'):
                m['pitcherHand'] = prof['pitcherHand']

    # ── Hitter projections ─────────────────────────────────────────
    hitter_rows: list[dict] = []
    for m in matchups:
        try:
            proj = project_hitter(
                m, pitcher_profiles, hitter_splits, hitter_similar,
                park_factors, recent_form, spray_profiles,
            )
            hitter_rows.append(_to_hitter_row(proj, m, game_date))
        except Exception as e:
            print(f'  !! {m.get("hitterName")} vs {m.get("pitcherName")}: {e}')

    # ── Pitcher projections ────────────────────────────────────────
    pitcher_rows: list[dict] = []
    for pm in _extract_pitcher_matchups(matchups):
        try:
            pproj = project_pitcher(pm, pitcher_profiles, pitcher_season_stats, park_factors)
            pitcher_rows.append(_to_pitcher_row(pproj, pm, game_date))
        except Exception as e:
            print(f'  !! pitcher {pm.get("pitcherName")}: {e}')

    session = get_session()
    try:
        bulk_upsert(session, Projection, hitter_rows,
                    pk_cols=['hitter_id', 'game_pk'])
        bulk_upsert(session, PitcherProjection, pitcher_rows,
                    pk_cols=['pitcher_id', 'game_pk'])
        session.commit()
        print(f'  hitter projections:  {len(hitter_rows)}')
        print(f'  pitcher projections: {len(pitcher_rows)}')
    finally:
        session.close()

    print('Done.')


def _to_hitter_row(proj: dict, m: dict, game_date: str) -> dict:
    return {
        'hitter_id':        proj['hitterId'],
        'game_pk':          proj.get('gamePk'),
        'game_date':        game_date,
        'pitcher_id':       proj.get('pitcherId'),
        'proj':             proj.get('proj', {}),
        'dk_pts':           proj.get('dkPts', 0.0),
        'fd_pts':           proj.get('fdPts', 0.0),
        'baseline_dk_pts':  proj.get('baselineDkPts') or 0.0,
        'dk_delta':         proj.get('dkDelta') or 0.0,
        'factors':          proj.get('factors', {}),
        'factor_score':     proj.get('factorScore', 0.0),
        'contact_quality':  proj.get('contactQuality'),
        'expected_pa':      proj.get('expectedPA', 4.0),
        'lineup_slot':      proj.get('lineupSlot') or None,
        'weather':          proj.get('weather'),
        'hitter_hand':      proj.get('hitterHand'),
        'pitcher_hand':     proj.get('pitcherHand'),
        'lineup_source':    proj.get('lineupSource', 'confirmed'),
        'side':             m.get('side'),
    }


def _to_pitcher_row(proj: dict, pm: dict, game_date: str) -> dict:
    return {
        'pitcher_id':    proj['pitcherId'],
        'game_pk':       proj.get('gamePk'),
        'game_date':     game_date,
        'proj':          proj.get('proj', {}),
        'dk_pts':        proj.get('dkPts', 0.0),
        'fd_pts':        proj.get('fdPts', 0.0),
        'fip':           proj.get('fip'),
        'games_started': proj.get('gamesStarted') or 0,
        'stuff_signal':  str(proj.get('stuffSignal')) if proj.get('stuffSignal') is not None else None,
        'model_version': proj.get('modelVersion', '2.1'),
        'lineup_source': proj.get('lineupSource', 'confirmed'),
        'side':          pm.get('side'),
    }


# ═══════════════════════════════════════════════════════════════════
# HITTER PROJECTION
# ═══════════════════════════════════════════════════════════════════

def project_hitter(m: dict, pitcher_profiles: dict, hitter_splits: dict,
                   hitter_similar: dict, park_factors: dict, recent_form: dict,
                   spray_profiles: dict | None = None) -> dict:

    pitcher_id  = m['pitcherId']
    hitter_id   = m['hitterId']
    venue_id    = m.get('venueId')
    slot        = m.get('lineupSlot', 0)
    weather     = m.get('weather', {})
    hitter_hand = m.get('hitterHand', '')
    pitcher_hand = m.get('pitcherHand', '')

    expected_pa = PA_BY_SLOT.get(slot, DEFAULT_PA)

    profile = pitcher_profiles.get(pitcher_id)
    arsenal = profile.get('arsenal', {}) if profile else {}

    # ── Platoon multiplier ──────────────────────────────────────────
    platoon_mult, platoon_signal = _platoon(hitter_hand, pitcher_hand)

    # ── Pitcher stuff quality ────────────────────────────────────────
    stuff_mult, stuff_signal = _stuff_quality(arsenal)

    # ── Recent form ─────────────────────────────────────────────────
    form_doc = recent_form.get(hitter_id, {})
    form_mult, form_signal = _form_mult(form_doc)

    # ── Aggregate pitch-type rates ──────────────────────────────────
    proj_avg = proj_slg = proj_k_rate = proj_bb_rate = proj_hr_rate = 0.0
    total_weight = 0.0

    for pitch_fam, pitch_info in arsenal.items():
        usage = pitch_info.get('usagePct', 0)
        if usage < 0.03:
            continue

        split = hitter_splits.get((hitter_id, pitch_fam))
        split_pa = split.get('pa', 0) if split else 0

        if split and split_pa >= MIN_PA_BLEND:
            h_avg  = split.get('avg', 0.250) or 0.250
            h_slg  = split.get('slg', 0.380) or 0.380
            h_k    = split.get('k', 0) / max(split.get('pa', 1), 1)
            h_bb   = split.get('bb', 0) / max(split.get('pa', 1), 1)
            h_hr   = split.get('hr', 0) / max(split.get('ab', 1), 1)
            blend_w = 1.0
        else:
            h_avg, h_slg, h_k, h_bb, h_hr, blend_w = _blend_with_similar(
                hitter_id, pitch_fam, hitter_splits, hitter_similar, split
            )

        w = usage * blend_w
        proj_avg     += h_avg  * w
        proj_slg     += h_slg  * w
        proj_k_rate  += h_k    * w
        proj_bb_rate += h_bb   * w
        proj_hr_rate += h_hr   * w
        total_weight += w

    # Accumulate contact quality (launch angle, exit velo) weighted by pitch usage
    proj_launch_angle = 0.0
    proj_exit_velo    = 0.0
    la_weight = 0.0
    for pitch_fam, pitch_info in arsenal.items():
        usage = pitch_info.get('usagePct', 0)
        if usage < 0.03:
            continue
        split = hitter_splits.get((hitter_id, pitch_fam))
        if split:
            la = split.get('avgLaunchAngle')
            ev = split.get('avgExitVelo')
            if la is not None:
                proj_launch_angle += la * usage
                la_weight += usage
            if ev is not None:
                proj_exit_velo += ev * usage

    if la_weight > 0:
        proj_launch_angle /= la_weight
        proj_exit_velo    = proj_exit_velo / la_weight if la_weight > 0 else None
    else:
        proj_launch_angle = None
        proj_exit_velo    = None

    if total_weight > 0:
        proj_avg     /= total_weight
        proj_slg     /= total_weight
        proj_k_rate  /= total_weight
        proj_bb_rate /= total_weight
        proj_hr_rate /= total_weight
    else:
        proj_avg, proj_slg, proj_k_rate, proj_bb_rate, proj_hr_rate = 0.248, 0.400, 0.22, 0.09, 0.034

    # ── Apply platoon multiplier ────────────────────────────────────
    proj_avg     *= platoon_mult
    proj_slg     *= platoon_mult
    proj_hr_rate *= platoon_mult
    proj_k_rate  *= (2 - platoon_mult)   # inverse: platoon disadvantage raises K rate

    # ── Apply pitcher stuff quality (tough pitcher → worse for hitter) ──
    proj_avg     *= stuff_mult
    proj_slg     *= stuff_mult
    proj_hr_rate *= stuff_mult

    # ── Apply recent form ───────────────────────────────────────────
    proj_avg     *= form_mult
    proj_slg     *= form_mult
    proj_hr_rate *= form_mult

    # ── Launch angle HR adjustment ───────────────────────────────────
    # Ideal HR launch angle: 25–35°. Penalize flat (<10°) or extreme pop-up (>45°).
    # Reward hitters in the optimal window vs this pitcher's arsenal.
    la_mult = 1.0
    if proj_launch_angle is not None:
        la = proj_launch_angle
        if 25 <= la <= 35:
            la_mult = 1.06   # ideal power window
        elif 20 <= la < 25 or 35 < la <= 42:
            la_mult = 1.02   # near-optimal
        elif la < 10:
            la_mult = 0.90   # groundball tendency vs this arsenal — suppresses HR
        elif la > 45:
            la_mult = 0.94   # extreme pop-up tendency
    # Also boost slg for hard contact (exit velo > 91 mph avg)
    ev_slg_mult = 1.0
    if proj_exit_velo is not None and proj_exit_velo > 91:
        ev_slg_mult = 1.0 + (proj_exit_velo - 91) * 0.005   # ~0.5% per mph above 91

    proj_hr_rate *= la_mult
    proj_slg     *= ev_slg_mult

    # ── Park factors ────────────────────────────────────────────────
    pf = park_factors.get(venue_id, {})
    hr_factor        = pf.get('hrFactor',       1.0)
    hit_factor       = pf.get('hitFactor',      1.0)
    hard_hit_factor  = pf.get('hardHitFactor',  1.0)
    k_factor         = pf.get('kFactor',        1.0)

    # Spray × park directional adjustment:
    #   If hitter is a pull-heavy hitter and park's pull_deep zone is above average,
    #   apply a small positive HR multiplier (and vice versa for oppo hitters).
    spray = (spray_profiles or {}).get(hitter_id, {})
    spray_park_mult = 1.0
    if spray:
        pull_pct   = spray.get('pullPct',  0.33)
        oppo_pct   = spray.get('oppoPct',  0.33)
        park_locs  = pf.get('hitLocations', {})
        # pull_deep and oppo_deep are fractions of all BIP in the park
        # League average deep pull ≈ 0.12–0.15; higher = park favors pull power
        pull_deep  = park_locs.get('pull_deep',  0.13)
        oppo_deep  = park_locs.get('oppo_deep',  0.10)
        # Directional tendency score: pull-heavy hitter gets weighted toward pull_deep
        directional_score = (pull_pct - 0.33) * (pull_deep - 0.13) * 20 + \
                            (oppo_pct - 0.33) * (oppo_deep - 0.10) * 20
        spray_park_mult = max(0.96, min(1 + directional_score * 0.02, 1.04))

    park_signal = (1 if hr_factor > 1.05 or hit_factor > 1.05
                   else -1 if hr_factor < 0.95 and hit_factor < 0.97
                   else 0)
    proj_avg     *= hit_factor
    proj_slg     *= hit_factor * hard_hit_factor   # hard hit factor boosts XBH potential
    proj_hr_rate *= hr_factor * spray_park_mult
    proj_k_rate  *= k_factor                       # pitcher-friendly park → more Ks

    # ── Weather ─────────────────────────────────────────────────────
    temp    = weather.get('tempF')
    w_speed = weather.get('windSpeedMph')
    w_dir   = (weather.get('windDir', '') or '').lower()
    weather_mult = 1.0

    if temp is not None:
        weather_mult *= max(0.5, min(1 + (temp - TEMP_BASELINE) * TEMP_HR_PER_DEG, 1.6))

    if w_speed is not None and w_speed > 0:
        if w_dir in WIND_BOOST_DIRS:
            weather_mult *= (1 + w_speed * WIND_HR_PER_MPH)
        elif w_dir in WIND_SUPPRESS:
            weather_mult *= (1 - w_speed * WIND_HR_PER_MPH * 0.7)

    weather_signal = (1 if weather_mult > 1.05 else -1 if weather_mult < 0.95 else 0)
    proj_hr_rate *= weather_mult
    proj_hr_rate  = max(0, proj_hr_rate)

    # ── Matchup signal ───────────────────────────────────────────────
    matchup_signal = _matchup_signal(hitter_id, hitter_splits, arsenal)

    # ── Batting order signal ─────────────────────────────────────────
    batting_order_signal = (1 if slot in (1, 2, 3) else -1 if slot in (7, 8, 9) else 0)

    # ── Scale to expected PA ─────────────────────────────────────────
    proj_ab  = expected_pa * (1 - proj_bb_rate)
    proj_h   = proj_ab * proj_avg
    proj_hr  = proj_ab * proj_hr_rate
    proj_k   = proj_ab * proj_k_rate
    proj_bb  = expected_pa * proj_bb_rate
    proj_rbi = proj_hr * 1.4 + (proj_h - proj_hr) * 0.35
    proj_r   = proj_h * 0.4 + proj_bb * 0.25

    dk_pts = round(
        (proj_h - proj_hr) * 3.0 +
        proj_hr  * 10.0 +
        proj_bb  * 2.0 +
        proj_r   * 2.0 +
        proj_rbi * 2.0 +
        proj_k   * (-0.5),
        2
    )
    fd_pts = round(
        (proj_h - proj_hr) * 3.0 +
        proj_hr  * 12.0 +
        proj_bb  * 3.0 +
        proj_r   * 3.2 +
        proj_rbi * 3.5,
        2
    )

    # ── Baseline DK (neutral day) ────────────────────────────────────
    baseline_dk = _baseline_dk(hitter_id, hitter_splits, expected_pa)
    dk_delta    = round(dk_pts - baseline_dk, 2) if baseline_dk is not None else None

    return {
        'hitterId':    hitter_id,
        'hitterName':  m.get('hitterName', ''),
        'hitterHand':  hitter_hand,
        'pitcherId':   pitcher_id,
        'pitcherName': m.get('pitcherName', ''),
        'pitcherHand': pitcher_hand,
        'gamePk':      m.get('gamePk'),
        'venueId':     venue_id,
        'venueName':   m.get('venueName', ''),
        'side':        m.get('side', ''),
        'homeAbbr':    m.get('homeAbbr', ''),
        'awayAbbr':    m.get('awayAbbr', ''),
        'lineupSlot':  slot,
        'expectedPA':  round(expected_pa, 2),
        'weather':     weather,
        'proj': {
            'h':   round(proj_h,   2),
            'hr':  round(proj_hr,  2),
            'bb':  round(proj_bb,  2),
            'k':   round(proj_k,   2),
            'r':   round(proj_r,   2),
            'rbi': round(proj_rbi, 2),
            'avg': round(proj_avg, 4),
            'slg': round(proj_slg, 4),
        },
        'dkPts':        dk_pts,
        'fdPts':        fd_pts,
        'baselineDkPts': round(baseline_dk, 2) if baseline_dk is not None else None,
        'dkDelta':      dk_delta,
        'contactQuality': {
            'avgLaunchAngle': round(proj_launch_angle, 1) if proj_launch_angle is not None else None,
            'avgExitVelo':    round(proj_exit_velo, 1)    if proj_exit_velo    is not None else None,
        },
        'factors': {
            'park':         park_signal,
            'weather':      weather_signal,
            'platoon':      platoon_signal,
            'stuffQuality': stuff_signal,
            'recentForm':   form_signal,
            'battingOrder': batting_order_signal,
            'matchup':      matchup_signal,
        },
        'factorScore': round(
            park_signal         * FACTOR_WEIGHTS['park'] +
            weather_signal      * FACTOR_WEIGHTS['weather'] +
            platoon_signal      * FACTOR_WEIGHTS['platoon'] +
            stuff_signal        * FACTOR_WEIGHTS['stuffQuality'] +
            form_signal         * FACTOR_WEIGHTS['recentForm'] +
            batting_order_signal * FACTOR_WEIGHTS['battingOrder'] +
            matchup_signal      * FACTOR_WEIGHTS['matchup'],
            3
        ),
        'formSignal':   form_doc.get('formSignal', 'normal'),
        'modelVersion': '2.1',
        'dataQuality':  _data_quality_flag(m, hitter_splits, arsenal),
        'lineupSource': m.get('lineupSource', 'confirmed'),
    }


# ═══════════════════════════════════════════════════════════════════
# PITCHER PROJECTION
# ═══════════════════════════════════════════════════════════════════

def project_pitcher(pm: dict, pitcher_profiles: dict, pitcher_season_stats: dict,
                    park_factors: dict) -> dict:
    pitcher_id   = pm['pitcherId']
    venue_id     = pm.get('venueId')
    weather      = pm.get('weather', {})
    pitcher_hand = pm.get('pitcherHand', '')

    # Season stats (most recent season)
    stats = pitcher_season_stats.get(pitcher_id)
    if not stats:
        # Fallback: league average starter
        avg_ip, avg_k, avg_bb, avg_h, avg_hr, fip = 5.5, 5.5, 2.0, 5.5, 0.7, LEAGUE_ERA
    else:
        avg_ip = stats.get('avgIP', 5.5)
        avg_k  = stats.get('avgK',  5.5)
        avg_bb = stats.get('avgBB', 2.0)
        avg_h  = stats.get('avgH',  5.5)
        avg_hr = stats.get('avgHR', 0.7)
        fip    = stats.get('fip') or LEAGUE_ERA

    # Projected ER from FIP (FIP ≈ ERA in the long run)
    proj_er = fip / 9.0 * avg_ip

    # ── Park factor (inverse for pitcher: hitter-friendly = worse) ──
    pf = park_factors.get(venue_id, {})
    hr_factor       = pf.get('hrFactor',      1.0)
    hit_factor      = pf.get('hitFactor',     1.0)
    hard_hit_factor = pf.get('hardHitFactor', 1.0)
    k_factor        = pf.get('kFactor',       1.0)
    avg_h  *= hit_factor
    avg_hr *= hr_factor
    avg_k  *= k_factor                                  # pitcher-friendly park → more Ks
    proj_er *= (hr_factor * 0.4 + hit_factor * 0.4 + hard_hit_factor * 0.2)

    # ── Weather (wind in suppresses HR, helps pitcher) ───────────────
    temp    = weather.get('tempF')
    w_speed = weather.get('windSpeedMph')
    w_dir   = (weather.get('windDir', '') or '').lower()
    weather_mult = 1.0

    if temp is not None:
        # colder = fewer HR = better for pitcher → inverse
        weather_mult *= max(0.6, min(1 + (TEMP_BASELINE - temp) * TEMP_HR_PER_DEG, 1.5))

    if w_speed is not None and w_speed > 0:
        if w_dir in WIND_SUPPRESS:       # wind in = fewer HR = better for pitcher
            weather_mult *= (1 + w_speed * WIND_HR_PER_MPH * 0.5)
        elif w_dir in WIND_BOOST_DIRS:   # wind out = more HR = worse for pitcher
            weather_mult *= (1 - w_speed * WIND_HR_PER_MPH * 0.5)

    avg_hr  *= (2 - weather_mult)     # inverse: good weather for pitcher → fewer HR
    proj_er *= (2 - weather_mult)
    avg_hr  = max(0, avg_hr)
    proj_er = max(0, proj_er)

    # ── Stuff quality bonus (elite stuff → lower H/BB/ER allowed) ───
    profile = pitcher_profiles.get(pitcher_id)
    arsenal = profile.get('arsenal', {}) if profile else {}
    _, stuff_signal = _stuff_quality(arsenal)
    if stuff_signal == 1:       # elite stuff
        proj_er *= 0.92
        avg_h   *= 0.93
    elif stuff_signal == -1:    # below average stuff
        proj_er *= 1.08
        avg_h   *= 1.06

    # ── DK pts ───────────────────────────────────────────────────────
    dk_pts = round(
        avg_ip   * DK_P_IP +
        avg_k    * DK_P_K  +
        proj_er  * DK_P_ER +
        avg_h    * DK_P_H  +
        avg_bb   * DK_P_BB,
        2
    )
    # FD pts (outs-based)
    fd_pts = round(
        avg_ip * 3 * FD_P_OUT +
        avg_k        * FD_P_K  +
        proj_er      * FD_P_ER +
        avg_h        * FD_P_H  +
        avg_bb       * FD_P_BB,
        2
    )

    return {
        'pitcherId':   pitcher_id,
        'pitcherName': pm.get('pitcherName', ''),
        'pitcherHand': pitcher_hand,
        'gamePk':      pm.get('gamePk'),
        'venueId':     venue_id,
        'venueName':   pm.get('venueName', ''),
        'side':        pm.get('side', ''),
        'homeAbbr':    pm.get('homeAbbr', ''),
        'awayAbbr':    pm.get('awayAbbr', ''),
        'weather':     weather,
        'proj': {
            'ip':  round(avg_ip, 2),
            'k':   round(avg_k,  2),
            'bb':  round(avg_bb, 2),
            'h':   round(avg_h,  2),
            'hr':  round(avg_hr, 3),
            'er':  round(proj_er, 2),
        },
        'dkPts':        dk_pts,
        'fdPts':        fd_pts,
        'fip':          round(fip, 2),
        'gamesStarted': stats.get('gamesStarted') if stats else None,
        'stuffSignal':  stuff_signal,
        'modelVersion': '2.1',
        'dataQuality':  'high' if stats and stats.get('gamesStarted', 0) >= 5 else 'low',
        'lineupSource': pm.get('lineupSource', 'confirmed'),
    }


# ═══════════════════════════════════════════════════════════════════
# FACTOR HELPERS
# ═══════════════════════════════════════════════════════════════════

def _platoon(hitter_hand: str, pitcher_hand: str):
    """Returns (multiplier, signal) where signal = 1 (platoon advantage), -1 (disadvantage), 0."""
    if not hitter_hand or not pitcher_hand:
        return 1.0, 0
    # Batter has advantage when hands are opposite
    if hitter_hand != pitcher_hand:
        return PLATOON_OPP, 1
    return PLATOON_SAME, -1


def _stuff_quality(arsenal: dict):
    """
    Computes a pitcher stuff quality multiplier based on velocity + spin
    relative to league averages.

    Returns (multiplier, signal):
      multiplier: applied to hitter rates (>1 means pitcher below avg = easier for hitter)
      signal: +1 = below-avg stuff (hitter advantage), -1 = elite stuff, 0 = neutral
    """
    if not arsenal:
        return 1.0, 0

    score = 0.0
    total_usage = 0.0

    for fam, info in arsenal.items():
        usage = info.get('usagePct', 0)
        if usage < 0.03:
            continue
        lg_speed = LEAGUE_AVG_SPEED.get(fam, 88.0)
        lg_spin  = LEAGUE_AVG_SPIN.get(fam, 2200)

        speed = info.get('avgSpeed')
        spin  = info.get('avgSpin')

        # Delta above/below league average (normalized)
        d_speed = ((speed - lg_speed) / 3.0) if speed is not None else 0.0
        d_spin  = ((spin  - lg_spin)  / 200.0) if spin is not None else 0.0

        # Positive score = pitcher above avg = harder for hitter
        pitch_score = d_speed * 0.6 + d_spin * 0.4
        score      += pitch_score * usage
        total_usage += usage

    if total_usage > 0:
        score /= total_usage
    else:
        return 1.0, 0

    # score > 0 = pitcher above avg = multiplier < 1 (worse for hitter)
    # Apply ±10% max from stuff quality
    mult = max(0.90, min(1 - score * 0.05, 1.10))
    signal = -1 if score > 0.5 else 1 if score < -0.5 else 0
    return round(mult, 4), signal


def _form_mult(form_doc: dict):
    """Returns (multiplier, signal) from hitter recent form."""
    sig = form_doc.get('formSignal', 'normal')
    if sig == 'hot':
        return FORM_HOT, 1
    if sig == 'cold':
        return FORM_COLD, -1
    return 1.0, 0


def _matchup_signal(hitter_id: int, hitter_splits: dict, arsenal: dict) -> int:
    """
    Signal based on hitter's whiff rate vs pitcher's primary pitch family.
    -1: hitter struggles with pitcher's best pitch
     0: neutral
    +1: hitter handles pitcher's primary pitch well
    """
    if not arsenal:
        return 0

    top_fam = max(arsenal, key=lambda k: arsenal[k].get('usagePct', 0))
    split   = hitter_splits.get((hitter_id, top_fam))
    if not split:
        return 0

    whiff = split.get('whiffPct')
    if whiff is None:
        return 0

    # League avg whiff ≈ 0.24; well above → bad matchup, well below → good
    if whiff > 0.30:
        return -1
    if whiff < 0.18:
        return 1
    return 0


def _baseline_dk(hitter_id: int, hitter_splits: dict, expected_pa: float):
    """
    Estimate a neutral-day DK pts baseline using league-average arsenal weights.
    Returns None if insufficient data.
    """
    base_avg = base_slg = base_k = base_bb = base_hr = 0.0
    total_w = 0.0

    for fam, usage in LEAGUE_ARSENAL.items():
        split = hitter_splits.get((hitter_id, fam))
        if not split or split.get('pa', 0) < 5:
            continue
        base_avg += (split.get('avg', 0) or 0) * usage
        base_slg += (split.get('slg', 0) or 0) * usage
        base_k   += (split.get('k', 0) / max(split.get('pa', 1), 1)) * usage
        base_bb  += (split.get('bb', 0) / max(split.get('pa', 1), 1)) * usage
        base_hr  += (split.get('hr', 0) / max(split.get('ab', 1), 1)) * usage
        total_w  += usage

    if total_w < 0.30:
        return None

    base_avg /= total_w; base_slg /= total_w
    base_k   /= total_w; base_bb  /= total_w; base_hr /= total_w

    ab   = expected_pa * (1 - base_bb)
    h    = ab * base_avg
    hr   = ab * base_hr
    k    = ab * base_k
    bb   = expected_pa * base_bb
    rbi  = hr * 1.4 + (h - hr) * 0.35
    r    = h * 0.4 + bb * 0.25

    return (h - hr) * 3.0 + hr * 10.0 + bb * 2.0 + r * 2.0 + rbi * 2.0 + k * (-0.5)


# ═══════════════════════════════════════════════════════════════════
# BLENDING / QUALITY HELPERS
# ═══════════════════════════════════════════════════════════════════

def _blend_with_similar(hitter_id, pitch_fam, hitter_splits, hitter_similar, own_split):
    similar = hitter_similar.get(hitter_id, {}).get('similar', [])[:SIMILAR_TOP]
    vals = {'avg': [], 'slg': [], 'k_rate': [], 'bb_rate': [], 'hr_rate': [], 'weights': []}

    if own_split and own_split.get('pa', 0) > 0:
        pa = own_split['pa']
        vals['avg'].append(own_split.get('avg', 0.250) or 0.250)
        vals['slg'].append(own_split.get('slg', 0.380) or 0.380)
        vals['k_rate'].append(own_split.get('k', 0) / max(pa, 1))
        vals['bb_rate'].append(own_split.get('bb', 0) / max(pa, 1))
        vals['hr_rate'].append(own_split.get('hr', 0) / max(own_split.get('ab', 1), 1))
        vals['weights'].append(pa / MIN_PA_BLEND)

    for sim in similar:
        s_id  = sim['hitterId']
        s_sim = sim['similarity']
        s_split = hitter_splits.get((s_id, pitch_fam))
        if not s_split or s_split.get('pa', 0) < 5:
            continue
        pa = s_split['pa']
        vals['avg'].append(s_split.get('avg', 0.250) or 0.250)
        vals['slg'].append(s_split.get('slg', 0.380) or 0.380)
        vals['k_rate'].append(s_split.get('k', 0) / max(pa, 1))
        vals['bb_rate'].append(s_split.get('bb', 0) / max(pa, 1))
        vals['hr_rate'].append(s_split.get('hr', 0) / max(s_split.get('ab', 1), 1))
        vals['weights'].append(s_sim * (pa / 50))

    if not vals['weights']:
        return 0.248, 0.400, 0.220, 0.090, 0.034, 0.5

    w = np.array(vals['weights'])
    w = w / w.sum()

    return (
        float(np.dot(vals['avg'],     w)),
        float(np.dot(vals['slg'],     w)),
        float(np.dot(vals['k_rate'],  w)),
        float(np.dot(vals['bb_rate'], w)),
        float(np.dot(vals['hr_rate'], w)),
        min(1.0, float(w.sum())),
    )


def _data_quality_flag(m, hitter_splits, arsenal) -> str:
    if not arsenal:
        return 'low'
    top_pitch = max(arsenal, key=lambda k: arsenal[k].get('usagePct', 0))
    split = hitter_splits.get((m['hitterId'], top_pitch))
    if not split:
        return 'low'
    pa = split.get('pa', 0)
    if pa >= 40:
        return 'high'
    if pa >= 15:
        return 'medium'
    return 'low'


def _extract_pitcher_matchups(matchups: list[dict]) -> list[dict]:
    """Extract unique per-game starters from the hitter matchups list."""
    seen = set()
    result = []
    for m in matchups:
        key = (m['pitcherId'], m['gamePk'])
        if key in seen:
            continue
        seen.add(key)
        result.append({
            'pitcherId':   m['pitcherId'],
            'pitcherName': m['pitcherName'],
            'pitcherHand': m['pitcherHand'],
            'gamePk':      m['gamePk'],
            'venueId':     m['venueId'],
            'venueName':   m['venueName'],
            'side':        m['side'],
            'homeAbbr':    m['homeAbbr'],
            'awayAbbr':    m['awayAbbr'],
            'weather':     m['weather'],
            'lineupSource': m.get('lineupSource', 'confirmed'),
        })
    return result


# ═══════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════

def _q(sql: str, ids: list):
    """Run a parametrized SQL with an :ids bind param, return DataFrame."""
    engine = get_engine()
    stmt = text(sql)
    with engine.connect() as conn:
        return pd.read_sql_query(stmt, conn, params={'ids': list(ids)})


def _load_pitcher_profiles(pitcher_ids: list) -> dict:
    if not pitcher_ids:
        return {}
    df = _q(
        """
        SELECT DISTINCT ON (pp.pitcher_id)
               pp.pitcher_id, pp.season, pp.arsenal, pp.total_pitches,
               pp.k_pct, pp.primary_pitch, p.pitch_hand
        FROM pitcher_profiles pp
        LEFT JOIN players p ON p.player_id = pp.pitcher_id
        WHERE pp.pitcher_id = ANY(:ids)
        ORDER BY pp.pitcher_id, pp.season DESC
        """,
        pitcher_ids,
    )
    return {
        int(r['pitcher_id']): {
            'pitcherId':   int(r['pitcher_id']),
            'season':      int(r['season']),
            'arsenal':     r['arsenal'] or {},
            'totalPitches': int(r['total_pitches'] or 0),
            'kPct':        float(r['k_pct'] or 0),
            'primaryPitch': r['primary_pitch'],
            'pitcherHand': r['pitch_hand'] or '',
        }
        for _, r in df.iterrows()
    }


def _load_hitter_splits(hitter_ids: list) -> dict:
    if not hitter_ids:
        return {}
    df = _q(
        """
        SELECT DISTINCT ON (hitter_id, pitch_family)
               hitter_id, pitch_family, season,
               pa, ab, hits, hr, bb, k,
               avg, slg, obp, ops,
               swing_pct, whiff_pct, hard_hit_pct,
               high_velo_whiff_pct, high_spin_whiff_pct,
               avg_exit_velo, avg_launch_angle
        FROM hitter_pitch_splits
        WHERE hitter_id = ANY(:ids)
        ORDER BY hitter_id, pitch_family, season DESC
        """,
        hitter_ids,
    )
    out: dict = {}
    for _, r in df.iterrows():
        key = (int(r['hitter_id']), r['pitch_family'])
        out[key] = {
            'hitterId':     int(r['hitter_id']),
            'pitchFamily':  r['pitch_family'],
            'season':       int(r['season']),
            'pa': int(r['pa']), 'ab': int(r['ab']), 'hits': int(r['hits']),
            'hr': int(r['hr']), 'bb': int(r['bb']), 'k': int(r['k']),
            'avg': float(r['avg'] or 0), 'slg': float(r['slg'] or 0),
            'obp': float(r['obp'] or 0), 'ops': float(r['ops'] or 0),
            'swingPct':    _fv(r['swing_pct']),
            'whiffPct':    _fv(r['whiff_pct']),
            'hardHitPct':  _fv(r['hard_hit_pct']),
            'highVeloWhiffPct': _fv(r['high_velo_whiff_pct']),
            'highSpinWhiffPct': _fv(r['high_spin_whiff_pct']),
            'avgExitVelo':     _fv(r['avg_exit_velo']),
            'avgLaunchAngle':  _fv(r['avg_launch_angle']),
        }
    return out


def _load_similar(hitter_ids: list) -> dict:
    if not hitter_ids:
        return {}
    df = _q(
        """
        SELECT hitter_id, similar_list
        FROM hitter_similar
        WHERE hitter_id = ANY(:ids)
        """,
        hitter_ids,
    )
    out: dict = {}
    for _, r in df.iterrows():
        sim_list = r['similar_list'] or []
        # remap snake_case → camelCase for business-logic consumers
        mapped = [
            {'hitterId': int(s.get('hitter_id') or s.get('hitterId')),
             'hitterName': s.get('hitter_name') or s.get('hitterName', ''),
             'similarity': float(s.get('similarity', 0.0))}
            for s in sim_list
        ]
        out[int(r['hitter_id'])] = {'hitterId': int(r['hitter_id']), 'similar': mapped}
    return out


def _load_park_factors(venue_ids: list) -> dict:
    if not venue_ids:
        return {}
    df = _q(
        """
        SELECT venue_id, hr_factor, hit_factor, hard_hit_factor,
               k_factor, bb_factor, sample_size, hit_locations
        FROM park_factors
        WHERE venue_id = ANY(:ids)
        """,
        venue_ids,
    )
    return {
        int(r['venue_id']): {
            'venueId':        int(r['venue_id']),
            'hrFactor':       float(r['hr_factor'] or 1.0),
            'hitFactor':      float(r['hit_factor'] or 1.0),
            'hardHitFactor':  float(r['hard_hit_factor'] or 1.0),
            'kFactor':        float(r['k_factor'] or 1.0),
            'bbFactor':       float(r['bb_factor'] or 1.0),
            'sampleSize':     int(r['sample_size'] or 0),
            'hitLocations':   r['hit_locations'] or {},
        }
        for _, r in df.iterrows()
    }


def _load_recent_form(hitter_ids: list) -> dict:
    if not hitter_ids:
        return {}
    df = _q(
        """
        SELECT hitter_id, form_signal, form_ratio, last_7
        FROM hitter_recent_form
        WHERE hitter_id = ANY(:ids)
        """,
        hitter_ids,
    )
    return {
        int(r['hitter_id']): {
            'hitterId':   int(r['hitter_id']),
            'formSignal': r['form_signal'] or 'normal',
            'formRatio':  float(r['form_ratio'] or 1.0),
            'last7':      r['last_7'] or {},
        }
        for _, r in df.iterrows()
    }


def _load_hitter_hands(hitter_ids: list) -> dict:
    if not hitter_ids:
        return {}
    df = _q(
        """
        SELECT DISTINCT ON (hitter_id) hitter_id, hitter_side
        FROM at_bats
        WHERE hitter_id = ANY(:ids)
          AND hitter_side IS NOT NULL
        ORDER BY hitter_id, season DESC
        """,
        hitter_ids,
    )
    return {int(r['hitter_id']): r['hitter_side'] for _, r in df.iterrows()}


def _load_spray_profiles(hitter_ids: list) -> dict:
    if not hitter_ids:
        return {}
    df = _q(
        """
        SELECT hitter_id, pull_pct, center_pct, oppo_pct,
               deep_pct, hr_pull_pct, avg_exit_velo, avg_launch_angle
        FROM hitter_spray_profiles
        WHERE hitter_id = ANY(:ids)
        """,
        hitter_ids,
    )
    return {
        int(r['hitter_id']): {
            'hitterId':       int(r['hitter_id']),
            'pullPct':        float(r['pull_pct'] or 0),
            'centerPct':      float(r['center_pct'] or 0),
            'oppoPct':        float(r['oppo_pct'] or 0),
            'deepPct':        float(r['deep_pct'] or 0),
            'hrPullPct':      float(r['hr_pull_pct'] or 0),
            'avgExitVelo':    _fv(r['avg_exit_velo']),
            'avgLaunchAngle': _fv(r['avg_launch_angle']),
        }
        for _, r in df.iterrows()
    }


def _load_pitcher_season_stats(pitcher_ids: list) -> dict:
    if not pitcher_ids:
        return {}
    df = _q(
        """
        SELECT DISTINCT ON (pitcher_id)
               pitcher_id, season, avg_ip, avg_k, avg_bb, avg_h, avg_hr,
               fip, games_started
        FROM pitcher_season_stats
        WHERE pitcher_id = ANY(:ids)
        ORDER BY pitcher_id, season DESC
        """,
        pitcher_ids,
    )
    return {
        int(r['pitcher_id']): {
            'pitcherId':    int(r['pitcher_id']),
            'season':       int(r['season']),
            'avgIP':        float(r['avg_ip'] or 0),
            'avgK':         float(r['avg_k'] or 0),
            'avgBB':        float(r['avg_bb'] or 0),
            'avgH':         float(r['avg_h'] or 0),
            'avgHR':        float(r['avg_hr'] or 0),
            'fip':          float(r['fip']) if r['fip'] is not None else None,
            'gamesStarted': int(r['games_started'] or 0),
        }
        for _, r in df.iterrows()
    }


def _fv(v):
    """Float-or-None, handling pandas NaN."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return float(v)


# ═══════════════════════════════════════════════════════════════════
# LINEUP FETCHING
# ═══════════════════════════════════════════════════════════════════

def fetch_lineups(game_date: str) -> list[dict]:
    """Fetch confirmed lineups + probable pitchers from MLB schedule API.

    Falls back to the most recent completed game's batting order for each team
    when no confirmed lineups are available (e.g. early morning, off-season).

    Side-effect: upserts today's games/teams/venues into Postgres so the UI
    can join projections → games → teams/venues cleanly.
    """
    url = (
        f'{MLB_API_BASE}/schedule'
        f'?sportId=1&date={game_date}&gameType=R'
        f'&hydrate=lineups,probablePitcher,weather,venue,team'
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Upsert schedule metadata before returning matchups
    _upsert_todays_schedule(data, game_date)

    matchups = []
    fallback_candidates = []

    for date_entry in data.get('dates', []):
        for game in date_entry.get('games', []):
            game_pk    = game['gamePk']
            venue      = game.get('venue', {})
            venue_id   = venue.get('id')
            venue_name = venue.get('name', '')
            weather    = _parse_weather(game.get('weather', {}))

            home_abbr = game['teams']['home'].get('team', {}).get('abbreviation', '')
            away_abbr = game['teams']['away'].get('team', {}).get('abbreviation', '')

            for side in ('home', 'away'):
                team_data = game['teams'][side]
                team_id   = team_data.get('team', {}).get('id')
                pitcher   = team_data.get('probablePitcher', {})
                pitcher_id   = pitcher.get('id')
                pitcher_name = pitcher.get('fullName', '')

                lineup = game.get('lineups', {}).get(f'{side}Players', [])
                if lineup:
                    for slot_idx, player in enumerate(lineup[:9], start=1):
                        matchups.append({
                            'gamePk':       game_pk,
                            'venueId':      venue_id,
                            'venueName':    venue_name,
                            'weather':      weather,
                            'side':         side,
                            'homeAbbr':     home_abbr,
                            'awayAbbr':     away_abbr,
                            'hitterId':     player.get('id'),
                            'hitterName':   player.get('fullName', ''),
                            'hitterHand':   player.get('batSide', {}).get('code', ''),
                            'pitcherId':    pitcher_id,
                            'pitcherName':  pitcher_name,
                            'pitcherHand':  pitcher.get('pitchHand', {}).get('code', ''),
                            'lineupSlot':   slot_idx,
                            'lineupSource': 'confirmed',
                        })
                elif pitcher_id and team_id:
                    fallback_candidates.append({
                        'gamePk':      game_pk,
                        'venueId':     venue_id,
                        'venueName':   venue_name,
                        'weather':     weather,
                        'side':        side,
                        'homeAbbr':    home_abbr,
                        'awayAbbr':    away_abbr,
                        'teamId':      team_id,
                        'pitcher':     pitcher,
                        'pitcherId':   pitcher_id,
                        'pitcherName': pitcher_name,
                        'pitcherHand': pitcher.get('pitchHand', {}).get('code', ''),
                    })

    confirmed = [m for m in matchups if m['hitterId'] and m['pitcherId']]

    # Games that already have a confirmed lineup — don't fall back for these
    confirmed_game_pks = {m['gamePk'] for m in confirmed}

    # For games with a probable pitcher but no confirmed lineup, use recent batting orders
    needs_fallback = [c for c in fallback_candidates if c['gamePk'] not in confirmed_game_pks]

    if needs_fallback:
        label = 'some games' if confirmed else 'all games'
        print(f'  No confirmed lineups for {label} ({len(needs_fallback)} sides). Using fallback batting orders...')
        fallback = _fallback_from_recent_lineups(needs_fallback, game_date)
        confirmed.extend(fallback)

    return confirmed


def _fallback_from_recent_lineups(candidates: list[dict], game_date: str) -> list[dict]:
    from datetime import date as date_cls, timedelta
    target = date_cls.fromisoformat(game_date)
    start  = (target - timedelta(days=210)).isoformat()
    end    = (target - timedelta(days=1)).isoformat()

    matchups = []
    for cand in candidates:
        team_id       = cand['teamId']
        recent_lineup = _fetch_recent_team_lineup(team_id, start, end)
        if not recent_lineup:
            print(f'    No recent lineup found for team {team_id}')
            continue

        for slot_idx, player in enumerate(recent_lineup[:9], start=1):
            matchups.append({
                'gamePk':       cand['gamePk'],
                'venueId':      cand['venueId'],
                'venueName':    cand['venueName'],
                'weather':      cand['weather'],
                'side':         cand['side'],
                'homeAbbr':     cand['homeAbbr'],
                'awayAbbr':     cand['awayAbbr'],
                'hitterId':     player.get('id'),
                'hitterName':   player.get('fullName', ''),
                'hitterHand':   player.get('batSide', {}).get('code', ''),
                'pitcherId':    cand['pitcherId'],
                'pitcherName':  cand['pitcherName'],
                'pitcherHand':  cand['pitcherHand'],
                'lineupSlot':   slot_idx,
                'lineupSource': 'fallback',
            })

    return [m for m in matchups if m['hitterId'] and m['pitcherId']]


def _fetch_recent_team_lineup(team_id: int, start: str, end: str) -> list[dict]:
    url = (
        f'{MLB_API_BASE}/schedule'
        f'?sportId=1&teamId={team_id}'
        f'&startDate={start}&endDate={end}'
        f'&gameType=R&hydrate=lineups,team'
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f'    Warning: could not fetch recent lineup for team {team_id}: {e}')
        return []

    for date_entry in reversed(data.get('dates', [])):
        for game in date_entry.get('games', []):
            if game.get('status', {}).get('abstractGameState', '') != 'Final':
                continue
            home_id = game['teams']['home'].get('team', {}).get('id')
            side    = 'home' if home_id == team_id else 'away'
            lineup  = game.get('lineups', {}).get(f'{side}Players', [])
            if lineup:
                return lineup

    return []


def _upsert_todays_schedule(schedule: dict, game_date: str):
    """Upsert games/teams/venues from today's schedule so UI joins work."""
    from datetime import datetime as _dt
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from db.models import Game, Team, Venue

    season = int(game_date.split('-')[0])
    teams_seen: dict[int, dict] = {}
    venues_seen: dict[int, dict] = {}
    games_to_upsert: list[dict] = []

    for date_entry in schedule.get('dates', []):
        for game in date_entry.get('games', []):
            home = (game.get('teams', {}).get('home') or {}).get('team') or {}
            away = (game.get('teams', {}).get('away') or {}).get('team') or {}
            venue = game.get('venue') or {}

            if home.get('id'):
                teams_seen[home['id']] = {
                    'team_id': home['id'],
                    'name': home.get('name', ''),
                    'abbrev': home.get('abbreviation'),
                }
            if away.get('id'):
                teams_seen[away['id']] = {
                    'team_id': away['id'],
                    'name': away.get('name', ''),
                    'abbrev': away.get('abbreviation'),
                }
            if venue.get('id'):
                venues_seen[venue['id']] = {
                    'venue_id': venue['id'],
                    'name': venue.get('name', ''),
                }

            gtime = None
            try:
                gtime = _dt.fromisoformat(game.get('gameDate', '').replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass

            status_map = {'Final': 'final', 'Live': 'in_progress',
                          'Preview': 'scheduled', 'Postponed': 'postponed'}
            abstract = game.get('status', {}).get('abstractGameState', '')
            status = status_map.get(abstract, abstract.lower() or 'scheduled')

            games_to_upsert.append({
                'game_pk':      game['gamePk'],
                'game_date':    game_date,
                'season':       season,
                'home_team_id': home.get('id'),
                'away_team_id': away.get('id'),
                'venue_id':     venue.get('id'),
                'status':       status,
                'double_header': game.get('doubleHeader', 'N'),
                'game_time_utc': gtime,
                'weather':      _parse_weather(game.get('weather', {})),
            })

    session = get_session()
    try:
        for t in teams_seen.values():
            stmt = pg_insert(Team).values(**t)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Team.team_id],
                set_={'name': stmt.excluded.name, 'abbrev': stmt.excluded.abbrev},
            )
            session.execute(stmt)
        for v in venues_seen.values():
            stmt = pg_insert(Venue).values(**v)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Venue.venue_id],
                set_={'name': stmt.excluded.name},
            )
            session.execute(stmt)
        for g in games_to_upsert:
            stmt = pg_insert(Game).values(**g)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Game.game_pk],
                set_={k: getattr(stmt.excluded, k) for k in g if k != 'game_pk'},
            )
            session.execute(stmt)
        session.commit()
    finally:
        session.close()


def _parse_weather(w: dict) -> dict:
    temp_str = w.get('temp', '')
    wind_str = w.get('wind', '')
    temp = None
    try:
        temp = int(temp_str)
    except (ValueError, TypeError):
        pass
    wind_speed, wind_dir = None, ''
    if wind_str:
        parts = wind_str.split(',')
        try:
            wind_speed = int(parts[0].strip().split()[0])
        except (ValueError, IndexError):
            pass
        if len(parts) > 1:
            wind_dir = parts[1].strip()
    return {'condition': w.get('condition', ''), 'tempF': temp,
            'windSpeedMph': wind_speed, 'windDir': wind_dir}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='YYYY-MM-DD (default: today)')
    args = parser.parse_args()
    run(game_date=args.date)
