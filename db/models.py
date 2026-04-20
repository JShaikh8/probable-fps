"""
SQLAlchemy schema for the sports-oracle Postgres database.

All pipeline writes go through tables defined here. Shapes mirror the prior
MongoDB collections but with real PKs, indexes, and JSONB for nested blobs
(factor maps, similar-hitter lists, arsenal dicts).
"""
from __future__ import annotations

from datetime import datetime
from sqlalchemy import (
    BigInteger, Boolean, Column, Date, DateTime, Float, ForeignKey, Index,
    Integer, String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# ── Reference tables ────────────────────────────────────────────────

class Venue(Base):
    __tablename__ = 'venues'
    venue_id = Column(Integer, primary_key=True)
    name = Column(String(200))
    city = Column(String(100), nullable=True)
    state = Column(String(50), nullable=True)
    roof_type = Column(String(50), nullable=True)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    # Outfield fence distances in feet (LF line, CF, RF line)
    lf_ft = Column(Float, nullable=True)
    cf_ft = Column(Float, nullable=True)
    rf_ft = Column(Float, nullable=True)


class Team(Base):
    __tablename__ = 'teams'
    team_id = Column(Integer, primary_key=True)
    name = Column(String(200))
    abbrev = Column(String(10), nullable=True)
    venue_id = Column(Integer, ForeignKey('venues.venue_id'), nullable=True)
    league = Column(String(50), nullable=True)


class Player(Base):
    __tablename__ = 'players'
    player_id = Column(Integer, primary_key=True)
    full_name = Column(String(200))
    bat_side = Column(String(1), nullable=True)   # R, L, S
    pitch_hand = Column(String(1), nullable=True)
    primary_position = Column(String(10), nullable=True)
    team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=True)
    active = Column(Boolean, default=True)


# ── Game tables ─────────────────────────────────────────────────────

class Game(Base):
    __tablename__ = 'games'
    game_pk = Column(BigInteger, primary_key=True)
    game_date = Column(Date, index=True)
    season = Column(Integer, index=True)
    home_team_id = Column(Integer, ForeignKey('teams.team_id'), index=True)
    away_team_id = Column(Integer, ForeignKey('teams.team_id'), index=True)
    venue_id = Column(Integer, ForeignKey('venues.venue_id'), index=True, nullable=True)
    status = Column(String(20), index=True)         # scheduled, in_progress, final, postponed
    double_header = Column(String(1), nullable=True)
    game_time_utc = Column(DateTime, nullable=True)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    weather = Column(JSONB, nullable=True)  # {condition, tempF, windSpeedMph, windDir}


class GameLog(Base):
    """Ingestion tracking — one row per game_pk with status and counters."""
    __tablename__ = 'games_log'
    game_pk = Column(BigInteger, primary_key=True)
    status = Column(String(20), index=True)         # done, failed, pending
    pitch_count = Column(Integer, default=0)
    ab_count = Column(Integer, default=0)
    ingested_at = Column(DateTime, default=func.now())
    error = Column(Text, nullable=True)


class AtBat(Base):
    __tablename__ = 'at_bats'
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    game_pk = Column(BigInteger, index=True)
    at_bat_index = Column(Integer)
    inning = Column(Integer)
    half_inning = Column(String(10))                # top, bottom
    hitter_id = Column(Integer, index=True)
    pitcher_id = Column(Integer, index=True)
    hitter_side = Column(String(1), nullable=True)
    pitcher_hand = Column(String(1), nullable=True)
    event = Column(String(80), nullable=True)
    event_type = Column(String(80), nullable=True)
    description = Column(Text, nullable=True)
    rbi = Column(Integer, default=0)
    exit_velocity = Column(Float, nullable=True)
    launch_angle = Column(Float, nullable=True)
    launch_speed_angle = Column(Integer, nullable=True)
    total_distance = Column(Float, nullable=True)
    hit_coord_x = Column(Float, nullable=True)
    hit_coord_y = Column(Float, nullable=True)
    trajectory = Column(String(50), nullable=True)
    hardness = Column(String(50), nullable=True)
    location = Column(String(20), nullable=True)
    game_date = Column(Date, index=True)
    season = Column(Integer, index=True)
    __table_args__ = (
        UniqueConstraint('game_pk', 'at_bat_index', name='uq_atbats_game_idx'),
        Index('ix_atbats_hitter_date', 'hitter_id', 'game_date'),
        Index('ix_atbats_pitcher_date', 'pitcher_id', 'game_date'),
    )


class Pitch(Base):
    __tablename__ = 'pitches'
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    game_pk = Column(BigInteger, index=True)
    at_bat_index = Column(Integer)
    pitch_index = Column(Integer)
    hitter_id = Column(Integer, index=True)
    pitcher_id = Column(Integer, index=True)
    pitch_type = Column(String(10), nullable=True)      # raw MLBAM code: FF, SL, CH...
    pitch_family = Column(String(20), index=True, nullable=True)  # fastball / breaking / offspeed
    start_speed = Column(Float, nullable=True)
    end_speed = Column(Float, nullable=True)
    spin_rate = Column(Float, nullable=True)
    spin_direction = Column(Float, nullable=True)
    px = Column(Float, nullable=True)
    pz = Column(Float, nullable=True)
    # Phase-4: movement + release geometry (Statcast pitch tracking)
    pfx_x = Column(Float, nullable=True)       # horizontal break (inches, pitcher-perspective)
    pfx_z = Column(Float, nullable=True)       # vertical break (inches, gravity-adjusted — "IVB")
    x0 = Column(Float, nullable=True)          # release side (feet, from center)
    z0 = Column(Float, nullable=True)          # release height (feet, from ground)
    extension = Column(Float, nullable=True)   # release distance in front of rubber (ft)
    plate_time = Column(Float, nullable=True)  # release → plate (seconds)
    pitch_result = Column(String(50), nullable=True)
    zone = Column(Integer, nullable=True)
    strikes = Column(Integer, nullable=True)
    balls = Column(Integer, nullable=True)
    game_date = Column(Date, index=True)
    season = Column(Integer, index=True)
    __table_args__ = (
        UniqueConstraint('game_pk', 'at_bat_index', 'pitch_index', name='uq_pitches_game_ab_idx'),
        Index('ix_pitches_hitter_family', 'hitter_id', 'pitch_family'),
        Index('ix_pitches_pitcher_family', 'pitcher_id', 'pitch_family'),
    )


# ── Feature tables ──────────────────────────────────────────────────

class HitterPitchSplit(Base):
    __tablename__ = 'hitter_pitch_splits'
    hitter_id = Column(Integer, primary_key=True)
    pitch_family = Column(String(20), primary_key=True)
    season = Column(Integer, primary_key=True)
    pa = Column(Integer, default=0)
    ab = Column(Integer, default=0)
    hits = Column(Integer, default=0)
    hr = Column(Integer, default=0)
    bb = Column(Integer, default=0)
    k = Column(Integer, default=0)
    avg = Column(Float, default=0.0)
    slg = Column(Float, default=0.0)
    obp = Column(Float, default=0.0)
    ops = Column(Float, default=0.0)
    swing_pct = Column(Float, default=0.0)
    whiff_pct = Column(Float, default=0.0)
    avg_exit_velo = Column(Float, nullable=True)
    avg_launch_angle = Column(Float, nullable=True)
    hard_hit_pct = Column(Float, nullable=True)
    high_velo_whiff_pct = Column(Float, nullable=True)
    high_spin_whiff_pct = Column(Float, nullable=True)
    # Phase-1 batted-ball quality additions
    barrel_pct = Column(Float, nullable=True)     # EV ≥ 95 AND LA in [25, 35]
    fb_pct = Column(Float, nullable=True)         # fly_ball / popup trajectory share
    ld_pct = Column(Float, nullable=True)
    gb_pct = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class PitcherProfile(Base):
    __tablename__ = 'pitcher_profiles'
    pitcher_id = Column(Integer, primary_key=True)
    season = Column(Integer, primary_key=True)
    arsenal = Column(JSONB)                     # {fastball: {usagePct, avgSpeed, avgSpin, whiffPct}, ...}
    total_pitches = Column(Integer, default=0)
    k_pct = Column(Float, default=0.0)
    primary_pitch = Column(String(20), nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class PitcherSeasonStats(Base):
    __tablename__ = 'pitcher_season_stats'
    pitcher_id = Column(Integer, primary_key=True)
    season = Column(Integer, primary_key=True)
    avg_ip = Column(Float, default=0.0)
    avg_k = Column(Float, default=0.0)
    avg_bb = Column(Float, default=0.0)
    avg_h = Column(Float, default=0.0)
    avg_hr = Column(Float, default=0.0)
    fip = Column(Float, default=0.0)
    games_started = Column(Integer, default=0)
    # Phase-1 pitcher handedness splits + batted-ball quality allowed
    hr9_vs_l = Column(Float, nullable=True)
    hr9_vs_r = Column(Float, nullable=True)
    k_pct_vs_l = Column(Float, nullable=True)
    k_pct_vs_r = Column(Float, nullable=True)
    fb_pct_allowed = Column(Float, nullable=True)
    barrel_pct_allowed = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class ParkFactor(Base):
    __tablename__ = 'park_factors'
    venue_id = Column(Integer, primary_key=True)
    hr_factor = Column(Float, default=1.0)
    hit_factor = Column(Float, default=1.0)
    hard_hit_factor = Column(Float, default=1.0)
    k_factor = Column(Float, default=1.0)
    bb_factor = Column(Float, default=1.0)
    sample_size = Column(Integer, default=0)
    hit_locations = Column(JSONB, nullable=True)  # 3x3 grid
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class HitterRecentForm(Base):
    __tablename__ = 'hitter_recent_form'
    hitter_id = Column(Integer, primary_key=True)
    form_signal = Column(String(10))            # hot / normal / cold
    form_ratio = Column(Float, default=1.0)
    last_7 = Column(JSONB, nullable=True)       # aggregate stats
    last_15 = Column(JSONB, nullable=True)
    last_30 = Column(JSONB, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class HitterSprayProfile(Base):
    __tablename__ = 'hitter_spray_profiles'
    hitter_id = Column(Integer, primary_key=True)
    pull_pct = Column(Float, default=0.0)
    center_pct = Column(Float, default=0.0)
    oppo_pct = Column(Float, default=0.0)
    deep_pct = Column(Float, default=0.0)
    shallow_pct = Column(Float, default=0.0)
    infield_pct = Column(Float, default=0.0)
    avg_exit_velo = Column(Float, nullable=True)
    avg_launch_angle = Column(Float, nullable=True)
    hr_pull_pct = Column(Float, default=0.0)
    # Fly-ball field distribution (by physical bearing angle from home plate).
    # Used for distance-interacted park HR multiplier: short LF wall ×
    # hitter-who-pulls-FBs-to-LF = real HR boost.
    fb_lf_pct = Column(Float, nullable=True)
    fb_cf_pct = Column(Float, nullable=True)
    fb_rf_pct = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class HitterVector(Base):
    __tablename__ = 'hitter_vectors'
    hitter_id = Column(Integer, primary_key=True)
    vector = Column(JSONB)                       # named feature map
    scaled_vector = Column(ARRAY(Float))         # for cosine similarity
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class HitterSimilar(Base):
    __tablename__ = 'hitter_similar'
    hitter_id = Column(Integer, primary_key=True)
    similar_list = Column(JSONB)                      # [{hitter_id, hitter_name, similarity}]
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class PitcherVector(Base):
    __tablename__ = 'pitcher_vectors'
    pitcher_id = Column(Integer, primary_key=True)
    vector = Column(JSONB)
    scaled_vector = Column(ARRAY(Float))
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class PitcherSimilar(Base):
    __tablename__ = 'pitcher_similar'
    pitcher_id = Column(Integer, primary_key=True)
    similar_list = Column(JSONB)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


# ── Projection tables ───────────────────────────────────────────────

class Projection(Base):
    """Hitter projection, one row per hitter per game_pk."""
    __tablename__ = 'projections'
    hitter_id = Column(Integer, primary_key=True)
    game_pk = Column(BigInteger, primary_key=True)
    game_date = Column(Date, index=True)
    pitcher_id = Column(Integer, index=True, nullable=True)
    proj = Column(JSONB)                         # {h, hr, bb, k, r, rbi, avg, slg}
    dk_pts = Column(Float, default=0.0)
    fd_pts = Column(Float, default=0.0)
    baseline_dk_pts = Column(Float, default=0.0)
    dk_delta = Column(Float, default=0.0)
    factors = Column(JSONB)                      # {park, weather, platoon, stuffQuality, recentForm, battingOrder, matchup}
    factor_score = Column(Float, default=0.0)
    contact_quality = Column(JSONB, nullable=True)
    expected_pa = Column(Float, default=4.0)
    lineup_slot = Column(Integer, nullable=True)
    weather = Column(JSONB, nullable=True)
    hitter_hand = Column(String(1), nullable=True)
    pitcher_hand = Column(String(1), nullable=True)
    lineup_source = Column(String(30), nullable=True)
    side = Column(String(4), nullable=True)          # 'home' or 'away'
    tuned_dk_pts = Column(Float, nullable=True)      # factor + learned bias correction
    ml_dk_pts = Column(Float, nullable=True)         # matchup-classifier prediction (DK)
    ml_fd_pts = Column(Float, nullable=True)         # matchup-classifier prediction (FD)
    ml_delta = Column(Float, nullable=True)          # ml_dk_pts - dk_pts
    blend_dk_pts = Column(Float, nullable=True)      # consensus: (tuned + ml) / 2
    blend_fd_pts = Column(Float, nullable=True)
    ml_outcome_probs = Column(JSONB, nullable=True)  # {single: 0.14, double: 0.04, ...} from classifier
    created_at = Column(DateTime, default=func.now())
    __table_args__ = (
        Index('ix_projections_date', 'game_date'),
    )


class PitcherProjection(Base):
    __tablename__ = 'pitcher_projections'
    pitcher_id = Column(Integer, primary_key=True)
    game_pk = Column(BigInteger, primary_key=True)
    game_date = Column(Date, index=True)
    proj = Column(JSONB)                         # {ip, k, bb, h, hr, er}
    dk_pts = Column(Float, default=0.0)
    fd_pts = Column(Float, default=0.0)
    fip = Column(Float, nullable=True)
    games_started = Column(Integer, default=0)
    stuff_signal = Column(String(10), nullable=True)   # pos / neutral / neg
    model_version = Column(String(20), nullable=True)
    lineup_source = Column(String(30), nullable=True)
    side = Column(String(4), nullable=True)
    ml_dk_pts = Column(Float, nullable=True)
    ml_fd_pts = Column(Float, nullable=True)
    ml_delta = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now())


class NrfiProjection(Base):
    __tablename__ = 'nrfi_projections'
    game_pk = Column(BigInteger, primary_key=True)
    game_date = Column(Date, index=True)
    nrfi_prob = Column(Float)
    nrfi_pct = Column(Float)
    yrfi_pct = Column(Float)
    home_xr = Column(Float)
    away_xr = Column(Float)
    home_p_scoreless = Column(Float)
    away_p_scoreless = Column(Float)
    home_p_score = Column(Float)
    away_p_score = Column(Float)
    top_threats = Column(JSONB)
    home_top_batters = Column(JSONB, nullable=True)
    away_top_batters = Column(JSONB, nullable=True)
    home_pitcher = Column(JSONB, nullable=True)
    away_pitcher = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=func.now())


# ── Reconciliation (backtest) tables — new ──────────────────────────

class ProjectionActual(Base):
    """Nightly reconciliation: yesterday's hitter projection vs real box score."""
    __tablename__ = 'projection_actuals'
    hitter_id = Column(Integer, primary_key=True)
    game_pk = Column(BigInteger, primary_key=True)
    game_date = Column(Date, index=True)
    proj_dk_pts = Column(Float)
    actual_dk_pts = Column(Float)
    proj = Column(JSONB)
    actual = Column(JSONB)                       # {h, hr, bb, k, r, rbi, pa}
    dk_error = Column(Float)                     # actual - proj
    abs_dk_error = Column(Float)
    reconciled_at = Column(DateTime, default=func.now())


class PitcherProjectionActual(Base):
    __tablename__ = 'pitcher_projection_actuals'
    pitcher_id = Column(Integer, primary_key=True)
    game_pk = Column(BigInteger, primary_key=True)
    game_date = Column(Date, index=True)
    proj_dk_pts = Column(Float)
    actual_dk_pts = Column(Float)
    proj = Column(JSONB)
    actual = Column(JSONB)
    dk_error = Column(Float)
    abs_dk_error = Column(Float)
    reconciled_at = Column(DateTime, default=func.now())


class FdSlatePrice(Base):
    """FanDuel DFS slate prices, uploaded daily via CSV."""
    __tablename__ = 'fd_slate_prices'
    slate_date = Column(Date, primary_key=True)
    fd_player_id = Column(String(50), primary_key=True)
    fd_name = Column(String(200))
    position = Column(String(20))            # 'P' | 'C/1B' | '2B' | '3B' | 'SS' | 'OF' | 'UTIL'
    salary = Column(Integer)
    fppg = Column(Float, nullable=True)
    team = Column(String(10))
    opponent = Column(String(10))
    game = Column(String(20))                 # e.g. 'DET@BOS'
    injury_indicator = Column(String(10), nullable=True)
    batting_order = Column(Integer, nullable=True)
    probable_pitcher = Column(Boolean, default=False)
    matched_player_id = Column(Integer, ForeignKey('players.player_id'), nullable=True)
    uploaded_at = Column(DateTime, default=func.now())


class NrfiActual(Base):
    __tablename__ = 'nrfi_actuals'
    game_pk = Column(BigInteger, primary_key=True)
    game_date = Column(Date, index=True)
    predicted_nrfi_prob = Column(Float)
    actual_nrfi = Column(Boolean)                # true if first inning had 0 runs
    home_fi_runs = Column(Integer)
    away_fi_runs = Column(Integer)
    reconciled_at = Column(DateTime, default=func.now())


class HitterGameStats(Base):
    """Per-hitter per-game runs + stolen bases pulled from MLB boxscores.

    Runs and SB don't appear in the /game/feed/live at-bat stream, so they need
    a separate ingest pass. Each run = 2 DK pts, each SB = 5 DK pts — skipping
    this left a systematic ~1.4-pt under-projection for active leadoff hitters.
    """
    __tablename__ = 'hitter_game_stats'
    hitter_id = Column(Integer, primary_key=True)
    game_pk = Column(BigInteger, primary_key=True)
    runs = Column(Integer, default=0)
    stolen_bases = Column(Integer, default=0)
    caught_stealing = Column(Integer, default=0)
    sac_flies = Column(Integer, default=0)
    ingested_at = Column(DateTime, default=datetime.utcnow)
