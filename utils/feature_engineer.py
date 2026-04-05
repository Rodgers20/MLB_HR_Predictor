"""
Feature engineering — builds the matchup feature matrix.
Uses FanGraphs batting/pitching data as primary source (it already includes
Statcast-derived metrics: Barrel%, HardHit%, EV, LA, xwOBA, xSLG, xBA).
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns sourced from FanGraphs batting
# ---------------------------------------------------------------------------

# Actual FanGraphs batting column names (verified from live data)
FG_BATTER_FEATURE_COLS = [
    "Barrel%",   # barrel rate
    "EV",        # average exit velocity
    "LA",        # average launch angle
    "HardHit%",  # hard hit %
    "xwOBA",     # expected wOBA
    "xSLG",      # expected slugging
    "xBA",       # expected batting average
    "ISO",       # isolated power
    "Pull%",     # pull %
    "Cent%",     # center field spray %
    "Oppo%",     # opposite field spray %
    "FB%",       # fly ball %
    "HR/FB",     # HR per fly ball
    "wRC+",      # weighted runs created plus
    "Hard%",     # hard contact % (alternative to HardHit%)
]

# Actual FanGraphs pitching column names
FG_PITCHER_FEATURE_COLS = [
    "HR/9",      # HR per 9 innings
    "HR/FB",     # HR per fly ball
    "FIP",       # fielding independent pitching
    "xFIP",      # expected FIP
    "GB%",       # ground ball %
    "Barrel%",   # barrel % allowed
    "HardHit%",  # hard hit % allowed
    "EV",        # exit velocity allowed
]

# Final model feature list (must match exactly what build_matchup_features produces)
MODEL_FEATURES = [
    # Batter 3yr weighted Statcast quality metrics
    "barrel_pct_w3yr",
    "exit_velo_w3yr",
    "launch_angle_w3yr",
    "hard_hit_pct_w3yr",
    "xwoba_w3yr",
    "xslg_w3yr",
    # Batter current season (batted-ball profile — NOT hr_rate to avoid label leakage)
    "iso",
    "pull_pct",
    "cent_pct",
    "oppo_pct",
    "fb_pct",
    "hr_fb_rate",
    "xhr_delta",       # actual HR rate - expected (FB% * league_HR_FB) → over/under-performing power
    "wrc_plus",
    # Platoon matchup
    "platoon_adj",     # batter/pitcher handedness multiplier
    # Pitcher
    "pitcher_hr9",
    "pitcher_hr_fb",
    "pitcher_fip",
    "pitcher_gb_pct",
    "pitcher_barrel_pct",
    "pitcher_hard_hit_pct",
    # Context
    "park_hr_factor",
    "batter_park_factor",  # park factor adjusted for batter's pull/FB profile
    "temp_f",
    "temp_adjustment",
    "wind_speed_mph",
    "wind_direction_enc",
    "is_day_game",
]

# ---------------------------------------------------------------------------
# 3-year weighted average
# ---------------------------------------------------------------------------

YEAR_WEIGHTS = {0: 0.60, 1: 0.30, 2: 0.10}  # offset from most recent year


def build_3yr_weighted_fg(
    fg_batting: pd.DataFrame,
    current_year: int,
    id_col: str = "IDfg",
) -> pd.DataFrame:
    """
    Build 3-year weighted Statcast-derived features from FanGraphs batting data.
    Weight: 60% current year, 30% year-1, 10% year-2.
    Returns one row per player (id_col) with suffixed _w3yr columns.
    """
    # Normalise Season column name
    year_col = "Season" if "Season" in fg_batting.columns else "season"

    raw_cols = ["Barrel%", "EV", "LA", "HardHit%", "xwOBA", "xSLG", "xBA",
                "HR/FB", "ISO", "FB%", "Cent%", "Oppo%", "Pull%"]
    available = [c for c in raw_cols if c in fg_batting.columns]

    result_rows = []
    for player_id, group in fg_batting.groupby(id_col):
        row = {id_col: player_id}
        for col in available:
            vals, wts = [], []
            for offset, weight in YEAR_WEIGHTS.items():
                year = current_year - offset
                year_data = group[group[year_col] == year]
                if not year_data.empty:
                    val = year_data[col].iloc[0]
                    vals.append(float(val) if pd.notna(val) else np.nan)
                    wts.append(weight)
            if vals:
                total_w = sum(w for v, w in zip(vals, wts) if not np.isnan(v))
                total_v = sum(v * w for v, w in zip(vals, wts) if not np.isnan(v))
                row[f"{col}_w3yr"] = total_v / total_w if total_w > 0 else np.nan
            else:
                row[f"{col}_w3yr"] = np.nan
        result_rows.append(row)

    return pd.DataFrame(result_rows)


def build_3yr_weighted_pitcher(
    fg_pitching: pd.DataFrame,
    current_year: int,
    id_col: str = "IDfg",
) -> pd.DataFrame:
    """3-year weighted pitcher features from FanGraphs pitching."""
    year_col = "Season" if "Season" in fg_pitching.columns else "season"
    raw_cols = ["HR/9", "HR/FB", "FIP", "xFIP", "GB%", "Barrel%", "HardHit%", "EV"]
    available = [c for c in raw_cols if c in fg_pitching.columns]

    result_rows = []
    for player_id, group in fg_pitching.groupby(id_col):
        row = {id_col: player_id}
        for col in available:
            vals, wts = [], []
            for offset, weight in YEAR_WEIGHTS.items():
                year = current_year - offset
                year_data = group[group[year_col] == year]
                if not year_data.empty:
                    val = year_data[col].iloc[0]
                    vals.append(float(val) if pd.notna(val) else np.nan)
                    wts.append(weight)
            if vals:
                total_w = sum(w for v, w in zip(vals, wts) if not np.isnan(v))
                total_v = sum(v * w for v, w in zip(vals, wts) if not np.isnan(v))
                row[f"{col}_w3yr"] = total_v / total_w if total_w > 0 else np.nan
            else:
                row[f"{col}_w3yr"] = np.nan
        result_rows.append(row)

    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# Platoon adjustment
# ---------------------------------------------------------------------------

def platoon_adj(stand: str, p_throws: str) -> float:
    """
    Return a multiplier reflecting the batter vs pitcher handedness matchup.
    Based on historical MLB platoon split data:
        L batter vs R pitcher → neutral (most common, ~50% of PA)
        R batter vs L pitcher → +15% HR advantage (right-on-left advantage)
        L batter vs L pitcher → -22% (left-on-left disadvantage)
        R batter vs R pitcher → -12% (right-on-right moderate disadvantage)
    """
    s = str(stand or "").upper().strip()
    p = str(p_throws or "").upper().strip()
    if s == "L" and p == "R":
        return 1.00   # neutral (largest matchup pool)
    elif s == "R" and p == "L":
        return 1.15   # righty's platoon advantage vs lefty
    elif s == "L" and p == "L":
        return 0.78   # lefty-on-lefty disadvantage
    elif s == "R" and p == "R":
        return 0.88   # same-hand moderate disadvantage
    return 1.00       # unknown / switch hitter


# ---------------------------------------------------------------------------
# Wind / temperature helpers
# ---------------------------------------------------------------------------

WIND_DIRECTION_MAP = {"out": 1.0, "cross": 0.0, "calm": 0.0, "in": -1.0}


def encode_wind_direction(direction: str) -> float:
    return WIND_DIRECTION_MAP.get(str(direction).lower().strip(), 0.0)


def temp_adjustment(temp_f: float, baseline_f: float = 72.0) -> float:
    """Every 10°F above baseline ≈ +1% ball distance."""
    return 1.0 + ((temp_f - baseline_f) / 10.0) * 0.01


# ---------------------------------------------------------------------------
# Build matchup feature dict for a single batter vs pitcher
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# League-average constants used for small-sample Bayesian smoothing
# ---------------------------------------------------------------------------
_LEAGUE_HR_FB   = 0.123  # MLB career average HR/FB rate
_LEAGUE_ISO     = 0.165  # MLB average ISO
_LEAGUE_FB_PCT  = 0.36   # MLB average FB%
_LEAGUE_CENT_PCT = 0.37  # MLB average center-field spray %
_LEAGUE_OPPO_PCT = 0.28  # MLB average opposite-field spray %
_LEAGUE_PULL_PCT = 0.35  # MLB average pull %
_PA_TRUST_FLOOR  = 60    # PA below this → shrink toward league mean


def _regress_to_mean(current: float, fallback: float, pa: float) -> float:
    """Shrink a noisy stat toward `fallback` when PA is small.

    trust = min(pa / _PA_TRUST_FLOOR, 1.0)
    result = trust * current + (1 - trust) * fallback

    `fallback` should be the player's own 3yr historical average when available,
    otherwise the league mean.  This preserves known elite players
    (e.g. Ohtani's historical 32% HR/FB) even early in the season.

    At PA=0  → pure fallback.
    At PA=60 → full current-season value.
    """
    if np.isnan(current):
        return fallback
    trust = min(pa / _PA_TRUST_FLOOR, 1.0)
    return trust * current + (1.0 - trust) * fallback


def build_matchup_features(
    batter_row: pd.Series,
    batter_weighted: pd.Series,
    pitcher_row: pd.Series,
    park_factor: float,
    temp_f: float,
    wind_speed_mph: float,
    wind_direction: str,
    is_day_game: bool,
    batter_stand: str = "",
    pitcher_throws: str = "",
) -> dict:
    """
    Combine batter (current + 3yr weighted) + pitcher + context into one feature dict.
    Column names align exactly with MODEL_FEATURES.

    Small-sample guard: hr_fb_rate, iso, and fb_pct are regressed toward league
    averages when the batter has < 60 PA in the current season.  This prevents
    fluky early-season rates (e.g. 1 HR in 3 fly balls = 33%) from dominating
    predictions before a meaningful sample has accumulated.
    """
    hr_count = float(batter_row.get("HR", 0) or 0)
    pa_count = float(batter_row.get("PA", 0) or 0)
    hr_rate  = hr_count / pa_count if pa_count > 0 else 0.0

    raw_hr_fb = float(batter_row.get("HR/FB", np.nan) or np.nan)
    raw_iso   = float(batter_row.get("ISO",   np.nan) or np.nan)
    raw_fb    = float(batter_row.get("FB%",   np.nan) or np.nan)
    raw_cent  = float(batter_row.get("Cent%", np.nan) or np.nan)
    raw_oppo  = float(batter_row.get("Oppo%", np.nan) or np.nan)
    raw_pull  = float(batter_row.get("Pull%", np.nan) or np.nan)

    # Use player's own 3yr history as the regression baseline; fall back to league mean.
    hist_hr_fb = batter_weighted.get("HR/FB_w3yr", np.nan)
    hist_iso   = batter_weighted.get("ISO_w3yr",   np.nan)
    hist_fb    = batter_weighted.get("FB%_w3yr",   np.nan)
    hist_cent  = batter_weighted.get("Cent%_w3yr", np.nan)
    hist_oppo  = batter_weighted.get("Oppo%_w3yr", np.nan)
    hist_pull  = batter_weighted.get("Pull%_w3yr", np.nan)

    baseline_hr_fb = hist_hr_fb if not np.isnan(hist_hr_fb) else _LEAGUE_HR_FB
    baseline_iso   = hist_iso   if not np.isnan(hist_iso)   else _LEAGUE_ISO
    baseline_fb    = hist_fb    if not np.isnan(hist_fb)    else _LEAGUE_FB_PCT
    baseline_cent  = hist_cent  if not np.isnan(hist_cent)  else _LEAGUE_CENT_PCT
    baseline_oppo  = hist_oppo  if not np.isnan(hist_oppo)  else _LEAGUE_OPPO_PCT
    baseline_pull  = hist_pull  if not np.isnan(hist_pull)  else _LEAGUE_PULL_PCT

    hr_fb_rate = _regress_to_mean(raw_hr_fb, baseline_hr_fb, pa_count)
    iso        = _regress_to_mean(raw_iso,   baseline_iso,   pa_count)
    fb_pct     = _regress_to_mean(raw_fb,    baseline_fb,    pa_count)
    cent_pct   = _regress_to_mean(raw_cent,  baseline_cent,  pa_count)
    oppo_pct   = _regress_to_mean(raw_oppo,  baseline_oppo,  pa_count)
    pull_pct   = _regress_to_mean(raw_pull,  baseline_pull,  pa_count)

    # xHR delta: how much the batter's (smoothed) HR/FB rate exceeds the league average.
    # Consistent with training definition (historical HR/FB − league_mean).
    # Positive = consistent power over-performer vs. fly-ball profile.
    xhr_delta = hr_fb_rate - _LEAGUE_HR_FB

    # Batter-specific park factor: pulls hitters benefit more from hitter-friendly parks
    # Formula: base_pf adjusted by how much pull/FB tilts toward HR-friendly conditions
    batter_park_factor = park_factor * (
        1.0
        + (pull_pct - _LEAGUE_PULL_PCT) * 0.30
        + (fb_pct   - _LEAGUE_FB_PCT)   * 0.15
    )

    return {
        # Batter 3yr weighted
        "barrel_pct_w3yr": batter_weighted.get("Barrel%_w3yr", np.nan),
        "exit_velo_w3yr":  batter_weighted.get("EV_w3yr", np.nan),
        "launch_angle_w3yr": batter_weighted.get("LA_w3yr", np.nan),
        "hard_hit_pct_w3yr": batter_weighted.get("HardHit%_w3yr", np.nan),
        "xwoba_w3yr": batter_weighted.get("xwOBA_w3yr", np.nan),
        "xslg_w3yr":  batter_weighted.get("xSLG_w3yr", np.nan),
        # Batter current season (small-sample smoothed)
        "iso":        iso,
        "pull_pct":   pull_pct,
        "cent_pct":   cent_pct,
        "oppo_pct":   oppo_pct,
        "fb_pct":     fb_pct,
        "hr_fb_rate": hr_fb_rate,
        "xhr_delta":  xhr_delta,
        "hr_rate":    hr_rate,
        "wrc_plus":   float(batter_row.get("wRC+", np.nan) or np.nan),
        # Platoon matchup
        "platoon_adj": platoon_adj(batter_stand, pitcher_throws),
        # Pitcher
        "pitcher_hr9":       float(pitcher_row.get("HR/9", np.nan) or np.nan),
        "pitcher_hr_fb":     float(pitcher_row.get("HR/FB", np.nan) or np.nan),
        "pitcher_fip":       float(pitcher_row.get("FIP", np.nan) or np.nan),
        "pitcher_gb_pct":    float(pitcher_row.get("GB%", np.nan) or np.nan),
        "pitcher_barrel_pct":   float(pitcher_row.get("Barrel%", np.nan) or np.nan),
        "pitcher_hard_hit_pct": float(pitcher_row.get("HardHit%", np.nan) or np.nan),
        # Pitcher extended (present for game-level model)
        "pitcher_xfip":     float(pitcher_row.get("xFIP", np.nan) or np.nan),
        "pitcher_k_pct":    float(pitcher_row.get("K%", np.nan) or np.nan),
        # Context
        "park_hr_factor":     park_factor,
        "batter_park_factor": batter_park_factor,
        "temp_f":             temp_f,
        "temp_adjustment":    temp_adjustment(temp_f),
        "wind_speed_mph":     wind_speed_mph,
        "wind_direction_enc": encode_wind_direction(wind_direction),
        "is_day_game":        int(is_day_game),
    }
