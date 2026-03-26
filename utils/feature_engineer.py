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
    # Batter weighted
    "barrel_pct_w3yr",
    "exit_velo_w3yr",
    "launch_angle_w3yr",
    "hard_hit_pct_w3yr",
    "xwoba_w3yr",
    "xslg_w3yr",
    # Batter current season
    "iso",
    "pull_pct",
    "fb_pct",
    "hr_fb_rate",
    "hr_rate",
    "wrc_plus",
    # Pitcher
    "pitcher_hr9",
    "pitcher_hr_fb",
    "pitcher_fip",
    "pitcher_gb_pct",
    "pitcher_barrel_pct",
    "pitcher_hard_hit_pct",
    # Context
    "park_hr_factor",
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

    raw_cols = ["Barrel%", "EV", "LA", "HardHit%", "xwOBA", "xSLG", "xBA"]
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

def build_matchup_features(
    batter_row: pd.Series,
    batter_weighted: pd.Series,
    pitcher_row: pd.Series,
    park_factor: float,
    temp_f: float,
    wind_speed_mph: float,
    wind_direction: str,
    is_day_game: bool,
) -> dict:
    """
    Combine batter (current + 3yr weighted) + pitcher + context into one feature dict.
    Column names align exactly with MODEL_FEATURES.
    """
    hr_count = float(batter_row.get("HR", 0) or 0)
    pa_count = float(batter_row.get("PA", 1) or 1)
    hr_rate = hr_count / pa_count if pa_count > 0 else 0.0

    return {
        # Batter 3yr weighted
        "barrel_pct_w3yr": batter_weighted.get("Barrel%_w3yr", np.nan),
        "exit_velo_w3yr":  batter_weighted.get("EV_w3yr", np.nan),
        "launch_angle_w3yr": batter_weighted.get("LA_w3yr", np.nan),
        "hard_hit_pct_w3yr": batter_weighted.get("HardHit%_w3yr", np.nan),
        "xwoba_w3yr": batter_weighted.get("xwOBA_w3yr", np.nan),
        "xslg_w3yr":  batter_weighted.get("xSLG_w3yr", np.nan),
        # Batter current season
        "iso":       float(batter_row.get("ISO", np.nan) or np.nan),
        "pull_pct":  float(batter_row.get("Pull%", np.nan) or np.nan),
        "fb_pct":    float(batter_row.get("FB%", np.nan) or np.nan),
        "hr_fb_rate": float(batter_row.get("HR/FB", np.nan) or np.nan),
        "hr_rate":   hr_rate,
        "wrc_plus":  float(batter_row.get("wRC+", np.nan) or np.nan),
        # Pitcher
        "pitcher_hr9":       float(pitcher_row.get("HR/9", np.nan) or np.nan),
        "pitcher_hr_fb":     float(pitcher_row.get("HR/FB", np.nan) or np.nan),
        "pitcher_fip":       float(pitcher_row.get("FIP", np.nan) or np.nan),
        "pitcher_gb_pct":    float(pitcher_row.get("GB%", np.nan) or np.nan),
        "pitcher_barrel_pct":   float(pitcher_row.get("Barrel%", np.nan) or np.nan),
        "pitcher_hard_hit_pct": float(pitcher_row.get("HardHit%", np.nan) or np.nan),
        # Context
        "park_hr_factor":   park_factor,
        "temp_f":           temp_f,
        "temp_adjustment":  temp_adjustment(temp_f),
        "wind_speed_mph":   wind_speed_mph,
        "wind_direction_enc": encode_wind_direction(wind_direction),
        "is_day_game":      int(is_day_game),
    }
