"""
Game-level training dataset builder for MLB HR Predictor.

Fetches Statcast play-by-play for each season and aggregates to one row per
(batter, game_pk, game_date).  Target column: hit_hr (1 if ≥1 HR that game).

Usage (CLI):
    python utils/game_log_builder.py                  # build all seasons
    python utils/game_log_builder.py --seasons 2024   # single season
    python utils/game_log_builder.py --force           # ignore cache
"""

import argparse
import logging
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/game_level_training.parquet")

# Regular-season date windows per year (approx; Statcast ignores non-game days)
SEASON_WINDOWS = {
    2020: ("2020-07-23", "2020-09-27"),  # COVID-shortened
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
}

# Extended feature list for the game-level retrained model.
# Import from here in model_trainer.py for the game-level training function.
GAME_LEVEL_FEATURES = [
    # Batter 3yr weighted Statcast
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
    "wrc_plus",
    # Contact quality (game-level Statcast)
    "sweet_spot_pct",    # % balls at 8–32° launch angle (season aggregated)
    "max_ev_95th",       # 95th-percentile exit velocity (season)
    # Platoon
    "platoon_adj",       # L/R matchup multiplier (0.78–1.15)
    # Pitcher — now with ACTUAL per-pitcher stats (not league avg)
    "pitcher_hr9",
    "pitcher_hr_fb",
    "pitcher_fip",
    "pitcher_xfip",      # xFIP (better HR predictor than FIP)
    "pitcher_gb_pct",
    "pitcher_k_pct",     # K% — high K means fewer balls in play
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
# Fetch
# ---------------------------------------------------------------------------

def _fetch_statcast_season(start_dt: str, end_dt: str) -> pd.DataFrame:
    """Download one season of Statcast pitch-by-pitch data via pybaseball."""
    try:
        from pybaseball import statcast
        logger.info("Downloading Statcast %s → %s (this takes several minutes) ...", start_dt, end_dt)
        sc = statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
        if sc is None or sc.empty:
            return pd.DataFrame()
        return sc[sc["game_type"] == "R"].copy()  # regular-season only
    except Exception as exc:
        logger.error("Statcast download failed for %s–%s: %s", start_dt, end_dt, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Aggregate to batter-game level
# ---------------------------------------------------------------------------

def _aggregate_to_game_level(sc: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Collapse pitch-by-pitch rows into one row per (batter, game_pk, game_date).
    Produces:
        hit_hr          — 1 if player hit ≥1 HR that game
        max_ev          — max launch_speed that game
        avg_ev          — mean launch_speed
        sweet_spot_pct  — fraction of batted balls at 8–32° LA
        ab_count        — total plate appearances (proxy)
        opposing_pitcher_id — most common pitcher faced
        stand           — batter hand (L/R)
    """
    if sc.empty:
        return pd.DataFrame()

    need = {"batter", "game_pk", "game_date"}
    if not need.issubset(sc.columns):
        logger.warning("Statcast data missing required columns %s", need - set(sc.columns))
        return pd.DataFrame()

    group_cols = [c for c in ["batter", "game_pk", "game_date", "stand"] if c in sc.columns]

    agg_kwargs: dict = {}
    if "events" in sc.columns:
        agg_kwargs["hit_hr"]    = ("events", lambda x: int("home_run" in x.values))
        agg_kwargs["ab_count"]  = ("events", "count")
    if "launch_speed" in sc.columns:
        agg_kwargs["max_ev"]    = ("launch_speed", lambda x: x.dropna().max() if x.dropna().size > 0 else np.nan)
        agg_kwargs["avg_ev"]    = ("launch_speed", lambda x: x.dropna().mean() if x.dropna().size > 0 else np.nan)
    if "launch_angle" in sc.columns:
        agg_kwargs["sweet_spot_pct"] = (
            "launch_angle",
            lambda x: float(((x >= 8) & (x <= 32)).sum() / max(x.notna().sum(), 1)),
        )

    if not agg_kwargs:
        return pd.DataFrame()

    game_agg = sc.groupby(group_cols).agg(**agg_kwargs).reset_index()
    game_agg["season"] = year

    # Most common starting pitcher for each batter in each game
    if "pitcher" in sc.columns:
        pit_per_game = (
            sc.groupby(["game_pk", "batter"])["pitcher"]
            .agg(lambda x: x.mode().iloc[0] if not x.empty else np.nan)
            .reset_index()
            .rename(columns={"pitcher": "opposing_pitcher_id"})
        )
        game_agg = game_agg.merge(pit_per_game, on=["game_pk", "batter"], how="left")

    if "hit_hr" not in game_agg.columns:
        game_agg["hit_hr"] = 0

    return game_agg


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_game_level_dataset(
    years: list = None,
    cache_path: Path = CACHE_PATH,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Build or load the game-level training dataset.

    Returns DataFrame with one row per (batter, game_pk, game_date).
    Caches to parquet — subsequent calls skip the download.

    Args:
        years: list of int seasons to include (default: all in SEASON_WINDOWS)
        cache_path: parquet cache location
        force_rebuild: if True, ignore cache and re-download
    """
    if not force_rebuild and cache_path.exists():
        logger.info("Loading cached game-level dataset from %s", cache_path)
        df = pd.read_parquet(cache_path)
        logger.info("Loaded %d batter-game rows", len(df))
        return df

    if years is None:
        years = sorted(SEASON_WINDOWS.keys())

    all_frames = []
    for yr in years:
        if yr not in SEASON_WINDOWS:
            logger.warning("No season window defined for %d — skipping", yr)
            continue
        start, end = SEASON_WINDOWS[yr]
        sc = _fetch_statcast_season(start, end)
        if sc.empty:
            logger.warning("No data for %d", yr)
            continue
        agg = _aggregate_to_game_level(sc, yr)
        if not agg.empty:
            all_frames.append(agg)
            logger.info("Season %d: %d batter-game rows, HR rate=%.3f",
                        yr, len(agg), agg["hit_hr"].mean() if "hit_hr" in agg.columns else 0)

    if not all_frames:
        logger.error("No game-level data was built. Check pybaseball installation.")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Saved %d rows to %s  |  Overall HR rate: %.4f",
                len(df), cache_path, df.get("hit_hr", pd.Series([0])).mean())
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build game-level Statcast training dataset")
    parser.add_argument("--seasons", nargs="+", type=int, default=None,
                        help="Seasons to include (e.g. --seasons 2023 2024 2025)")
    parser.add_argument("--force", action="store_true", help="Ignore cache and re-download")
    args = parser.parse_args()

    df = build_game_level_dataset(years=args.seasons, force_rebuild=args.force)
    if not df.empty:
        print(f"\nDataset: {len(df):,} batter-game rows")
        if "hit_hr" in df.columns:
            print(f"HR rate:  {df['hit_hr'].mean():.4f}  ({int(df['hit_hr'].sum())} HRs)")
        print(f"Seasons:  {sorted(df['season'].unique()) if 'season' in df.columns else 'unknown'}")
        print(f"Cache:    {CACHE_PATH}")
