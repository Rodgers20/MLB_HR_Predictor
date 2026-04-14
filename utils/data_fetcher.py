"""
Data fetcher — wraps pybaseball with local CSV caching.
Fetches Statcast leaderboard, FanGraphs batting/pitching stats, and park factors.
Respects rate limits; never re-fetches data that's already cached.
"""

import os
import time
import logging
import pandas as pd
from pathlib import Path
from pybaseball import (
    batting_stats,
    pitching_stats,
    statcast_batter_exitvelo_barrels,
    statcast_pitcher_exitvelo_barrels,
    playerid_lookup,
    cache,
)

logger = logging.getLogger(__name__)

RAW_DIR = Path(os.getenv("DATA_CACHE_DIR", "data")) / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Enable pybaseball's built-in cache
cache.enable()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_or_fetch(cache_path: Path, fetch_fn, *args, **kwargs) -> pd.DataFrame:
    """Return cached CSV if it exists, otherwise call fetch_fn and save."""
    if cache_path.exists():
        logger.info("Cache hit: %s", cache_path)
        return pd.read_csv(cache_path)
    logger.info("Cache miss — fetching %s", cache_path.name)
    df = fetch_fn(*args, **kwargs)
    if df is not None and not df.empty:
        df.to_csv(cache_path, index=False)
        logger.info("Saved %d rows → %s", len(df), cache_path)
    return df


# ---------------------------------------------------------------------------
# Statcast exit-velocity / barrel leaderboards
# ---------------------------------------------------------------------------

def fetch_statcast_batter_leaderboard(year: int) -> pd.DataFrame:
    """
    Fetch Statcast batter leaderboard for a given year.
    Columns include: exit_velocity_avg, launch_angle_avg, barrel_batted_rate,
    hard_hit_percent, xwoba, xslg, xba, player_id, last_name, first_name, etc.
    """
    path = RAW_DIR / f"statcast_batter_{year}.csv"
    df = _load_or_fetch(
        path,
        statcast_batter_exitvelo_barrels,
        year,
        minBBE=20,
    )
    if df is not None:
        df["season"] = year
    return df


def fetch_statcast_pitcher_leaderboard(year: int) -> pd.DataFrame:
    """Fetch Statcast pitcher leaderboard for a given year."""
    path = RAW_DIR / f"statcast_pitcher_{year}.csv"
    df = _load_or_fetch(
        path,
        statcast_pitcher_exitvelo_barrels,
        year,
        minBBE=20,
    )
    if df is not None:
        df["season"] = year
    return df


# ---------------------------------------------------------------------------
# FanGraphs season batting / pitching stats
# ---------------------------------------------------------------------------

def _fetch_current_season_with_fallback(
    path: Path, year: int, fetch_fn, *args, **kwargs
) -> pd.DataFrame:
    """Fetch fresh data for the current season; fall back to stale cache on error.

    Unlike the old _invalidate_if_stale approach, the cached file is NOT deleted
    before the fetch attempt.  This means a 403 / network error keeps the last
    good data alive rather than leaving the system with nothing.
    """
    from datetime import date as _date, datetime as _dt

    # Non-current seasons: use the simple cache — they never change.
    if year != _date.today().year:
        return _load_or_fetch(path, fetch_fn, *args, **kwargs)

    # Current season: re-fetch if the cached file is from a previous day.
    if path.exists():
        file_date = _dt.fromtimestamp(path.stat().st_mtime).date()
        if file_date >= _date.today():
            logger.info("Cache hit: %s", path)
            return pd.read_csv(path)
        logger.info("Stale current-season cache — refreshing %s", path.name)
    else:
        logger.info("Cache miss — fetching %s", path.name)

    # Attempt live fetch without deleting the old file first.
    try:
        df = fetch_fn(*args, **kwargs)
        if df is not None and not df.empty:
            df.to_csv(path, index=False)
            logger.info("Saved %d rows → %s", len(df), path)
            df["season"] = year
            return df
        logger.warning("Empty response fetching %s", path.name)
    except Exception as exc:
        logger.warning("Fetch failed for %s: %s — keeping stale cache", path.name, exc)

    # Fall back to whatever is on disk (may be stale; still better than nothing).
    if path.exists():
        logger.warning("Using stale cache for %s", path.name)
        df = pd.read_csv(path)
        df["season"] = year
        return df

    return pd.DataFrame()


def fetch_fangraphs_batting(year: int) -> pd.DataFrame:
    """
    FanGraphs season batting stats including ISO, Pull%, FB%, HR/FB, wRC+, WAR.
    Uses qual=0 to get all players (not just qualified).
    Current-season cache is refreshed daily; stale data is kept on fetch failure.
    """
    path = RAW_DIR / f"fangraphs_batting_{year}.csv"
    df = _fetch_current_season_with_fallback(path, year, batting_stats, year, year, qual=0)
    if df is not None and "season" not in df.columns:
        df["season"] = year
    return df


def fetch_fangraphs_pitching(year: int) -> pd.DataFrame:
    """
    FanGraphs season pitching stats including FIP, xFIP, GB%, HR/9, HR/FB.
    Current-season cache is refreshed daily; stale data is kept on fetch failure.
    """
    path = RAW_DIR / f"fangraphs_pitching_{year}.csv"
    df = _fetch_current_season_with_fallback(path, year, pitching_stats, year, year, qual=0)
    if df is not None and "season" not in df.columns:
        df["season"] = year
    return df


# ---------------------------------------------------------------------------
# Multi-year bulk fetch
# ---------------------------------------------------------------------------

def fetch_all_years(start_year: int = 2017, end_year: int = 2025) -> dict:
    """
    Fetch all data for every year in range.
    Returns dict with keys: statcast_batters, statcast_pitchers,
                             fg_batting, fg_pitching
    Each is a concatenated DataFrame across all years.
    """
    sc_batters, sc_pitchers, fg_bat, fg_pit = [], [], [], []

    for year in range(start_year, end_year + 1):
        logger.info("Fetching year %d ...", year)
        try:
            sc_batters.append(fetch_statcast_batter_leaderboard(year))
            time.sleep(1)  # throttle
            sc_pitchers.append(fetch_statcast_pitcher_leaderboard(year))
            time.sleep(1)
            fg_bat.append(fetch_fangraphs_batting(year))
            time.sleep(1)
            fg_pit.append(fetch_fangraphs_pitching(year))
            time.sleep(1)
        except Exception as exc:
            logger.warning("Failed year %d: %s", year, exc)

    return {
        "statcast_batters": pd.concat([d for d in sc_batters if d is not None], ignore_index=True),
        "statcast_pitchers": pd.concat([d for d in sc_pitchers if d is not None], ignore_index=True),
        "fg_batting": pd.concat([d for d in fg_bat if d is not None], ignore_index=True),
        "fg_pitching": pd.concat([d for d in fg_pit if d is not None], ignore_index=True),
    }


# ---------------------------------------------------------------------------
# Park factors
# ---------------------------------------------------------------------------

PARK_FACTORS_URL = (
    "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
    "?type=year&year=2024&batSide=&stat=index_Hits&condition=is&rolling=&sort=1&sortDir=asc"
)

MANUAL_PARK_FACTORS = {
    # team_abbr: HR park factor (100=average)
    "COL": 116, "CIN": 111, "NYY": 110, "PHI": 108, "BOS": 107,
    "MIL": 106, "HOU": 105, "ATL": 104, "STL": 103, "LAD": 102,
    "NYM": 101, "CHC": 101, "MIN": 100, "TOR": 100, "TB": 99,
    "DET": 99, "BAL": 98, "TEX": 98, "KC": 97, "WSH": 97,
    "SF": 96, "CLE": 96, "ARI": 95, "LAA": 95, "SEA": 94,
    "CHW": 94, "PIT": 94, "MIA": 93, "SD": 92, "OAK": 91,
}


def load_park_factors() -> pd.DataFrame:
    """
    Load HR park factors. Returns DataFrame with columns: team, hr_park_factor.
    Uses manually curated values if CSV not present.
    """
    path = Path(os.getenv("DATA_CACHE_DIR", "data")) / "park_factors.csv"
    if path.exists():
        return pd.read_csv(path)

    df = pd.DataFrame([
        {"team": team, "hr_park_factor": factor}
        for team, factor in MANUAL_PARK_FACTORS.items()
    ])
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fetch MLB data")
    parser.add_argument("--years", nargs=2, type=int, default=[2017, 2025],
                        metavar=("START", "END"))
    args = parser.parse_args()

    logger.info("Fetching data %d–%d ...", args.years[0], args.years[1])
    data = fetch_all_years(args.years[0], args.years[1])
    for key, df in data.items():
        logger.info("%s: %d rows", key, len(df))
    logger.info("Done.")
