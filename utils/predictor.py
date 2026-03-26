"""
Daily prediction pipeline.
1. Fetch today's MLB schedule and lineups via mlb-statsapi
2. Build matchup feature matrix for each batter vs starting pitcher
3. Run XGBoost → output HR probability + projected HRs
4. Flag Best Bets by confidence tier
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsapi

from utils.data_fetcher import (
    fetch_fangraphs_batting,
    fetch_fangraphs_pitching,
    load_park_factors,
)
from utils.feature_engineer import (
    MODEL_FEATURES,
    build_3yr_weighted_fg,
    build_3yr_weighted_pitcher,
    build_matchup_features,
)
from utils.model_trainer import load_model
from utils.weather_fetcher import fetch_all_game_weather
from utils.roster_fetcher import get_current_roster_map

logger = logging.getLogger(__name__)

HIGH_CONF = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", 0.18))
MED_CONF  = float(os.getenv("MEDIUM_CONFIDENCE_THRESHOLD", 0.12))


def _data_year_for_date(game_date: str) -> int:
    """
    Return the best 'current year' for loading player data given a game date.
    If the game date is early in the season (before June) or in a prior year,
    use the prior full season so stats are meaningful rather than tiny samples.
    """
    from datetime import datetime
    dt = datetime.strptime(game_date, "%Y-%m-%d")
    # If date is before June, prior season stats are more complete than current
    if dt.month < 6:
        return dt.year - 1
    return dt.year


# ---------------------------------------------------------------------------
# Team name → abbreviation
# ---------------------------------------------------------------------------

TEAM_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
    "Athletics": "OAK",
}


def _abbr(full_name: str) -> str:
    for key, abbr in TEAM_NAME_TO_ABBR.items():
        if key in full_name:
            return abbr
    return full_name[:3].upper()


# ---------------------------------------------------------------------------
# Schedule + lineup helpers
# ---------------------------------------------------------------------------

def fetch_todays_games(game_date: str = None) -> list:
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")
    try:
        schedule = statsapi.schedule(date=game_date)
    except Exception as exc:
        logger.error("Failed to fetch schedule: %s", exc)
        return []

    games = []
    for g in schedule:
        games.append({
            "game_id": g.get("game_id"),
            "home_team": _abbr(g.get("home_name", "")),
            "away_team": _abbr(g.get("away_name", "")),
            "game_date": game_date,
            "game_hour_utc": 19,
            "probable_home_pitcher": g.get("home_probable_pitcher", "TBD"),
            "probable_away_pitcher": g.get("away_probable_pitcher", "TBD"),
        })
    logger.info("Found %d games on %s", len(games), game_date)
    return games


def fetch_lineups(game_id: int) -> dict:
    try:
        game = statsapi.get("game", {"gamePk": game_id})
        lineups = {"home": [], "away": []}
        box = game.get("liveData", {}).get("boxscore", {}).get("teams", {})
        for side in ("home", "away"):
            players = box.get(side, {}).get("battingOrder", [])
            for pid in players:
                player_info = box[side]["players"].get(f"ID{pid}", {})
                name = player_info.get("person", {}).get("fullName", "")
                if name:
                    lineups[side].append(name)
        return lineups
    except Exception as exc:
        logger.warning("Could not fetch lineup for game %s: %s", game_id, exc)
        return {"home": [], "away": []}


# ---------------------------------------------------------------------------
# Load player data (current year + 2 prior for 3yr weighted)
# ---------------------------------------------------------------------------

def _load_player_data(current_year: int) -> tuple:
    fg_bat_frames, fg_pit_frames = [], []
    for yr in range(current_year - 2, current_year + 1):
        try:
            fg_bat_frames.append(fetch_fangraphs_batting(yr))
            fg_pit_frames.append(fetch_fangraphs_pitching(yr))
        except Exception as exc:
            logger.warning("Could not load year %d: %s", yr, exc)

    fg_batting  = pd.concat([d for d in fg_bat_frames  if d is not None], ignore_index=True)
    fg_pitching = pd.concat([d for d in fg_pit_frames  if d is not None], ignore_index=True)
    return fg_batting, fg_pitching


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict_today(game_date: str = None) -> pd.DataFrame:
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    logger.info("Running predictions for %s ...", game_date)

    model, scaler   = load_model()
    park_factors    = load_park_factors()
    park_factor_map = dict(zip(park_factors["team"], park_factors["hr_park_factor"]))

    current_year    = _data_year_for_date(game_date)
    logger.info("Using player data year: %d", current_year)
    fg_batting, fg_pitching = _load_player_data(current_year)

    # Live 2026 roster map: {player_name_lower: current_team_abbr}
    # Used to (1) build correct batter pools when no lineup is confirmed,
    # and (2) override stale FanGraphs team assignments in output.
    game_season = int(game_date[:4])
    roster_map  = get_current_roster_map(game_season)
    logger.info("Loaded roster map: %d players for %d season", len(roster_map), game_season)

    # 3-year weighted batter features
    batter_weighted  = build_3yr_weighted_fg(fg_batting, current_year)
    pitcher_weighted = build_3yr_weighted_pitcher(fg_pitching, current_year)

    # Current-year slices (FanGraphs uses capital "Season")
    year_col    = "Season" if "Season" in fg_batting.columns else "season"
    fg_bat_cur  = fg_batting[fg_batting[year_col]  == current_year].copy()
    fg_pit_cur  = fg_pitching[fg_pitching[year_col] == current_year].copy()
    fg_bat_cur["hr_rate"] = fg_bat_cur["HR"] / fg_bat_cur["PA"].replace(0, np.nan)

    games = fetch_todays_games(game_date)
    if not games:
        logger.warning("No games found for %s", game_date)
        return pd.DataFrame()

    weather_map  = fetch_all_game_weather(games)
    predictions  = []

    for game in games:
        home        = game["home_team"]
        away        = game["away_team"]
        weather     = weather_map.get(home, {"temp_f": 72.0, "wind_speed_mph": 0.0,
                                              "wind_direction": "calm", "is_indoor": False})
        park_factor = park_factor_map.get(home, 100.0)
        lineups     = fetch_lineups(game["game_id"]) if game.get("game_id") else {}

        sides = [
            (home, away, game["probable_away_pitcher"], lineups.get("home", [])),
            (away, home, game["probable_home_pitcher"], lineups.get("away", [])),
        ]

        for batting_team, fielding_team, pitcher_name, batters in sides:
            # Pitcher row (current season FanGraphs)
            pit_match = fg_pit_cur[
                fg_pit_cur["Name"].str.contains(pitcher_name, case=False, na=False)
            ] if pitcher_name and pitcher_name != "TBD" else pd.DataFrame()
            pitcher_series = pit_match.iloc[0] if not pit_match.empty else pd.Series()

            # Batter pool: confirmed lineup > live roster > FanGraphs team fallback
            if batters:
                # Lineup confirmed — match by name
                batter_pool = fg_bat_cur[
                    fg_bat_cur["Name"].apply(
                        lambda n: any(b.lower() in str(n).lower() for b in batters)
                    )
                ]
            elif roster_map:
                # Use live 2026 roster to find players currently on this team,
                # then match those names against FanGraphs historical stats.
                current_team_players = {
                    name for name, abbr in roster_map.items()
                    if abbr == batting_team
                }
                batter_pool = fg_bat_cur[
                    fg_bat_cur["Name"].apply(
                        lambda n: str(n).lower() in current_team_players
                        or any(
                            cp in str(n).lower() or str(n).lower() in cp
                            for cp in current_team_players
                            if len(cp) > 5  # skip very short names
                        )
                    )
                ]
                if batter_pool.empty:
                    # Final fallback: old FanGraphs team column
                    batter_pool = fg_bat_cur[fg_bat_cur["Team"] == batting_team]
            else:
                batter_pool = fg_bat_cur[fg_bat_cur["Team"] == batting_team]

            for _, batter_series in batter_pool.iterrows():
                batter_id = batter_series.get("IDfg")

                # Weighted batter features
                bw_row = batter_weighted[batter_weighted["IDfg"] == batter_id]
                batter_w = bw_row.iloc[0] if not bw_row.empty else pd.Series()

                # Weighted pitcher features
                pitcher_fg_id = pitcher_series.get("IDfg") if not pitcher_series.empty else None
                pw_row = pitcher_weighted[pitcher_weighted["IDfg"] == pitcher_fg_id] \
                         if pitcher_fg_id is not None else pd.DataFrame()
                # Merge current + weighted pitcher into one Series
                combined_pitcher = pd.concat([
                    pitcher_series,
                    pw_row.iloc[0] if not pw_row.empty else pd.Series(),
                ])

                features = build_matchup_features(
                    batter_row     = batter_series,
                    batter_weighted= batter_w,
                    pitcher_row    = combined_pitcher,
                    park_factor    = park_factor,
                    temp_f         = weather["temp_f"],
                    wind_speed_mph = weather["wind_speed_mph"],
                    wind_direction = weather["wind_direction"],
                    is_day_game    = False,
                )

                feat_df = pd.DataFrame([features])[MODEL_FEATURES]
                # Fill NaN with column median from training (approximate with 0 for simplicity)
                feat_df = feat_df.fillna(0)
                feat_scaled = scaler.transform(feat_df)
                hr_prob = float(model.predict_proba(feat_scaled)[0][1])

                confidence = (
                    "High"   if hr_prob >= HIGH_CONF else
                    "Medium" if hr_prob >= MED_CONF  else
                    "Low"
                )

                player_name = batter_series.get("Name", "Unknown")
                # Use live roster team; fall back to game schedule team
                live_team = (
                    roster_map.get(player_name.lower())
                    or batting_team
                )

                predictions.append({
                    "date":          game_date,
                    "player":        player_name,
                    "team":          live_team,
                    "opponent":      fielding_team,
                    "pitcher":       pitcher_name,
                    "hr_probability": round(hr_prob, 4),
                    "confidence":    confidence,
                    "park_factor":   park_factor,
                    "temp_f":        weather["temp_f"],
                    "wind_speed_mph": weather["wind_speed_mph"],
                    "wind_direction": weather["wind_direction"],
                    "is_indoor":     weather.get("is_indoor", False),
                    "home_game":     batting_team == home,
                })

    df = pd.DataFrame(predictions)
    if not df.empty:
        df = df.sort_values("hr_probability", ascending=False).reset_index(drop=True)
        logger.info("Generated %d predictions (%d High confidence)",
                    len(df), (df["confidence"] == "High").sum())
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Game date YYYY-MM-DD (default: today)")
    parser.add_argument("--save", action="store_true",
                        help="Save predictions to Excel tracker")
    args = parser.parse_args()

    preds = predict_today(args.date)
    if not preds.empty:
        print(preds[["player", "team", "pitcher", "hr_probability", "confidence",
                      "park_factor", "wind_direction"]].to_string(index=False))
        if args.save:
            from tracker.prediction_tracker import save_predictions
            save_predictions(preds)
            print("Saved to MLB_HR_Predictions.xlsx")
    else:
        print("No predictions generated.")
