"""
Results fetcher — pulls actual HR totals for every batter from completed MLB games.
Uses mlb-statsapi boxscore_data which returns pre-formatted batter lines
with hr, name, personId fields directly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import date, timedelta

import statsapi

logger = logging.getLogger(__name__)


def fetch_results_for_date(game_date: str = None) -> list[dict]:
    """
    Fetch actual HR results for all batters on a given date.

    Args:
        game_date: YYYY-MM-DD (defaults to yesterday)

    Returns:
        List of dicts: {player, date, actual_hrs, game_id, team}
    """
    if game_date is None:
        game_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("Fetching results for %s ...", game_date)

    try:
        schedule = statsapi.schedule(date=game_date)
    except Exception as exc:
        logger.error("Failed to fetch schedule: %s", exc)
        return []

    results = []
    completed = [g for g in schedule if g.get("status") in ("Final", "Game Over")]
    logger.info("%d completed games on %s", len(completed), game_date)

    for game in completed:
        game_id = game["game_id"]
        try:
            bs = statsapi.boxscore_data(game_id)
        except Exception as exc:
            logger.warning("Boxscore failed for game %s: %s", game_id, exc)
            continue

        for side in ("away", "home"):
            batter_list_key = f"{side}Batters"
            batters = bs.get(batter_list_key, [])
            team_info = bs.get("teamInfo", {}).get(side, {})
            team_abbr = _team_abbr(team_info.get("teamName", ""))

            for b in batters:
                # Skip header row
                if b.get("personId", 0) == 0:
                    continue
                name = b.get("name", "").strip()
                # hr field is a string from the API ('0', '1', '2', etc.)
                raw_hr = b.get("hr", "0")
                try:
                    hr = int(raw_hr)
                except (ValueError, TypeError):
                    hr = 0

                # Normalize name: "Judge, Aar" → "Aaron Judge" style matching
                full_name = _normalize_name(name)

                results.append({
                    "player":      full_name,
                    "date":        game_date,
                    "actual_hrs":  hr,
                    "game_id":     game_id,
                    "team":        team_abbr,
                    "person_id":   b.get("personId"),
                })

    logger.info("Fetched results for %d batters (date: %s)", len(results), game_date)
    return results


def _normalize_name(name: str) -> str:
    """
    Convert 'Last, First' → 'First Last' for matching against FanGraphs names.
    Handles: 'Judge, Aar' → 'Aar Judge', 'Ohtani, Sho' → 'Sho Ohtani'
    Full names are used as-is when no comma.
    """
    if "," in name:
        parts = name.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name.strip()


TEAM_NAME_MAP = {
    "yankees": "NYY", "red sox": "BOS", "rays": "TB", "blue jays": "TOR",
    "orioles": "BAL", "white sox": "CHW", "guardians": "CLE", "tigers": "DET",
    "royals": "KC", "twins": "MIN", "astros": "HOU", "angels": "LAA",
    "athletics": "OAK", "mariners": "SEA", "rangers": "TEX", "braves": "ATL",
    "marlins": "MIA", "mets": "NYM", "phillies": "PHI", "nationals": "WSH",
    "cubs": "CHC", "reds": "CIN", "brewers": "MIL", "pirates": "PIT",
    "cardinals": "STL", "diamondbacks": "ARI", "rockies": "COL",
    "dodgers": "LAD", "padres": "SD", "giants": "SF",
}


def _team_abbr(team_name: str) -> str:
    lower = team_name.lower()
    for key, abbr in TEAM_NAME_MAP.items():
        if key in lower:
            return abbr
    return team_name[:3].upper()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    results = fetch_results_for_date(args.date)
    hrs = [r for r in results if r["actual_hrs"] > 0]
    print(f"\nHR hitters on {args.date or 'yesterday'} ({len(hrs)} total):")
    for r in sorted(hrs, key=lambda x: -x["actual_hrs"]):
        print(f"  {r['player']:<25} {r['actual_hrs']} HR  ({r['team']})")
