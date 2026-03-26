"""
Current-season roster fetcher.
Fetches active rosters for all 30 MLB teams from mlb-statsapi and caches
them to data/rosters_{season}.json.  Cache refreshes after 24 hours.

Provides:
  get_current_roster_map(season)  → {player_full_name_lower: team_abbr}
  get_player_team(player_name, season) → "NYY" | None
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import statsapi

logger = logging.getLogger(__name__)

DATA_DIR    = Path("data")
CACHE_TTL_H = 24          # refresh after 24 hours

# Same abbreviation map used across the project
TEAM_ID_TO_ABBR = {
    133: "OAK", 134: "PIT", 135: "SD",  136: "SEA", 137: "SF",
    138: "STL", 139: "TB",  140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CHW", 146: "MIA", 147: "NYY",
    158: "MIL", 108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS",
    112: "CHC", 113: "CIN", 114: "CLE", 115: "COL", 116: "DET",
    117: "HOU", 118: "KC",  119: "LAD", 120: "WSH", 121: "NYM",
}


def _cache_path(season: int) -> Path:
    return DATA_DIR / f"rosters_{season}.json"


def _cache_is_fresh(season: int) -> bool:
    p = _cache_path(season)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
        fetched_at = datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
        return datetime.now() - fetched_at < timedelta(hours=CACHE_TTL_H)
    except Exception:
        return False


def _load_cache(season: int) -> dict:
    try:
        return json.loads(_cache_path(season).read_text()).get("roster_map", {})
    except Exception:
        return {}


def _save_cache(season: int, roster_map: dict):
    DATA_DIR.mkdir(exist_ok=True)
    _cache_path(season).write_text(json.dumps({
        "fetched_at": datetime.now().isoformat(),
        "season":     season,
        "roster_map": roster_map,
    }, indent=2))


def _parse_roster_text(text: str) -> list:
    """
    Parse statsapi.roster() text output into a list of player names.
    Each line looks like: '#99  RF  Aaron Judge'
    """
    names = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Drop the jersey number and position tokens (first two tokens)
        parts = line.split()
        if len(parts) >= 3:
            # parts[0] = '#99', parts[1] = 'RF', parts[2:] = name
            names.append(" ".join(parts[2:]))
    return names


def fetch_all_rosters(season: int = 2026) -> dict:
    """
    Fetch active rosters for all 30 teams via statsapi.roster().
    Returns {player_full_name_lower: team_abbr}.
    """
    logger.info("Fetching %d rosters from MLB Stats API ...", season)

    # Get all 30 MLB teams
    try:
        team_list = statsapi.lookup_team("")   # returns list of {id, name, ...}
    except Exception as exc:
        logger.error("Failed to fetch team list: %s", exc)
        return {}

    roster_map: dict[str, str] = {}
    failed = 0

    for team in team_list:
        team_id   = team.get("id")
        team_abbr = TEAM_ID_TO_ABBR.get(team_id) or team.get("teamCode", "???").upper()

        try:
            text = statsapi.roster(team_id, rosterType="active", season=season)
            for name in _parse_roster_text(text):
                if name:
                    roster_map[name.lower()] = team_abbr
        except Exception as exc:
            logger.warning("Roster fetch failed for %s (%s): %s", team_abbr, team_id, exc)
            failed += 1

    logger.info("Built roster map: %d players across %d teams (%d failures)",
                len(roster_map), len(team_list) - failed, failed)
    return roster_map


def get_current_roster_map(season: int = 2026) -> dict:
    """
    Return {player_full_name_lower: team_abbr} using a 24-hour disk cache.
    """
    if _cache_is_fresh(season):
        return _load_cache(season)

    roster_map = fetch_all_rosters(season)
    if roster_map:
        _save_cache(season, roster_map)
    else:
        # Fall back to stale cache rather than returning empty
        stale = _load_cache(season)
        if stale:
            logger.warning("Using stale roster cache (fetch failed)")
            return stale

    return roster_map


def get_player_team(player_name: str, season: int = 2026) -> str | None:
    """
    Return current team abbreviation for a player name, or None if not found.
    Tries exact match first, then partial match.
    """
    rmap = get_current_roster_map(season)
    key  = player_name.lower().strip()

    # Exact match
    if key in rmap:
        return rmap[key]

    # Partial: "Aaron Judge" found by "judge, aaron" style
    for roster_name, abbr in rmap.items():
        parts = key.split()
        if len(parts) >= 2 and parts[-1] in roster_name and parts[0] in roster_name:
            return abbr

    return None


def refresh_rosters(season: int = 2026):
    """Force-refresh roster cache regardless of TTL."""
    roster_map = fetch_all_rosters(season)
    if roster_map:
        _save_cache(season, roster_map)
        logger.info("Roster cache refreshed: %d players", len(roster_map))
    return roster_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fetch current MLB rosters")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--force",  action="store_true", help="Bypass cache")
    parser.add_argument("--team",   type=str, default=None, help="Show roster for team abbr")
    args = parser.parse_args()

    rmap = refresh_rosters(args.season) if args.force else get_current_roster_map(args.season)

    if args.team:
        team_players = sorted([n for n, t in rmap.items() if t == args.team.upper()])
        print(f"\n{args.team.upper()} active roster ({len(team_players)} players):")
        for p in team_players:
            print(f"  {p.title()}")
    else:
        from collections import Counter
        counts = Counter(rmap.values())
        print(f"\nRoster map: {len(rmap)} players across {len(counts)} teams")
        for abbr, cnt in sorted(counts.items()):
            print(f"  {abbr:<5} {cnt} players")
