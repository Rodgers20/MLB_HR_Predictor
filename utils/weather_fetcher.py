"""
Weather fetcher — pulls game-time weather from Open-Meteo (free, no API key).
Uses stadium lat/lon coordinates to get temperature, wind speed, and wind direction.
"""

import logging
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

# Stadium coordinates: team_abbr -> (lat, lon, elevation_ft)
STADIUM_COORDS = {
    "ARI": (33.4455, -112.0667, 1082),   # Chase Field (retractable roof)
    "ATL": (33.8908, -84.4678, 1050),    # Truist Park
    "BAL": (39.2838, -76.6218, 20),      # Camden Yards
    "BOS": (42.3467, -71.0972, 20),      # Fenway Park
    "CHC": (41.9484, -87.6553, 595),     # Wrigley Field
    "CHW": (41.8300, -87.6339, 595),     # Guaranteed Rate Field
    "CIN": (39.0979, -84.5082, 490),     # Great American Ball Park
    "CLE": (41.4961, -81.6852, 650),     # Progressive Field
    "COL": (39.7559, -104.9942, 5200),   # Coors Field (highest elevation)
    "DET": (42.3390, -83.0485, 600),     # Comerica Park
    "HOU": (29.7572, -95.3555, 43),      # Minute Maid Park (retractable roof)
    "KC":  (39.0517, -94.4803, 910),     # Kauffman Stadium
    "LAA": (33.8003, -117.8827, 160),    # Angel Stadium
    "LAD": (34.0739, -118.2400, 515),    # Dodger Stadium
    "MIA": (25.7781, -80.2197, 8),       # loanDepot Park (retractable roof)
    "MIL": (43.0282, -87.9712, 635),     # American Family Field (retractable roof)
    "MIN": (44.9817, -93.2783, 815),     # Target Field
    "NYM": (40.7571, -73.8458, 20),      # Citi Field
    "NYY": (40.8296, -73.9262, 20),      # Yankee Stadium
    "OAK": (37.7516, -122.2005, 25),     # Oakland Coliseum
    "PHI": (39.9061, -75.1665, 20),      # Citizens Bank Park
    "PIT": (40.4469, -80.0057, 730),     # PNC Park
    "SD":  (32.7073, -117.1566, 20),     # Petco Park
    "SEA": (47.5914, -122.3325, 20),     # T-Mobile Park (retractable roof)
    "SF":  (37.7786, -122.3893, 10),     # Oracle Park
    "STL": (38.6226, -90.1928, 465),     # Busch Stadium
    "TB":  (27.7682, -82.6534, 10),      # Tropicana Field (dome)
    "TEX": (32.7512, -97.0832, 550),     # Globe Life Field (retractable roof)
    "TOR": (43.6414, -79.3894, 250),     # Rogers Centre (retractable roof)
    "WSH": (38.8730, -77.0074, 20),      # Nationals Park
}

INDOOR_STADIUMS = {"ARI", "HOU", "MIA", "MIL", "SEA", "TB", "TEX", "TOR"}
OPEN_AIR_ONLY = {k for k in STADIUM_COORDS if k not in INDOOR_STADIUMS}

OPEN_METEO_FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL   = "https://archive-api.open-meteo.com/v1/archive"

# Wind direction: degrees → category
def _degrees_to_direction(degrees: float, stadium_orientation: float = 0.0) -> str:
    """
    Convert wind direction in degrees to In/Out/Cross/Calm relative to stadium.
    stadium_orientation: degrees from home plate to center field.
    Approximation: tailwind (out) = wind coming from home → CF, headwind (in) = opposite.
    """
    if degrees is None:
        return "calm"
    # Relative angle between wind and stadium axis
    relative = (degrees - stadium_orientation) % 360
    if relative < 45 or relative > 315:
        return "out"    # wind blowing toward CF / bleachers
    elif 135 < relative < 225:
        return "in"     # wind blowing toward home plate
    else:
        return "cross"


def fetch_game_weather(team: str, game_date: str, game_hour_utc: int = 19) -> dict:
    """
    Fetch weather for a stadium on a given game date.

    Args:
        team: Team abbreviation (home team)
        game_date: YYYY-MM-DD
        game_hour_utc: Approximate hour of first pitch in UTC (default 19 = 3pm ET)

    Returns:
        dict with keys: temp_f, wind_speed_mph, wind_direction, is_indoor
    """
    if team in INDOOR_STADIUMS:
        return {
            "temp_f": 72.0,
            "wind_speed_mph": 0.0,
            "wind_direction": "calm",
            "is_indoor": True,
        }

    if team not in STADIUM_COORDS:
        logger.warning("Unknown team '%s', using neutral weather", team)
        return {"temp_f": 72.0, "wind_speed_mph": 0.0, "wind_direction": "calm", "is_indoor": False}

    lat, lon, _ = STADIUM_COORDS[team]

    from datetime import datetime, date as _date
    is_past = datetime.strptime(game_date, "%Y-%m-%d").date() < _date.today()
    url = OPEN_METEO_ARCHIVE_URL if is_past else OPEN_METEO_FORECAST_URL

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/New_York",
        "start_date": game_date,
        "end_date": game_date,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        hours = hourly.get("time", [])

        # Find the hour closest to game time
        target_hour = f"{game_date}T{game_hour_utc:02d}:00"
        idx = 0
        if target_hour in hours:
            idx = hours.index(target_hour)

        temp_f = hourly["temperature_2m"][idx]
        wind_speed = hourly["wind_speed_10m"][idx]
        wind_deg = hourly["wind_direction_10m"][idx]
        wind_dir = _degrees_to_direction(wind_deg)

        return {
            "temp_f": round(temp_f, 1),
            "wind_speed_mph": round(wind_speed, 1),
            "wind_direction": wind_dir,
            "is_indoor": False,
        }

    except Exception as exc:
        logger.warning("Weather fetch failed for %s on %s: %s", team, game_date, exc)
        return {"temp_f": 72.0, "wind_speed_mph": 0.0, "wind_direction": "calm", "is_indoor": False}


def fetch_all_game_weather(games: list) -> dict:
    """
    Fetch weather for multiple games.

    Args:
        games: list of dicts with keys: home_team, game_date, game_hour_utc (optional)

    Returns:
        dict keyed by home_team -> weather dict
    """
    results = {}
    for game in games:
        team = game["home_team"]
        date = game["game_date"]
        hour = game.get("game_hour_utc", 19)
        results[team] = fetch_game_weather(team, date, hour)
    return results
