"""
Cached MLB player headshot URL lookup.
Uses statsapi.lookup_player() with a disk-backed JSON cache so subsequent
calls are instant. Falls back to the MLB generic avatar on any error.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import statsapi

logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/player_id_cache.json")
MLB_CDN = "https://img.mlbstatic.com/mlb-photos/image/upload"
FALLBACK = "d_people:generic:headshot:67:current.png"

# Module-level in-memory cache (loaded once from disk)
_id_cache: dict[str, int] = {}
_cache_loaded = False


def _load_cache():
    global _id_cache, _cache_loaded
    if _cache_loaded:
        return
    if CACHE_PATH.exists():
        try:
            _id_cache = json.loads(CACHE_PATH.read_text())
        except Exception:
            _id_cache = {}
    _cache_loaded = True


def _save_cache():
    CACHE_PATH.parent.mkdir(exist_ok=True)
    try:
        CACHE_PATH.write_text(json.dumps(_id_cache, indent=2, sort_keys=True))
    except Exception:
        pass


def lookup_player_id(player_name: str) -> int:
    """Return MLB player ID for a name (with disk-backed cache)."""
    _load_cache()
    key = player_name.lower().strip()
    if key in _id_cache:
        return _id_cache[key]
    try:
        results = statsapi.lookup_player(player_name)
        if results:
            pid = int(results[0]["id"])
            _id_cache[key] = pid
            _save_cache()
            return pid
    except Exception as exc:
        logger.debug("ID lookup failed for %s: %s", player_name, exc)
    _id_cache[key] = 0  # cache miss so we don't retry every page load
    _save_cache()
    return 0


def headshot_url(player_name: str, width: int = 120) -> str:
    """Return MLB headshot CDN URL (generic avatar if player not found)."""
    pid = lookup_player_id(player_name)
    return f"{MLB_CDN}/{FALLBACK}/w_{width},q_auto:best/v1/people/{pid}/headshot/67/current"


def batch_headshot_urls(player_names: list, width: int = 80) -> dict:
    """
    Fetch headshot URLs for many players concurrently.
    Returns {player_name: url}.  Cached players are instant; uncached ones
    are looked up in parallel (max 12 threads, 6 s timeout).
    """
    _load_cache()
    uncached = [n for n in player_names if n.lower().strip() not in _id_cache]

    if uncached:
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {executor.submit(lookup_player_id, n): n for n in uncached}
            for _ in as_completed(futures, timeout=6):
                pass  # side-effect: populates cache

    return {name: headshot_url(name, width) for name in player_names}
