"""
At-bat log fetcher for MLB HR Predictor.

Fetches Statcast pitch-by-pitch data for a specific batter via pybaseball,
filters to plate-appearance endings, enriches with pitcher names, and
computes current-season aggregated stats.

Cache policy:
  - Prior seasons: permanent (stats don't change)
  - Current season: invalidated daily so stats stay up to date
"""

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.getenv("DATA_CACHE_DIR", "data")) / "raw"
PLAYER_ID_CACHE  = Path("data/player_id_cache.json")
PITCHER_ID_CACHE = Path("data/pitcher_id_cache.json")

# Approximate regular-season opening day per year
SEASON_STARTS = {
    2021: "2021-04-01",
    2022: "2022-04-07",
    2023: "2023-03-30",
    2024: "2024-03-28",
    2025: "2025-03-27",
    2026: "2026-03-27",
}

# ---------------------------------------------------------------------------
# At-bat result taxonomy
# ---------------------------------------------------------------------------

AB_RESULT_MAP: dict[str, tuple[str, str]] = {
    "home_run":                  ("HR",   "#ff6b00"),
    "triple":                    ("3B",   "#22c55e"),
    "double":                    ("2B",   "#22c55e"),
    "single":                    ("1B",   "#22c55e"),
    "walk":                      ("BB",   "#3b82f6"),
    "intent_walk":               ("IBB",  "#3b82f6"),
    "hit_by_pitch":              ("HBP",  "#3b82f6"),
    "strikeout":                 ("K",    "#ef4444"),
    "strikeout_double_play":     ("K-DP", "#ef4444"),
    "field_out":                 ("Out",  "#8e909c"),
    "force_out":                 ("Out",  "#8e909c"),
    "grounded_into_double_play": ("GIDP", "#8e909c"),
    "double_play":               ("DP",   "#8e909c"),
    "field_error":               ("E",    "#fbbf24"),
    "fielders_choice":           ("FC",   "#8e909c"),
    "fielders_choice_out":       ("FC",   "#8e909c"),
    "sac_fly":                   ("SF",   "#8e909c"),
    "sac_bunt":                  ("SAC",  "#8e909c"),
    "catcher_interf":            ("CI",   "#8e909c"),
}

HIT_EVENTS = {"single", "double", "triple", "home_run"}

# Events that count as official at-bats (BB, HBP, SF, SAC bunt do NOT)
AB_EVENTS = {
    "single", "double", "triple", "home_run",
    "strikeout", "strikeout_double_play",
    "field_out", "force_out", "grounded_into_double_play",
    "double_play", "field_error", "fielders_choice", "fielders_choice_out",
}

BB_TYPE_SHORT = {
    "fly_ball":    "FB",
    "ground_ball": "GB",
    "line_drive":  "LD",
    "popup":       "PU",
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_json(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
    except Exception as exc:
        logger.warning("Could not save cache %s: %s", path, exc)


def _invalidate_if_stale(path: Path, season: int) -> None:
    """Delete a current-season cache file if it's from a prior date."""
    if season == date.today().year and path.exists():
        file_date = datetime.fromtimestamp(path.stat().st_mtime).date()
        if file_date < date.today():
            logger.info("Stale current-season cache — refreshing %s", path.name)
            path.unlink()


# ---------------------------------------------------------------------------
# Player / pitcher ID resolution
# ---------------------------------------------------------------------------

def _get_mlbam_id(player_name: str) -> int | None:
    """
    Return the MLBAM ID for a player name.
    Checks the shared player_id_cache.json first; falls back to pybaseball lookup.
    """
    cache = _load_json(PLAYER_ID_CACHE)
    key   = player_name.lower().strip()

    if key in cache:
        val = cache[key]
        return int(val) if val and int(val) > 0 else None

    try:
        from pybaseball import playerid_lookup
        parts = key.split()
        last, first = parts[-1], parts[0]
        result = playerid_lookup(last, first, fuzzy=True)
        if result is not None and not result.empty:
            mlbam_id = int(result["key_mlbam"].iloc[0])
            if mlbam_id > 0:
                cache[key] = mlbam_id
                _save_json(PLAYER_ID_CACHE, cache)
                return mlbam_id
    except Exception as exc:
        logger.warning("MLBAM lookup failed for '%s': %s", player_name, exc)

    return None


def _resolve_pitcher_names(pitcher_ids) -> dict:
    """
    Return {mlbam_id (int): name (str)} for a collection of pitcher MLBAM IDs.
    Uses pitcher_id_cache.json to avoid repeated API calls.
    """
    cache = _load_json(PITCHER_ID_CACHE)

    unique_ids = [int(pid) for pid in pd.to_numeric(
        pd.Series(pitcher_ids), errors="coerce"
    ).dropna().unique()]
    missing = [pid for pid in unique_ids if str(pid) not in cache]

    if missing:
        try:
            from pybaseball import playerid_reverse_lookup
            result = playerid_reverse_lookup(missing, key_type="mlbam")
            if result is not None and not result.empty:
                for _, row in result.iterrows():
                    mid  = str(int(row.get("key_mlbam", 0) or 0))
                    name = f"{row.get('name_first', '')} {row.get('name_last', '')}".strip()
                    if mid != "0" and name:
                        cache[mid] = name
                _save_json(PITCHER_ID_CACHE, cache)
        except Exception as exc:
            logger.warning("Pitcher reverse lookup failed: %s", exc)

    return {int(k): v for k, v in cache.items()}


# ---------------------------------------------------------------------------
# Raw fetch (with caching)
# ---------------------------------------------------------------------------

def _fetch_ab_log_raw(mlbam_id: int, season: int) -> pd.DataFrame:
    """
    Fetch Statcast pitch-by-pitch for a batter, caching to CSV.
    Current-season files are invalidated daily.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"ab_log_{mlbam_id}_{season}.csv"

    _invalidate_if_stale(cache_path, season)

    if cache_path.exists():
        logger.info("AB log cache hit: %s", cache_path.name)
        return pd.read_csv(cache_path, low_memory=False)

    start_dt = SEASON_STARTS.get(season, f"{season}-03-20")
    end_dt   = date.today().strftime("%Y-%m-%d")

    try:
        from pybaseball import statcast_batter
        logger.info("Fetching Statcast for MLBAM %d (%d season) ...", mlbam_id, season)
        df = statcast_batter(start_dt=start_dt, end_dt=end_dt, player_id=mlbam_id)
        if df is None or df.empty:
            logger.warning("No Statcast data returned for MLBAM %d", mlbam_id)
            return pd.DataFrame()
        # Regular season only
        if "game_type" in df.columns:
            df = df[df["game_type"] == "R"].copy()
        df.to_csv(cache_path, index=False)
        logger.info("Saved %d pitch rows for MLBAM %d → %s", len(df), mlbam_id, cache_path.name)
        return df
    except Exception as exc:
        logger.error("statcast_batter failed for MLBAM %d: %s", mlbam_id, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_player_ab_log(player_name: str, season: int) -> pd.DataFrame:
    """
    Return one row per completed plate appearance for *player_name* in *season*.

    Columns:
        game_date, opponent, batter_team, pitcher_name, p_throws,
        result_label, result_color,
        launch_speed, launch_angle, hit_distance_sc, bb_type_short,
        inning, count, outs_when_up,
        estimated_ba_using_speedangle, estimated_woba_using_speedangle,
        stand, game_pk, events
    """
    mlbam_id = _get_mlbam_id(player_name)
    if not mlbam_id:
        logger.warning("No MLBAM ID for '%s' — returning empty log", player_name)
        return pd.DataFrame()

    raw = _fetch_ab_log_raw(mlbam_id, season)
    if raw is None or raw.empty:
        return pd.DataFrame()

    # ── Filter to plate-appearance ending rows ─────────────────────────────
    pa = raw[
        raw["events"].notna() &
        (raw["events"].astype(str).str.strip() != "") &
        (raw["events"].astype(str) != "nan")
    ].copy()

    if pa.empty:
        return pd.DataFrame()

    # ── Result label / color ───────────────────────────────────────────────
    pa["result_label"] = pa["events"].map(
        lambda e: AB_RESULT_MAP.get(str(e), ("Out", "#8e909c"))[0]
    )
    pa["result_color"] = pa["events"].map(
        lambda e: AB_RESULT_MAP.get(str(e), ("Out", "#8e909c"))[1]
    )

    # ── Opponent / batter team ─────────────────────────────────────────────
    if "inning_topbot" in pa.columns and "home_team" in pa.columns:
        pa["batter_team"] = pa.apply(
            lambda r: r.get("home_team", "") if r.get("inning_topbot") == "Bot"
            else r.get("away_team", ""), axis=1,
        )
        pa["opponent"] = pa.apply(
            lambda r: r.get("away_team", "") if r.get("inning_topbot") == "Bot"
            else r.get("home_team", ""), axis=1,
        )
    elif "home_team" in pa.columns and "away_team" in pa.columns:
        pa["opponent"]    = pa["home_team"]
        pa["batter_team"] = pa["away_team"]
    else:
        pa["opponent"]    = "—"
        pa["batter_team"] = "—"

    # ── Pitcher names ──────────────────────────────────────────────────────
    if "pitcher" in pa.columns:
        pitcher_map = _resolve_pitcher_names(pa["pitcher"])
        pa["pitcher_name"] = pa["pitcher"].apply(
            lambda pid: pitcher_map.get(int(pid), f"#{int(pid)}")
            if pd.notna(pid) else "—"
        )
    else:
        pa["pitcher_name"] = "—"

    # ── Count ──────────────────────────────────────────────────────────────
    if "balls" in pa.columns and "strikes" in pa.columns:
        pa["count"] = (
            pa["balls"].fillna(0).astype(int).astype(str)
            + "-"
            + pa["strikes"].fillna(0).astype(int).astype(str)
        )
    else:
        pa["count"] = "—"

    # ── Batted-ball type shorthand ─────────────────────────────────────────
    if "bb_type" in pa.columns:
        pa["bb_type_short"] = pa["bb_type"].map(
            lambda x: BB_TYPE_SHORT.get(str(x), "—") if pd.notna(x) else "—"
        )
    else:
        pa["bb_type_short"] = "—"

    # ── Sort most recent first ─────────────────────────────────────────────
    if "game_date" in pa.columns:
        pa = pa.sort_values("game_date", ascending=False)

    keep = [
        "game_date", "opponent", "batter_team", "pitcher_name",
        "p_throws", "result_label", "result_color",
        "launch_speed", "launch_angle", "hit_distance_sc", "bb_type_short",
        "inning", "count", "outs_when_up",
        "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
        "stand", "game_pk", "events",
    ]
    return pa[[c for c in keep if c in pa.columns]].reset_index(drop=True)


def get_season_stats(ab_log: pd.DataFrame) -> dict:
    """
    Compute current-season aggregated stats from a plate-appearance log.

    Returns keys: PA, AB, H, HR, 2B, 3B, BB, K, HBP,
                  AVG, OBP, SLG, OPS, BABIP,
                  K_pct, BB_pct, avg_ev, avg_la, hard_hit_pct
    """
    if ab_log.empty:
        return {}

    ev = ab_log.get("events", pd.Series(dtype=str)).astype(str)

    pa  = len(ab_log)
    h   = int(ev.isin(HIT_EVENTS).sum())
    hr  = int((ev == "home_run").sum())
    d   = int((ev == "double").sum())
    t   = int((ev == "triple").sum())
    bb  = int(ev.isin(["walk", "intent_walk"]).sum())
    hbp = int((ev == "hit_by_pitch").sum())
    sf  = int((ev == "sac_fly").sum())
    k   = int(ev.isin(["strikeout", "strikeout_double_play"]).sum())
    ab  = int(ev.isin(AB_EVENTS).sum())

    avg  = round(h / ab, 3)          if ab > 0 else 0.0
    obp_d = ab + bb + hbp + sf
    obp  = round((h + bb + hbp) / obp_d, 3) if obp_d > 0 else 0.0
    tb   = (h - d - t - hr) + 2 * d + 3 * t + 4 * hr
    slg  = round(tb / ab, 3)         if ab > 0 else 0.0
    ops  = round(obp + slg, 3)
    bab_d = ab - k - hr + sf
    babip = round((h - hr) / bab_d, 3) if bab_d > 0 else 0.0

    ev_vals = pd.to_numeric(
        ab_log.get("launch_speed", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    la_vals = pd.to_numeric(
        ab_log.get("launch_angle", pd.Series(dtype=float)), errors="coerce"
    ).dropna()

    hard_hit = round(float((ev_vals >= 95).sum() / len(ev_vals) * 100), 1) if len(ev_vals) else 0.0
    avg_ev   = round(float(ev_vals.mean()), 1) if len(ev_vals) else None
    avg_la   = round(float(la_vals.mean()), 1) if len(la_vals) else None

    return {
        "PA": pa, "AB": ab, "H": h,
        "HR": hr, "2B": d, "3B": t,
        "BB": bb, "K": k, "HBP": hbp,
        "AVG": avg, "OBP": obp, "SLG": slg, "OPS": ops, "BABIP": babip,
        "K_pct":  round(k  / pa * 100, 1) if pa > 0 else 0.0,
        "BB_pct": round(bb / pa * 100, 1) if pa > 0 else 0.0,
        "avg_ev":       avg_ev,
        "avg_la":       avg_la,
        "hard_hit_pct": hard_hit,
    }


# ---------------------------------------------------------------------------
# Batter vs Pitcher head-to-head (H2H)
# ---------------------------------------------------------------------------

def compute_h2h(ab_log: pd.DataFrame, pitcher_name: str) -> dict:
    """Filter ab_log by pitcher last name and compute career H2H stats."""
    if ab_log.empty or "pitcher_name" not in ab_log.columns or not pitcher_name:
        return {}

    last_name = pitcher_name.strip().split()[-1]
    mask = ab_log["pitcher_name"].str.contains(last_name, case=False, na=False)
    vs = ab_log[mask]
    if vs.empty:
        return {}

    pa  = len(vs)
    ab  = int(vs["events"].isin(AB_EVENTS).sum())
    h   = int(vs["events"].isin(HIT_EVENTS).sum())
    hr  = int((vs["events"] == "home_run").sum())
    k   = int(vs["events"].isin({"strikeout", "strikeout_double_play"}).sum())
    bb  = int(vs["events"].isin({"walk", "intent_walk"}).sum())
    avg = f".{round(h / ab * 1000):03d}" if ab > 0 else ".000"

    return {"pa": pa, "ab": ab, "hits": h, "hr": hr, "k": k, "bb": bb, "avg": avg}


def format_h2h_line(pitcher_name: str, h2h: dict) -> str:
    """Return a compact H2H summary, e.g. 'vs Jacob deGrom: 0/8, 4 K, 0 HR (career)'"""
    if not h2h or h2h.get("pa", 0) == 0:
        return f"No career data vs {pitcher_name}"
    ab, h, hr, k, bb = h2h["ab"], h2h["hits"], h2h["hr"], h2h["k"], h2h["bb"]
    parts = [f"{h}/{ab}"]
    if k:
        parts.append(f"{k} K")
    if bb:
        parts.append(f"{bb} BB")
    parts.append(f"{hr} HR")
    return f"vs {pitcher_name}: {', '.join(parts)} (career)"
