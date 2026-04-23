"""
Hit predictor — scores players on likelihood of recording a hit (any type),
broken down by single, double, and triple probability.

Uses FanGraphs season batting stats already cached by utils/data_fetcher.py.
No separate ML model: weighted scoring from AVG, xBA, ISO, sprint speed, etc.
"""

import logging
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

MIN_AB = 20  # filter out players with too few at-bats


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_hit_scores(fg: pd.DataFrame) -> pd.DataFrame:
    """
    Given a FanGraphs batting DataFrame, return one row per player with:
      Player, Team, hit_score, single_score, double_score, triple_score,
      hot_streak, hit_type_label, confidence, H_per_AB, 2B_per_AB, 3B_per_AB
    """
    df = fg.copy()

    # ── Numeric coercion ───────────────────────────────────────────────────────
    for col in ["H", "AB", "2B", "3B", "HR", "ISO", "wRC+"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[df["AB"] >= MIN_AB].copy()
    if df.empty:
        return pd.DataFrame()

    # ── Per-AB rates ───────────────────────────────────────────────────────────
    ab = df["AB"].clip(lower=1)
    h  = df["H"].clip(lower=0)
    df["1B"] = (df["H"] - df["2B"] - df["3B"] - df["HR"]).clip(lower=0)
    df["H_per_AB"]  = h  / ab
    df["2B_per_AB"] = df["2B"] / ab
    df["3B_per_AB"] = df["3B"] / ab

    # ── Expected BA (Statcast xBA) ─────────────────────────────────────────────
    if "xBA" in df.columns:
        xba = pd.to_numeric(df["xBA"], errors="coerce").fillna(df["H_per_AB"])
    else:
        xba = df["H_per_AB"]
    df["xba_num"] = xba.clip(lower=0)

    # ── Hot streak: actual AVG meaningfully exceeds xBA ───────────────────────
    df["hot_streak"] = df["H_per_AB"] > (df["xba_num"] * 1.15)

    # ── wRC+ normalized (200 = elite) ─────────────────────────────────────────
    wrc_norm = df["wRC+"].clip(lower=0) / 200.0

    # ── Overall hit score ─────────────────────────────────────────────────────
    # 50% xBA (true talent), 40% actual H/AB, 10% wRC+ normalized
    df["hit_score"] = (
        0.50 * df["xba_num"] +
        0.40 * df["H_per_AB"] +
        0.10 * wrc_norm
    ).clip(lower=0)

    # ── Single score ──────────────────────────────────────────────────────────
    single_frac = df["1B"] / h.clip(lower=1)
    df["single_score"] = (df["hit_score"] * single_frac).clip(lower=0)

    # ── Double score (boosted by ISO for extra-base power) ───────────────────
    iso_norm = df["ISO"].clip(lower=0) / 0.300
    double_frac = df["2B"] / h.clip(lower=1)
    df["double_score"] = (df["hit_score"] * double_frac * (1 + 0.20 * iso_norm)).clip(lower=0)

    # ── Triple score (boosted by sprint speed) ───────────────────────────────
    triple_frac = df["3B"] / h.clip(lower=1)
    if "Sprint Speed" in df.columns:
        sprint = pd.to_numeric(df["Sprint Speed"], errors="coerce").fillna(27.0)
    else:
        sprint = pd.Series(27.0, index=df.index)
    sprint_factor = ((sprint - 27.0) / 3.0).clip(lower=0)
    df["triple_score"] = (df["hit_score"] * triple_frac * (1 + 0.30 * sprint_factor)).clip(lower=0)

    # ── Confidence label ──────────────────────────────────────────────────────
    def _conf(score: float) -> str:
        if score >= 0.32:
            return "High"
        if score >= 0.22:
            return "Medium"
        return "Low"

    df["confidence"] = df["hit_score"].apply(_conf)

    # ── Best hit type label ───────────────────────────────────────────────────
    def _best_type(row) -> str:
        scores = {
            "Single": row["single_score"],
            "Double": row["double_score"],
            "Triple": row["triple_score"],
        }
        return max(scores, key=scores.get)

    df["hit_type_label"] = df.apply(_best_type, axis=1)

    # ── Rename / select output columns ────────────────────────────────────────
    name_col = "Name" if "Name" in df.columns else "PlayerName"
    if name_col not in df.columns:
        # some FG exports use first/last separately
        if "FirstName" in df.columns and "LastName" in df.columns:
            df["Name"] = df["FirstName"] + " " + df["LastName"]
        else:
            df["Name"] = df.index.astype(str)

    keep = [
        name_col, "Team",
        "hit_score", "single_score", "double_score", "triple_score",
        "hot_streak", "hit_type_label", "confidence",
        "H_per_AB", "2B_per_AB", "3B_per_AB", "AB", "wRC+",
    ]
    available = [c for c in keep if c in df.columns]
    out = df[available].rename(columns={name_col: "Player"})
    return out.sort_values("hit_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_hit_predictions(year: int = 2026) -> pd.DataFrame:
    """Fetch (or load from cache) FanGraphs batting data and score all players.

    Falls back to the prior season when the requested year's data is unavailable
    (e.g. early-season 403s from FanGraphs before current-year stats are published).
    """
    from utils.data_fetcher import fetch_fangraphs_batting

    fg = fetch_fangraphs_batting(year)
    if fg is None or fg.empty or "Name" not in fg.columns:
        logger.warning(
            "FanGraphs batting data unavailable for %d — falling back to %d", year, year - 1
        )
        fg = fetch_fangraphs_batting(year - 1)

    if fg is None or fg.empty or "Name" not in fg.columns:
        logger.warning("FanGraphs batting data unavailable for %d and %d", year, year - 1)
        return pd.DataFrame()

    return compute_hit_scores(fg)


# ---------------------------------------------------------------------------
# Parlay builder
# ---------------------------------------------------------------------------

def build_hit_parlays(df: pd.DataFrame) -> dict:
    """
    Build 15 three-leg parlays across three categories.

    Returns dict:
      "two_base"  → list of 5 combos (list of 3 rows), leg type = "2+ Total Bases"
      "any_hit"   → list of 5 combos, leg type = "Any Hit"
      "mixed"     → list of 5 dicts {"legs": [...], "leg_type": [...]}:
                      [0-1] 3 singles each, [2-3] 3 doubles each, [4] 2 singles + 1 double

    Diversity rule: within each group, no player appears in more than 2 parlays.
    Groups are independent — a star player may appear in multiple groups.
    """
    if df.empty or len(df) < 3:
        return {"two_base": [], "any_hit": [], "mixed": []}

    MAX_PER_PLAYER = 2

    def _build_group(sorted_rows: list, n_parlays: int) -> list:
        """Greedily build n_parlays of 3-leg combos with no player > MAX_PER_PLAYER."""
        group_count: dict[str, int] = defaultdict(int)
        parlays: list = []
        for start in range(len(sorted_rows)):
            if len(parlays) >= n_parlays:
                break
            picked = []
            for row in sorted_rows[start:]:
                name = str(row.get("Player", ""))
                if group_count[name] < MAX_PER_PLAYER:
                    picked.append(row)
                if len(picked) == 3:
                    break
            if len(picked) == 3:
                parlays.append(picked)
                for row in picked:
                    group_count[str(row.get("Player", ""))] += 1
        return parlays

    # ── Group A: 2+ Total Bases (double or better) ─────────────────────────
    two_base_rows = [r for _, r in df.sort_values("double_score", ascending=False).head(40).iterrows()]
    two_base = _build_group(two_base_rows, 5)

    # ── Group B: Any Hit ───────────────────────────────────────────────────
    any_hit_rows = [r for _, r in df.sort_values("hit_score", ascending=False).head(40).iterrows()]
    any_hit = _build_group(any_hit_rows, 5)

    # ── Group C: Mixed Singles / Doubles ───────────────────────────────────
    # Each parlay is unique — no player appears in more than 1 parlay in this group.
    single_rows = [r for _, r in df.sort_values("single_score", ascending=False).head(40).iterrows()]
    double_rows = [r for _, r in df.sort_values("double_score", ascending=False).head(40).iterrows()]

    # Build two separate "single" parlays and two "double" parlays using disjoint players.
    single_parlays = _build_group(single_rows, 2)
    double_parlays = _build_group(double_rows, 2)

    # Parlay 5: pick 2 singles + 1 double, avoiding players already used above
    used_in_mixed = {
        str(r.get("Player", ""))
        for legs in single_parlays + double_parlays
        for r in legs
    }
    remaining_singles = [r for r in single_rows if str(r.get("Player", "")) not in used_in_mixed]
    remaining_doubles = [r for r in double_rows if str(r.get("Player", "")) not in used_in_mixed]
    s_legs = remaining_singles[:2]
    d_legs = remaining_doubles[:1]

    mixed: list = []
    for legs in single_parlays:
        mixed.append({"legs": legs, "leg_type": ["Single", "Single", "Single"]})
    for legs in double_parlays:
        mixed.append({"legs": legs, "leg_type": ["Double", "Double", "Double"]})
    if len(s_legs) == 2 and len(d_legs) == 1:
        mixed.append({"legs": s_legs + d_legs, "leg_type": ["Single", "Single", "Double"]})

    return {"two_base": two_base, "any_hit": any_hit, "mixed": mixed}
