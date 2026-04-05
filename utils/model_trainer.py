"""
Model trainer — trains XGBoost HR probability classifier on 2017–2025 data.
Uses FanGraphs batting/pitching as primary source (already contains Statcast metrics).
Training data: 2017–2025 seasons.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH     = MODEL_DIR / "xgboost_hr.pkl"
SCALER_PATH    = MODEL_DIR / "scaler.pkl"
FEAT_IMP_PATH  = MODEL_DIR / "feature_importance.json"

TRAIN_START = int(os.getenv("TRAIN_START_YEAR", 2017))
TRAIN_END   = int(os.getenv("TRAIN_END_YEAR",   2025))

from utils.feature_engineer import (
    MODEL_FEATURES,
    build_3yr_weighted_fg,
    build_3yr_weighted_pitcher,
    platoon_adj,
    _LEAGUE_HR_FB,
    _LEAGUE_FB_PCT,
    _LEAGUE_PULL_PCT,
)
from utils.game_log_builder import GAME_LEVEL_FEATURES


# ---------------------------------------------------------------------------
# Calibrated model wrapper (module-level so joblib can pickle/unpickle)
# ---------------------------------------------------------------------------

class CalibratedXGB:
    """Wraps an XGBClassifier + IsotonicRegression calibrator."""

    def __init__(self, model, calibrator):
        self.model      = model
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])


# ---------------------------------------------------------------------------
# Build training dataset
# ---------------------------------------------------------------------------

def build_training_data(
    fg_batting: pd.DataFrame,
    fg_pitching: pd.DataFrame,
    park_factors: pd.DataFrame,
    current_year: int,
) -> tuple:
    """
    Builds (X, y) for training.
    One row per batter-season.  Target: hr_hit = 1 if player hit ≥1 HR that season.
    """
    year_col = "Season" if "Season" in fg_batting.columns else "season"

    # Filter to min plate appearances
    df = fg_batting[fg_batting["PA"] >= 100].copy()
    df["hr_rate"] = df["HR"] / df["PA"].replace(0, np.nan)
    # Target: elite game-level HR threat = HR/PA >= 3.5%
    # This keeps ~15% of batters (true power hitters) and forces the model to
    # discriminate within the talent distribution rather than just flagging any slugger.
    df["hr_hit"] = (df["hr_rate"] >= 0.035).astype(int)

    # 3-year weighted Statcast features
    batter_w = build_3yr_weighted_fg(fg_batting, current_year)
    df = df.merge(batter_w, on="IDfg", how="left")

    # Pitcher aggregate — use league-average per season for batter-only model
    pit_agg = (
        fg_pitching.groupby(year_col)[["HR/9", "HR/FB", "FIP", "GB%", "Barrel%", "HardHit%"]]
        .mean().reset_index().rename(columns={year_col: "Season_pit"})
    )
    # We'll use leaguge-avg pitcher stats as a constant for training
    if not pit_agg.empty:
        latest_pit = pit_agg.iloc[-1]
    else:
        latest_pit = pd.Series()

    df["pitcher_hr9"]          = float(latest_pit.get("HR/9", np.nan))
    df["pitcher_hr_fb"]        = float(latest_pit.get("HR/FB", np.nan))
    df["pitcher_fip"]          = float(latest_pit.get("FIP", np.nan))
    df["pitcher_gb_pct"]       = float(latest_pit.get("GB%", np.nan))
    df["pitcher_barrel_pct"]   = float(latest_pit.get("Barrel%", np.nan))
    df["pitcher_hard_hit_pct"] = float(latest_pit.get("HardHit%", np.nan))

    # Park factor — map team → hr_park_factor
    team_col = "Team" if "Team" in df.columns else "team"
    pf_map = dict(zip(park_factors["team"], park_factors["hr_park_factor"]))
    df["park_hr_factor"] = df[team_col].map(pf_map).fillna(100)

    # Neutral weather for training (park factor carries environment)
    df["temp_f"]            = 72.0
    df["temp_adjustment"]   = 1.0
    df["wind_speed_mph"]    = 0.0
    df["wind_direction_enc"]= 0.0
    df["is_day_game"]       = 0

    # Rename FG columns → MODEL_FEATURES names
    rename = {
        "Barrel%_w3yr":  "barrel_pct_w3yr",
        "EV_w3yr":       "exit_velo_w3yr",
        "LA_w3yr":       "launch_angle_w3yr",
        "HardHit%_w3yr": "hard_hit_pct_w3yr",
        "xwOBA_w3yr":    "xwoba_w3yr",
        "xSLG_w3yr":     "xslg_w3yr",
        "ISO":           "iso",
        "Pull%":         "pull_pct",
        "Cent%":         "cent_pct",
        "Oppo%":         "oppo_pct",
        "FB%":           "fb_pct",
        "HR/FB":         "hr_fb_rate",
        "wRC+":          "wrc_plus",
    }
    df = df.rename(columns=rename)

    # Platoon adjustment — neutral (1.0) for batter-only season-level model
    df["platoon_adj"] = 1.0

    # xHR delta: batter's historical HR/FB minus league average.
    # Using 3yr-weighted historical rate to avoid same-season leakage with the target (hr_rate).
    # Positive = consistent power over-performer. Negative = under-performer.
    hr_fb_hist = df["HR/FB_w3yr"] if "HR/FB_w3yr" in df.columns else pd.Series(np.nan, index=df.index)
    df["xhr_delta"] = hr_fb_hist - _LEAGUE_HR_FB

    # Batter-specific park factor: pull hitters get extra boost in hitter-friendly parks
    pull_col = df["pull_pct"] if "pull_pct" in df.columns else pd.Series(_LEAGUE_PULL_PCT, index=df.index)
    fb_col2  = df["fb_pct"]   if "fb_pct"   in df.columns else pd.Series(_LEAGUE_FB_PCT,   index=df.index)
    df["batter_park_factor"] = df["park_hr_factor"] * (
        1.0
        + (pull_col - _LEAGUE_PULL_PCT) * 0.30
        + (fb_col2  - _LEAGUE_FB_PCT)   * 0.15
    )

    # Ensure all model features exist
    for col in MODEL_FEATURES:
        if col not in df.columns:
            logger.warning("Adding missing column: %s", col)
            df[col] = np.nan

    X = df[MODEL_FEATURES].copy()
    y = df["hr_hit"]
    return X, y


# ---------------------------------------------------------------------------
# Game-level training dataset (used after build_game_level_dataset() cache)
# ---------------------------------------------------------------------------

def build_training_data_game_level(
    game_logs: "pd.DataFrame",
    fg_batting: "pd.DataFrame",
    fg_pitching: "pd.DataFrame",
    park_factors: "pd.DataFrame",
) -> tuple:
    """
    Build (X, y) for game-level training.
    Each row = one batter in one game.  Target: hit_hr (0/1).

    CRITICAL FIX vs old build_training_data():
    - Each row gets the ACTUAL opposing pitcher's stats (not league average)
    - Platoon adjustment (L/R batter vs pitcher) is a real feature
    - Target is per-game HR, not season-level HR-rate classification

    Args:
        game_logs: output of build_game_level_dataset()
        fg_batting: multi-year FanGraphs batting data
        fg_pitching: multi-year FanGraphs pitching data
        park_factors: park factor table (team, hr_park_factor)
    """
    if game_logs.empty:
        raise ValueError("game_logs is empty — run build_game_level_dataset() first")

    year_col = "Season" if "Season" in fg_batting.columns else "season"
    pf_map   = dict(zip(park_factors["team"], park_factors["hr_park_factor"]))

    # Need batter MLBAM id → FG IDfg crosswalk
    try:
        from pybaseball import playerid_reverse_lookup
        # Build crosswalk from all unique batter IDs in game_logs
        mlb_ids = game_logs["batter"].dropna().unique().tolist()
        crosswalk = playerid_reverse_lookup(mlb_ids, key_type="mlbam")
        crosswalk = crosswalk[["key_mlbam", "key_fangraphs"]].drop_duplicates()
        crosswalk.columns = ["batter", "IDfg"]
        game_logs = game_logs.merge(crosswalk, on="batter", how="left")
        logger.info("Crosswalk: %d / %d batters resolved to FanGraphs IDfg",
                    game_logs["IDfg"].notna().sum(), len(game_logs))
    except Exception as exc:
        logger.warning("playerid_reverse_lookup failed: %s — skipping IDfg join", exc)

    all_years = sorted(game_logs["season"].unique()) if "season" in game_logs.columns else []
    result_rows = []

    for yr in all_years:
        yr_logs = game_logs[game_logs["season"] == yr].copy()
        if yr_logs.empty:
            continue

        # FanGraphs batting (current year + weighted features)
        batter_w = build_3yr_weighted_fg(fg_batting, yr)
        fg_bat_yr = fg_batting[fg_batting[year_col] == yr].copy()
        fg_bat_yr = fg_bat_yr.rename(columns={
            "Barrel%_w3yr": "barrel_pct_w3yr", "EV_w3yr": "exit_velo_w3yr",
            "LA_w3yr": "launch_angle_w3yr", "HardHit%_w3yr": "hard_hit_pct_w3yr",
            "xwOBA_w3yr": "xwoba_w3yr", "xSLG_w3yr": "xslg_w3yr",
            "ISO": "iso", "Pull%": "pull_pct", "FB%": "fb_pct",
            "HR/FB": "hr_fb_rate", "wRC+": "wrc_plus",
        })

        # FanGraphs pitching (current year)
        fg_pit_yr = fg_pitching[fg_pitching[year_col] == yr].copy()
        fg_pit_yr = fg_pit_yr.rename(columns={
            "HR/9": "pitcher_hr9", "HR/FB": "pitcher_hr_fb",
            "FIP": "pitcher_fip", "xFIP": "pitcher_xfip",
            "GB%": "pitcher_gb_pct", "K%": "pitcher_k_pct",
            "Barrel%": "pitcher_barrel_pct", "HardHit%": "pitcher_hard_hit_pct",
        })

        # Merge batter FG features into game logs for this year
        if "IDfg" in yr_logs.columns and "IDfg" in fg_bat_yr.columns:
            yr_logs = yr_logs.merge(fg_bat_yr, on="IDfg", how="left")
            yr_logs = yr_logs.merge(batter_w, on="IDfg", how="left", suffixes=("", "_w"))

        # Merge ACTUAL pitcher stats per row
        pit_needed = ["IDfg", "pitcher_hr9", "pitcher_hr_fb", "pitcher_fip",
                      "pitcher_xfip", "pitcher_gb_pct", "pitcher_k_pct",
                      "pitcher_barrel_pct", "pitcher_hard_hit_pct"]
        pit_available = [c for c in pit_needed if c in fg_pit_yr.columns]
        if "opposing_pitcher_id" in yr_logs.columns and len(pit_available) > 1:
            fg_pit_yr_sub = fg_pit_yr[pit_available].copy()
            fg_pit_yr_sub = fg_pit_yr_sub.rename(columns={"IDfg": "opposing_pitcher_id"})
            yr_logs = yr_logs.merge(fg_pit_yr_sub, on="opposing_pitcher_id", how="left")

        # Platoon adjustment
        if "stand" in yr_logs.columns:
            yr_logs["platoon_adj"] = yr_logs["stand"].apply(
                lambda s: platoon_adj(s, "R")  # pitcher hand unknown here; default R (majority)
            )
        else:
            yr_logs["platoon_adj"] = 1.0

        # Park factor (home team approximated by FG team column)
        team_col = "Team" if "Team" in yr_logs.columns else None
        if team_col and team_col in yr_logs.columns:
            yr_logs["park_hr_factor"] = yr_logs[team_col].map(pf_map).fillna(100)
        else:
            yr_logs["park_hr_factor"] = 100.0

        # Neutral weather for training (park factor carries environment)
        yr_logs["temp_f"]             = 72.0
        yr_logs["temp_adjustment"]    = 1.0
        yr_logs["wind_speed_mph"]     = 0.0
        yr_logs["wind_direction_enc"] = 0.0
        yr_logs["is_day_game"]        = 0

        # Sweet spot pct: from game-level aggregation
        if "sweet_spot_pct" not in yr_logs.columns:
            yr_logs["sweet_spot_pct"] = np.nan
        if "max_ev_95th" not in yr_logs.columns:
            yr_logs["max_ev_95th"] = np.nan  # filled below from avg_ev proxy

        result_rows.append(yr_logs)

    if not result_rows:
        raise ValueError("No rows assembled for game-level training")

    df = pd.concat(result_rows, ignore_index=True)

    # Ensure all GAME_LEVEL_FEATURES exist
    for col in GAME_LEVEL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    df = df[df["hit_hr"].notna()].copy() if "hit_hr" in df.columns else df.copy()

    X = df[GAME_LEVEL_FEATURES].copy()
    y = df["hit_hr"].astype(int) if "hit_hr" in df.columns else pd.Series(np.zeros(len(df)), dtype=int)

    logger.info("Game-level training set: %d rows, HR rate=%.4f  (%d features)",
                len(X), y.mean(), len(GAME_LEVEL_FEATURES))
    return X, y


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(X: pd.DataFrame, y: pd.Series):
    logger.info("Training: %d samples, %d features, HR-power rate=%.3f",
                len(X), len(X.columns), y.mean())

    X_filled = X.fillna(X.median(numeric_only=True))
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    # Class imbalance: use scale_pos_weight so XGBoost sees the true class distribution.
    # SMOTE was removed because it inflates probabilities by training on a 50/50 split
    # while inference sees the real ~15% positive rate — the gap causes overconfident output.
    pos_count = int(y.sum())
    neg_count = int((y == 0).sum())
    spw = neg_count / max(pos_count, 1)
    logger.info("scale_pos_weight = %.2f  (neg=%d / pos=%d)", spw, neg_count, pos_count)

    params = dict(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=8,        # higher → less overfit on rare positive cases
        gamma=0.3,
        reg_alpha=0.5,
        reg_lambda=2.5,
        scale_pos_weight=spw,      # compensates class imbalance without SMOTE distortion
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    # CV on original data for honest AUC + Brier score
    tscv       = TimeSeriesSplit(n_splits=5)
    auc_scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        if len(y_val.unique()) < 2:
            logger.info("Fold %d skipped (single class in val)", fold + 1)
            continue
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        probs = m.predict_proba(X_val)[:, 1]
        auc   = roc_auc_score(y_val, probs)
        brier = brier_score_loss(y_val, probs)
        auc_scores.append(auc)
        logger.info("Fold %d AUC: %.4f  Brier: %.4f", fold + 1, auc, brier)

    if auc_scores:
        logger.info("Mean CV AUC: %.4f ± %.4f", np.mean(auc_scores), np.std(auc_scores))

    # Final base model: train on first 80% (chronological), calibrate on last 20%
    cal_split = int(len(X_scaled) * 0.80)
    base = xgb.XGBClassifier(**params)
    base.fit(X_scaled[:cal_split], y.iloc[:cal_split])

    # Isotonic regression calibration on held-out 20% to map raw scores → true probs
    raw_cal  = base.predict_proba(X_scaled[cal_split:])[:, 1]
    y_cal    = y.iloc[cal_split:]
    ir       = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_cal, y_cal)

    # Brier score on calibration set (lower is better)
    cal_probs = ir.predict(raw_cal)
    brier_cal = brier_score_loss(y_cal, cal_probs)
    logger.info("Calibration Brier score (held-out 20%%): %.4f", brier_cal)

    # Refit base on full data so predictions use all available history
    base.fit(X_scaled, y)
    logger.info("Isotonic calibration applied (held-out 20%% split)")

    calibrated = CalibratedXGB(base, ir)

    # SHAP importance — use base model (TreeExplainer requires native XGBoost)
    explainer  = shap.TreeExplainer(base)
    sample_n   = min(500, len(X_scaled))
    sv         = explainer.shap_values(X_scaled[:sample_n])
    importance = dict(sorted(
        zip(MODEL_FEATURES, np.abs(sv).mean(axis=0).tolist()),
        key=lambda x: x[1], reverse=True,
    ))

    return calibrated, scaler, importance, auc_scores


# ---------------------------------------------------------------------------
# Persist / load
# ---------------------------------------------------------------------------

def save_model(model, scaler, feature_importance: dict):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEAT_IMP_PATH, "w") as f:
        json.dump(feature_importance, f, indent=2)
    logger.info("Saved model → %s", MODEL_PATH)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run model_trainer.py first.")
    # Compatibility shim: models saved while running model_trainer.py as __main__
    # pickle CalibratedXGB under the key '__main__.CalibratedXGB'.
    # Register it under that alias so joblib can resolve it regardless of call site.
    import sys
    if "__main__" not in sys.modules:
        import types
        sys.modules["__main__"] = types.ModuleType("__main__")
    if not hasattr(sys.modules["__main__"], "CalibratedXGB"):
        sys.modules["__main__"].CalibratedXGB = CalibratedXGB  # type: ignore[attr-defined]
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def load_feature_importance() -> dict:
    if not FEAT_IMP_PATH.exists():
        return {}
    with open(FEAT_IMP_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=TRAIN_START)
    parser.add_argument("--end",   type=int, default=TRAIN_END)
    args = parser.parse_args()

    from utils.data_fetcher import fetch_all_years, load_park_factors

    logger.info("Loading data %d–%d ...", args.start, args.end)
    data        = fetch_all_years(args.start, args.end)
    park_factors = load_park_factors()

    logger.info("Building training dataset ...")
    X, y = build_training_data(
        data["fg_batting"],
        data["fg_pitching"],
        park_factors,
        current_year=args.end,
    )
    logger.info("X shape: %s  |  HR rate: %.4f", X.shape, y.mean())

    logger.info("Training model ...")
    model, scaler, importance, auc_scores = train(X, y)
    save_model(model, scaler, importance)

    logger.info("\nTop 10 features by SHAP importance:")
    for feat, val in list(importance.items())[:10]:
        logger.info("  %-35s %.4f", feat, val)
