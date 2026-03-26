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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
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

from utils.feature_engineer import MODEL_FEATURES, build_3yr_weighted_fg, build_3yr_weighted_pitcher


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
    df = fg_batting[fg_batting["PA"] >= 50].copy()
    # Target: power hitter = ≥10 HRs in a season (balanced ~30% positive rate)
    df["hr_hit"]  = (df["HR"] >= 10).astype(int)
    df["hr_rate"] = df["HR"] / df["PA"].replace(0, np.nan)

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
        "Barrel%_w3yr": "barrel_pct_w3yr",
        "EV_w3yr":      "exit_velo_w3yr",
        "LA_w3yr":      "launch_angle_w3yr",
        "HardHit%_w3yr":"hard_hit_pct_w3yr",
        "xwOBA_w3yr":   "xwoba_w3yr",
        "xSLG_w3yr":    "xslg_w3yr",
        "ISO":          "iso",
        "Pull%":        "pull_pct",
        "FB%":          "fb_pct",
        "HR/FB":        "hr_fb_rate",
        "wRC+":         "wrc_plus",
    }
    df = df.rename(columns=rename)

    # Ensure all model features exist
    for col in MODEL_FEATURES:
        if col not in df.columns:
            logger.warning("Adding missing column: %s", col)
            df[col] = np.nan

    X = df[MODEL_FEATURES].copy()
    y = df["hr_hit"]
    return X, y


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(X: pd.DataFrame, y: pd.Series):
    logger.info("Training: %d samples, %d features, HR rate=%.3f",
                len(X), len(X.columns), y.mean())

    X_filled = X.fillna(X.median(numeric_only=True))
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    # SMOTE for class imbalance
    smote = SMOTE(random_state=42, k_neighbors=min(3, int(y.sum()) - 1))
    X_res, y_res = smote.fit_resample(X_scaled, y)
    logger.info("After SMOTE: %d samples", len(X_res))

    params = dict(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        eval_metric="auc", random_state=42, n_jobs=-1,
    )

    # CV on the ORIGINAL (pre-SMOTE) data to get honest AUC
    tscv       = TimeSeriesSplit(n_splits=5)
    auc_scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        if len(y_val.unique()) < 2:
            logger.info("Fold %d skipped (single class in val)", fold + 1)
            continue
        # SMOTE only on training fold
        sm   = SMOTE(random_state=42, k_neighbors=min(3, int(y_tr.sum()) - 1))
        Xr, yr = sm.fit_resample(X_tr, y_tr)
        m = xgb.XGBClassifier(**params)
        m.fit(Xr, yr, eval_set=[(X_val, y_val)], verbose=False)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        auc_scores.append(auc)
        logger.info("Fold %d AUC: %.4f", fold + 1, auc)

    if auc_scores:
        logger.info("Mean CV AUC: %.4f ± %.4f", np.mean(auc_scores), np.std(auc_scores))

    # Final model trained on SMOTE-resampled full dataset
    base = xgb.XGBClassifier(**params)
    base.fit(X_res, y_res)

    # SHAP importance (XGBoost has built-in probability output — no calibration wrapper needed)
    explainer  = shap.TreeExplainer(base)
    sample_n   = min(500, len(X_scaled))
    sv         = explainer.shap_values(X_scaled[:sample_n])
    importance = dict(sorted(
        zip(MODEL_FEATURES, np.abs(sv).mean(axis=0).tolist()),
        key=lambda x: x[1], reverse=True,
    ))

    return base, scaler, importance, auc_scores


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
