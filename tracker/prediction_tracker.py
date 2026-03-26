"""
Prediction tracker — writes daily predictions and results to MLB_HR_Predictions.xlsx.
Sheets: Predictions, Model_Performance, Feature_Importance.
"""

import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

logger = logging.getLogger(__name__)

EXCEL_PATH = Path("MLB_HR_Predictions.xlsx")

PRED_COLUMNS = [
    "Date", "Player", "Team", "Opponent", "Pitcher",
    "HR_Probability", "Confidence", "Park_Factor",
    "Temp_F", "Wind_Speed_MPH", "Wind_Direction", "Is_Indoor",
    "Home_Game", "Predicted_HRs", "Actual_HRs", "Hit",
]

PERF_COLUMNS = [
    "Date", "Total_Predictions", "High_Conf_Count",
    "Daily_Accuracy", "Cumulative_Accuracy",
    "High_Conf_Hit_Rate", "ROI_Flat_Bet",
]

HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
HIGH_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
MED_FILL = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
LOW_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
HIT_FILL = PatternFill(start_color="00B050", end_color="00B050", fill_type="solid")
MISS_FILL = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")


# ---------------------------------------------------------------------------
# Workbook initialisation
# ---------------------------------------------------------------------------

def _init_workbook() -> Workbook:
    """Create a new workbook with the three required sheets."""
    wb = Workbook()
    # Default sheet → Predictions
    ws_pred = wb.active
    ws_pred.title = "Predictions"
    _write_header(ws_pred, PRED_COLUMNS)

    ws_perf = wb.create_sheet("Model_Performance")
    _write_header(ws_perf, PERF_COLUMNS)

    ws_feat = wb.create_sheet("Feature_Importance")
    _write_header(ws_feat, ["Date", "Feature", "Importance_Score"])

    wb.save(EXCEL_PATH)
    logger.info("Created %s", EXCEL_PATH)
    return wb


def _write_header(ws, columns: list):
    ws.append(columns)
    for col_idx, _ in enumerate(columns, 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = HEADER_FILL
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center")
    ws.freeze_panes = "A2"


def _get_workbook() -> Workbook:
    if EXCEL_PATH.exists():
        return load_workbook(EXCEL_PATH)
    return _init_workbook()


# ---------------------------------------------------------------------------
# Save predictions
# ---------------------------------------------------------------------------

def save_predictions(predictions_df: pd.DataFrame):
    """
    Append today's predictions to the Predictions sheet.
    Skips rows that already exist for the same Date+Player+Pitcher combo.
    Applies conditional formatting by confidence tier.
    """
    wb = _get_workbook()
    ws = wb["Predictions"]

    # Build set of already-saved keys to prevent duplicates
    existing_keys: set = set()
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] and row[1] and row[4]:  # Date, Player, Pitcher
            existing_keys.add((str(row[0]), str(row[1]).lower(), str(row[4]).lower()))

    new_count = 0
    for _, row in predictions_df.iterrows():
        key = (
            str(row.get("date", date.today().isoformat())),
            str(row.get("player", "")).lower(),
            str(row.get("pitcher", "")).lower(),
        )
        if key in existing_keys:
            continue
        existing_keys.add(key)

        new_row = [
            row.get("date", date.today().isoformat()),
            row.get("player", ""),
            row.get("team", ""),
            row.get("opponent", ""),
            row.get("pitcher", ""),
            round(float(row.get("hr_probability", 0)), 4),
            row.get("confidence", "Low"),
            row.get("park_factor", 100),
            row.get("temp_f", 72),
            row.get("wind_speed_mph", 0),
            row.get("wind_direction", "calm"),
            bool(row.get("is_indoor", False)),
            bool(row.get("home_game", False)),
            round(float(row.get("hr_probability", 0)) * 0.6, 3),  # simple regression proxy
            "",    # Actual_HRs — filled after game
            "",    # Hit — filled after game
        ]
        ws.append(new_row)

        # Color-code by confidence
        row_idx = ws.max_row
        fill = (HIGH_FILL if row.get("confidence") == "High"
                else MED_FILL if row.get("confidence") == "Medium"
                else LOW_FILL)
        for col in range(1, len(PRED_COLUMNS) + 1):
            ws.cell(row=row_idx, column=col).fill = fill

    _auto_width(ws)
    wb.save(EXCEL_PATH)
    logger.info("Saved %d predictions → %s", len(predictions_df), EXCEL_PATH)


# ---------------------------------------------------------------------------
# Update actual results
# ---------------------------------------------------------------------------

def update_results(results: list):
    """
    Update actual HR results after games complete.

    Args:
        results: list of dicts {player, date, actual_hrs}
    """
    wb = _get_workbook()
    ws = wb["Predictions"]

    result_map = {
        (r["player"].lower(), r["date"]): r["actual_hrs"]
        for r in results
    }

    for row in ws.iter_rows(min_row=2):
        player = str(row[1].value or "").lower()
        game_date = str(row[0].value or "")
        key = (player, game_date)

        if key in result_map:
            actual = result_map[key]
            hr_prob = float(row[5].value or 0)
            confidence = str(row[6].value or "Low")
            threshold = 0.12 if confidence != "High" else 0.18

            row[14].value = actual  # Actual_HRs
            hit = int(actual) > 0
            row[15].value = int(hit)

            # Color hit/miss for high confidence
            if confidence in ("High", "Medium"):
                fill = HIT_FILL if hit else MISS_FILL
                for cell in row:
                    cell.fill = fill

    wb.save(EXCEL_PATH)
    logger.info("Updated results for %d players", len(results))

    # Recalculate daily performance
    _recalculate_performance(wb)
    wb.save(EXCEL_PATH)


# ---------------------------------------------------------------------------
# Performance recalculation
# ---------------------------------------------------------------------------

def _recalculate_performance(wb: Workbook):
    """Recompute Model_Performance sheet from Predictions sheet."""
    ws_pred = wb["Predictions"]
    ws_perf = wb["Model_Performance"]

    # Clear existing data (keep header)
    for row in ws_perf.iter_rows(min_row=2):
        for cell in row:
            cell.value = None

    # Read predictions into DataFrame
    data = list(ws_pred.iter_rows(min_row=2, values_only=True))
    if not data:
        return

    df = pd.DataFrame(data, columns=PRED_COLUMNS)
    df = df[df["Actual_HRs"].notna()].copy()
    df["Actual_HRs"] = pd.to_numeric(df["Actual_HRs"], errors="coerce").fillna(0)
    df["Hit"] = pd.to_numeric(df["Hit"], errors="coerce").fillna(0)

    if df.empty:
        return

    cumulative_hits = 0
    cumulative_total = 0

    for game_date, day_df in df.groupby("Date"):
        total = len(day_df)
        hits = day_df["Hit"].sum()
        high_df = day_df[day_df["Confidence"] == "High"]
        high_count = len(high_df)
        high_hits = high_df["Hit"].sum()

        cumulative_hits += hits
        cumulative_total += total

        daily_acc = hits / total if total > 0 else 0
        cum_acc = cumulative_hits / cumulative_total if cumulative_total > 0 else 0
        high_hit_rate = high_hits / high_count if high_count > 0 else 0
        roi = (high_hits * 0.9 - (high_count - high_hits)) / high_count if high_count > 0 else 0

        ws_perf.append([
            game_date, total, high_count,
            round(daily_acc, 4), round(cum_acc, 4),
            round(high_hit_rate, 4), round(roi, 4),
        ])

    logger.info("Recalculated performance metrics")


# ---------------------------------------------------------------------------
# Feature importance update
# ---------------------------------------------------------------------------

def update_feature_importance(importance_dict: dict):
    """Append today's feature importance to the Feature_Importance sheet."""
    wb = _get_workbook()
    ws = wb["Feature_Importance"]
    today = date.today().isoformat()
    for feature, score in importance_dict.items():
        ws.append([today, feature, round(float(score), 6)])
    wb.save(EXCEL_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_width(ws):
    for col in ws.columns:
        max_len = max((len(str(cell.value or "")) for cell in col), default=10)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 2, 40)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Demo: initialize workbook
    _init_workbook()
    print(f"Initialized {EXCEL_PATH}")
