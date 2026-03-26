"""
MLB HR Predictor — Automated Scheduler
Runs two jobs daily:

  09:00 AM ET  →  Fetch today's predictions and save to Excel
  11:30 PM ET  →  Fetch last night's results and update Excel

Run with:
    python3 scheduler.py

Keeps running in the foreground. Use Ctrl+C to stop.
For background: nohup python3 scheduler.py > logs/scheduler.log 2>&1 &
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import schedule
import time
from datetime import date, timedelta, datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/scheduler.log"),
    ],
)
logger = logging.getLogger("scheduler")


# ── Jobs ───────────────────────────────────────────────────────────────────────

def run_predictions():
    """Morning job: generate today's HR predictions and save to Excel."""
    today = date.today().strftime("%Y-%m-%d")
    logger.info("=" * 60)
    logger.info("MORNING JOB — Predictions for %s", today)
    logger.info("=" * 60)
    try:
        from utils.predictor import predict_today
        from tracker.prediction_tracker import save_predictions
        from utils.roster_fetcher import refresh_rosters

        # Refresh rosters every morning so team assignments are current
        refresh_rosters(int(today[:4]))

        preds = predict_today(today)
        if preds.empty:
            logger.warning("No predictions generated for %s", today)
            return

        save_predictions(preds)

        high = (preds["confidence"] == "High").sum()
        med  = (preds["confidence"] == "Medium").sum()
        logger.info(
            "Saved %d predictions (%d High, %d Medium) → MLB_HR_Predictions.xlsx",
            len(preds), high, med,
        )

        # Print top 10 high confidence to log
        top = preds[preds["confidence"] == "High"].head(10)
        if not top.empty:
            logger.info("Top High-confidence picks:")
            for _, row in top.iterrows():
                logger.info(
                    "  %-25s %s vs %-20s  prob=%.1f%%  park=%d  wind=%s",
                    row["player"], row["team"], row["pitcher"],
                    row["hr_probability"] * 100,
                    row["park_factor"],
                    row["wind_direction"],
                )
    except Exception as exc:
        logger.error("Predictions job FAILED: %s", exc, exc_info=True)


def run_results_update():
    """Night job: fetch yesterday's actual results and update Excel tracker."""
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info("=" * 60)
    logger.info("NIGHT JOB — Results update for %s", yesterday)
    logger.info("=" * 60)
    try:
        from utils.results_fetcher import fetch_results_for_date
        from tracker.prediction_tracker import update_results, update_feature_importance
        from utils.model_trainer import load_feature_importance

        results = fetch_results_for_date(yesterday)
        if not results:
            logger.warning("No results found for %s", yesterday)
            return

        update_results(results)

        hr_hitters = [r for r in results if r["actual_hrs"] > 0]
        logger.info(
            "Updated %d results — %d HR hitters yesterday",
            len(results), len(hr_hitters),
        )
        for r in sorted(hr_hitters, key=lambda x: -x["actual_hrs"])[:10]:
            logger.info("  %-25s %d HR (%s)", r["player"], r["actual_hrs"], r["team"])

        # Also refresh feature importance in Excel
        try:
            importance = load_feature_importance()
            if importance:
                update_feature_importance(importance)
        except Exception:
            pass

    except Exception as exc:
        logger.error("Results update job FAILED: %s", exc, exc_info=True)


# ── Schedule ───────────────────────────────────────────────────────────────────

def setup_schedule():
    PREDICT_TIME = os.getenv("PREDICT_TIME", "09:00")   # 9 AM
    RESULTS_TIME = os.getenv("RESULTS_TIME", "23:30")   # 11:30 PM

    schedule.every().day.at(PREDICT_TIME).do(run_predictions)
    schedule.every().day.at(RESULTS_TIME).do(run_results_update)

    logger.info("Scheduler started.")
    logger.info("  Predictions job : %s ET daily", PREDICT_TIME)
    logger.info("  Results job     : %s ET daily", RESULTS_TIME)
    logger.info("  Override times via .env: PREDICT_TIME, RESULTS_TIME")
    logger.info("  Logs → logs/scheduler.log")
    logger.info("Press Ctrl+C to stop.\n")


# ── CLI helpers ────────────────────────────────────────────────────────────────

def run_now(job: str):
    """Immediately run a job by name for testing."""
    if job == "predictions":
        run_predictions()
    elif job == "results":
        run_results_update()
    else:
        logger.error("Unknown job: %s  (use 'predictions' or 'results')", job)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLB HR Predictor scheduler")
    parser.add_argument(
        "--run-now",
        choices=["predictions", "results"],
        help="Run a job immediately instead of scheduling",
    )
    args = parser.parse_args()

    if args.run_now:
        # One-shot execution for testing
        run_now(args.run_now)
    else:
        setup_schedule()
        # Run predictions immediately on first start if it hasn't run today
        last_run_file = Path("logs/.last_prediction_run")
        today_str = date.today().isoformat()
        if not last_run_file.exists() or last_run_file.read_text().strip() != today_str:
            logger.info("First run today — executing predictions immediately ...")
            run_predictions()
            last_run_file.write_text(today_str)

        while True:
            schedule.run_pending()
            time.sleep(30)
