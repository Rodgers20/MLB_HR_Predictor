# MLB HR Predictor - Task List

## Phase 1: Setup ✅
- [x] Create project directory structure
- [x] Write README.md
- [x] Create requirements.txt (2017–2025 data range)
- [x] Create .env.example
- [x] Initialize CLAUDE.MD

## Phase 2: Data Pipeline ✅
- [x] `utils/data_fetcher.py` — pybaseball wrappers + CSV caching
- [x] `data/park_factors.csv` — manual HR park factors for all 30 stadiums
- [x] `utils/weather_fetcher.py` — Open-Meteo integration with stadium coords

## Phase 3: Feature Engineering ✅
- [x] `utils/feature_engineer.py` — 25+ features, 3yr weighted avg, wind encoding

## Phase 4: Model Training ✅
- [x] `utils/model_trainer.py` — XGBoost, TimeSeriesSplit, SMOTE, SHAP
- [ ] **TODO**: Run `python utils/model_trainer.py --start 2017 --end 2025` to train

## Phase 5: Daily Predictions ✅
- [x] `utils/predictor.py` — mlb-statsapi lineups, matchup features, confidence tiers

## Phase 6: Dashboard ✅
- [x] `dashboard/app.py` — Dash multi-page app
- [x] `dashboard/pages/today.py` — Today's picks with filters
- [x] `dashboard/pages/player.py` — Player deep-dive
- [x] `dashboard/pages/model_perf.py` — Accuracy, ROI, feature importance
- [x] `dashboard/pages/history.py` — Full prediction log
- [x] `dashboard/assets/custom.css` — Navy/green theme

## Phase 7: Excel Tracker ✅
- [x] `tracker/prediction_tracker.py` — Predictions + Model_Performance + Feature_Importance sheets

## Next Steps (Manual)
1. `pip install -r requirements.txt`
2. `python utils/data_fetcher.py --years 2017 2025`  (one-time, ~15 min)
3. `python utils/model_trainer.py`  (train model)
4. `python utils/predictor.py --save`  (run today's predictions)
5. `python dashboard/app.py`  (launch dashboard at localhost:8050)

## After Each Game Day
- `python tracker/prediction_tracker.py update --results "Aaron Judge:1,Mike Trout:0"`
  (or manually fill Actual_HRs column in Excel)
