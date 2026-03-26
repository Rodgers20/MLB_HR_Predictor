# MLB Home Run Predictor

A machine learning system that predicts MLB home run probabilities for each batter in today's games, powered by Statcast data, advanced FanGraphs metrics, park factors, and weather conditions.

---

## Features

- **Daily HR probability predictions** for all MLB batters in scheduled games
- **Statcast-powered features**: exit velocity, barrel%, launch angle, hard-hit%
- **Handedness-specific matchup analysis**: batter vs RHP/LHP splits, pitcher vs RHB/LHB
- **Park factor + weather adjustments**: temperature, wind speed/direction, elevation
- **XGBoost model** with SHAP feature importance explainability
- **Python Dash dashboard** with 4 pages: Today's Picks, Player Deep-Dive, Model Performance, History
- **Excel tracker** for monitoring model accuracy and ROI over time

---

## Data Sources

| Source | Access | Data |
|--------|--------|------|
| Baseball Savant (Statcast) | `pybaseball` (free) | Exit velocity, launch angle, barrel%, xBA, xwOBA, xSLG, HR tracking |
| FanGraphs | `pybaseball` (free) | ISO, Pull%, FB%, HR/FB, wRC+, WAR, 334+ advanced columns |
| Baseball Reference | `pybaseball` (free) | Historical HR totals, season splits |
| MLB Stats API | `mlb-statsapi` (free) | Game schedules, lineups, player info |
| Open-Meteo | REST API (free, no key) | Temperature, wind speed/direction, humidity |
| Park Factors | Baseball Savant | HR park factor per stadium (100 = average) |

---

## Model Architecture

### Primary: XGBoost Classifier
- **Task**: Binary classification — probability of HR per at-bat (0/1)
- **Training data**: 2017–2025 Statcast seasons
- **Validation**: TimeSeriesSplit (no data leakage), backtest on 2024–2025
- **Output**: HR probability score (0.0–1.0) per batter-pitcher matchup

### Secondary: XGBoost Regressor
- **Task**: Projected HRs per game (continuous)
- **Output**: Expected HRs for the game

### Feature Set (25+ features)

**Batter Features (current season + 3yr weighted average)**
| Feature | Description |
|---------|-------------|
| barrel_pct | % of batted balls classified as barrels (EV≥98mph, LA 26-30°) |
| exit_velocity_avg | Average exit velocity (mph) |
| launch_angle_avg | Average launch angle (degrees) |
| hard_hit_pct | % of balls hit at ≥95 mph |
| hr_fb_rate | HR as % of fly balls hit |
| iso | Isolated power: (Total Bases - Hits) / AB |
| xwoba | Expected weighted on-base average |
| xslg | Expected slugging percentage |
| pull_pct | % of batted balls to pull side |
| fb_pct | % of batted balls classified as fly balls |
| hr_vs_rhp | HR rate vs right-handed pitchers |
| hr_vs_lhp | HR rate vs left-handed pitchers |
| rolling_15g_hr | HRs in last 15 games |
| home_hr_rate | HR rate at home |
| away_hr_rate | HR rate on the road |

**Pitcher Features**
| Feature | Description |
|---------|-------------|
| pitcher_hr9 | Home runs allowed per 9 innings |
| pitcher_hr_fb | HR/FB ratio allowed |
| pitcher_fip | Fielding Independent Pitching |
| pitcher_gb_pct | Ground ball % (low GB = more fly balls = more HR risk) |
| pitcher_hr_vs_rhb | HR allowed rate vs right-handed batters |
| pitcher_hr_vs_lhb | HR allowed rate vs left-handed batters |
| pitcher_recent_hr_allowed | HRs allowed in last 5 starts |

**Context Features**
| Feature | Description |
|---------|-------------|
| park_hr_factor | Stadium HR park factor (100=average, >100=easier) |
| temp_f | Game-time temperature (°F) — every 10°F ≈ 1% ball distance |
| wind_speed_mph | Wind speed at game time |
| wind_direction | In/Out/Cross/Calm (Out boosts HR probability) |
| is_day_game | Day vs night flag |

---

## Setup Instructions

### 1. Clone and install dependencies
```bash
git clone <your-repo-url>
cd MLB_HR_Predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — only MLB_ODDS_API_KEY is optional (for betting odds integration)
```

### 3. Download initial data (one-time, takes ~15 min)
```bash
python utils/data_fetcher.py --years 2017 2025
```

### 4. Train the model
```bash
python utils/model_trainer.py
```

### 5. Run daily predictions
```bash
python utils/predictor.py
```

### 6. Launch dashboard
```bash
python dashboard/app.py
# Open http://localhost:8050
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Today's Picks** | All batters ranked by HR probability, with confidence tier, park factor, weather impact |
| **Player Deep-Dive** | Individual player Statcast stats, trend charts, historical HR probability |
| **Model Performance** | Accuracy over time, calibration curve, ROI tracker, win/loss chart |
| **Prediction History** | Full prediction log filterable by date, player, result |

---

## Excel Tracker (`MLB_HR_Predictions.xlsx`)

| Sheet | Contents |
|-------|----------|
| `Predictions` | Date, player, pitcher, HR probability, confidence, park factor, weather, actual result |
| `Model_Performance` | Daily accuracy, cumulative accuracy, high-confidence hit rate, ROI |
| `Feature_Importance` | Top SHAP features per day |

---

## Model Performance

> Updated automatically after each game day.

| Metric | Value |
|--------|-------|
| Overall Accuracy | TBD (tracking in progress) |
| High Confidence Hit Rate | TBD |
| Baseline (predict 0 every time) | ~97% accuracy (HR is rare) |
| AUC-ROC | TBD |

---

## Confidence Tiers

| Tier | HR Probability | Interpretation |
|------|---------------|----------------|
| High | ≥ 18% | Strong lean — favorable matchup, park, weather |
| Medium | 12–17% | Worth monitoring |
| Low | < 12% | Informational only |

> League average HR probability per PA is approximately 3–4%. A 15%+ probability represents a significant edge.

---

## Roadmap

- **Phase 1** (current): XGBoost model, Dash dashboard, Excel tracker
- **Phase 2**: Add LSTM time-series layer for recency/seasonality weighting
- **Phase 3**: Integrate betting odds for implied probability calibration
- **Phase 4**: Multi-HR game predictions, HR derby tracking, season pace projections

---

## Dependencies

See `requirements.txt` for full list. Key packages:
- `pybaseball` — Statcast, FanGraphs, Baseball Reference data
- `xgboost` — gradient boosting model
- `shap` — model explainability
- `dash` + `plotly` — dashboard
- `openpyxl` — Excel tracker
- `mlb-statsapi` — live schedules and lineups
