# Intelligent Air Quality Prediction System

**21CSC305P Machine Learning · SRM IST Ramapuram · Batch 10**
Hasini S (RA2411003020414) · Roshni Rajakumari M P (RA2411003020419) · Navashri N M (RA2411003020420)
Supervisor: Dr. M. B. Sudhan

Forecasts the **Air Quality Index 1, 2 and 3 days ahead** for 8 Indian cities from today's pollutant levels (PM2.5, PM10, CO, NO₂, SO₂, O₃), weather (temperature, humidity, wind, rain, pressure, boundary-layer height), the weather forecast and festival/season calendar. Each forecast comes with an 80% uncertainty range and a SHAP explanation of *why*. It classifies forecasts into CPCB categories and raises an early warning when the air is expected to be Poor or worse.

## Quick start

```bash
pip install -r requirements.txt
python -m airq.train          # fetch data, train all models for 1-3 day horizons (~8 min CPU), write reports/
streamlit run app.py          # dashboard at http://localhost:8501
python -m airq.predict Delhi  # command-line forecast
python -m pytest              # 28 unit tests
```

No API keys are needed. The processed daily dataset (`data/daily.csv`) is committed, so training works offline. The trained models are not committed (47 MB); `python -m airq.train` rebuilds them in about 8 minutes on a CPU.

## How it maps to the block diagram

| Block (review deck) | Implementation |
|---|---|
| Environmental data sources | `airq/data.py`: hourly pollutants from the CAMS model and weather from ERA5, both via [Open-Meteo](https://open-meteo.com); 8 cities, Aug 2022 → present |
| Data collection | cached hourly CSVs → daily values using CPCB averaging rules (24-h mean; 8-h max for CO, O₃) |
| Historical data | 12,048 city-days (1,506 days × 8 cities) |
| Data preprocessing | `airq/features.py`: regular daily index, impossible values removed, gaps ≤ 3 days interpolated, a 6-month ERA5 gap in boundary-layer height filled with training-period monthly medians, IQR outlier fences (log scale, fit on training data only, 48 readings clipped), standardisation inside each model |
| Feature engineering | lags, rolling stats, target-day weather forecast, season, Diwali proximity, crop-burning season (Delhi/Lucknow) |
| Feature selection | Random Forest permutation importance on the validation set: **34 of 55** features kept (next-day model) |
| ML model | Persistence baseline, Linear Regression, **Random Forest**, **XGBoost**, **LSTM** (`airq/models.py`) |
| Model training | `airq/train.py`: chronological split, hyper-parameter grid on validation, early stopping |
| AQI prediction | AQI 1, 2, 3 days ahead with an 80% range (quantile XGBoost + conformal calibration) and SHAP explanation (`airq/predict.py`) |
| Air quality classification | CPCB National AQI (`airq/aqi.py`) |
| Visualisation & alerts | Streamlit dashboard (`app.py`), alert banner, optional email alert (`airq/alerts.py`) |

## Why next-day prediction

The AQI is a fixed formula of the pollutant concentrations (the worst sub-index), so "predicting" today's AQI from today's pollutants would just re-learn the formula and report near-perfect accuracy. This project forecasts **tomorrow's** AQI from information available **today**, which is the useful early-warning task. The tests in `tests/test_features.py` check that no future information leaks into the features.

## Results (held-out test set, 1 Jul 2025 → 18 Sep 2026, 3,560 city-days)

The data is split by time: train Aug 2022–Dec 2024, validation Jan–Jun 2025, test Jul 2025 onward. Every model is scored only on days after its training data.

**Next-day forecast:**

| Model | MAE | RMSE | R² | Category accuracy | Within 1 category | Alert precision | Alert recall |
|---|---|---|---|---|---|---|---|
| Persistence (tomorrow = today) | 25.16 | 42.35 | 0.753 | 75.5% | 97.9% | 78% | 78% |
| Linear Regression | 24.01 | 37.36 | 0.808 | 73.9% | 99.1% | 79% | 79% |
| Random Forest | 22.42 | 37.36 | 0.808 | 75.7% | 99.0% | 77% | 84% |
| XGBoost | 22.05 | 37.12 | 0.811 | 77.2% | 99.0% | 77% | **85%** |
| **LSTM** | **21.71** | **36.57** | **0.816** | **77.6%** | 99.0% | **80%** | 82% |

**Further ahead (test MAE):**

| Model | +1 day | +2 days | +3 days |
|---|---|---|---|
| Persistence | 25.16 | 34.04 | 37.98 |
| Linear Regression | 24.01 | 30.41 | 32.17 |
| Random Forest | 22.42 | 27.90 | 30.02 |
| XGBoost | 22.05 | **26.41** | **28.65** |
| LSTM | **21.71** | 27.29 | 29.06 |
| 80% range: actual value inside | 85% | 86% | 86% |

- The LSTM, chosen on validation MAE, is **14% more accurate than the persistence baseline** next day (MAE 21.7 vs 25.2). Air quality changes slowly, so "tomorrow = today" is a strong baseline; beating it is the real test.
- The ML advantage grows with lead time: at 3 days XGBoost is **25% better** than persistence (28.7 vs 38.0).
- XGBoost and Random Forest catch the most Poor-or-worse days (84–85% recall).
- Tree models and the LSTM predict the *change* from today's AQI. Predicting the level directly left Random Forest and XGBoost with an RMSE of about 41, barely better than persistence, because trees cannot extrapolate to unseen highs.
- **Uncertainty:** raw quantile XGBoost covered only 72–74% of outcomes with its "80%" range (over-confident). Conformal calibration on the validation set widens it; on test it covers 85–86%, slightly conservative because the test period was more volatile than validation.
- **SHAP** (average impact, AQI points): today's AQI 13.1, forecast wind speed 8.5, forecast wind direction 4.6, today's ozone 3.7, forecast humidity 3.5, temperature change 3.3, forecast rain 3.0. Wind and rain disperse and wash out pollution, as physics says. The dashboard shows the same breakdown for every live forecast.

Figures are in `reports/figures/`, and all numbers are in `reports/metrics.json`.

## AQI categories (CPCB National AQI)

| AQI | Category |
|---|---|
| 0–50 | Good |
| 51–100 | Satisfactory |
| 101–200 | Moderate |
| 201–300 | Poor |
| 301–400 | Very Poor |
| 401–500 | Severe |

> **Note for the deck:** the block diagram labels 51–100 "Moderate" and 101–200 "Unhealthy for Sensitive Groups". Those are US EPA names on Indian bands. The CPCB names are **Satisfactory** and **Moderate**.

## Limitations (state these openly in the review)

1. **Model data, not station data.** Pollutant values come from the CAMS global atmospheric model (about 40 km grid), not CPCB monitoring stations. CAMS is known to **overestimate surface ozone over India**: O₃ is the dominant pollutant on most days here, while station data for cities like Chennai is usually PM-driven. It also shows dust (PM10) episodes in Delhi during the 2026 monsoon that station data would likely not confirm. The ML pipeline is source-independent. Training on CPCB station data (e.g. the Kaggle *Air Quality Data in India* dataset) would give more realistic absolute AQI values.
2. **City-level, not street-level.** One grid point per city.
3. **The weather forecast is idealised in training.** Training uses observed (ERA5) weather for "tomorrow". In live use it comes from a forecast, which is slightly less accurate.
4. **Diwali/crop-burning features have little effect here** because CAMS does not model firecracker smoke well; with station data they should matter more.
5. **Email alerts need SMTP settings** (`AQI_SMTP_HOST`, `AQI_SMTP_USER`, `AQI_SMTP_PASSWORD`, `AQI_ALERT_TO`). SMS and a mobile app are not implemented. The dashboard works in a phone browser.

## Project layout

```
airq/
  config.py     cities, dates, split, paths
  data.py       Open-Meteo collection + daily aggregation
  aqi.py        CPCB AQI calculator and categories
  features.py   cleaning, missing values, outliers, feature engineering, split
  models.py     persistence, ridge, RF, XGBoost, LSTM
  evaluate.py   MAE/RMSE/R², category and alert metrics
  train.py      end-to-end training + figures + reports
  predict.py    live next-day forecast
  alerts.py     early-warning message / email
app.py          Streamlit dashboard
tests/          unit tests (AQI maths, leakage, preprocessing, split)
reports/        metrics.json, model comparison, figures
```
