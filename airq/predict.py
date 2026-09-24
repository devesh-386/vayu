"""Live AQI forecasts (1-3 days ahead) for a city using the trained models.

    python -m airq.predict Chennai
"""
import sys
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import features as F
from .aqi import HEALTH_ADVICE, category, category_index
from .config import MODELS_DIR
from .data import fetch_recent, to_daily
from .models import LSTMForecaster, contributions, sequence_index

FEATURE_LABELS = {
    "aqi": "today's AQI", "o3": "today's ozone", "pm2_5": "today's PM2.5", "pm10": "today's PM10",
    "no2": "today's NO2", "so2": "today's SO2", "co": "today's CO",
    "aqi_roll7": "7-day AQI average", "aqi_roll3": "3-day AQI average", "aqi_std7": "7-day AQI variability",
    "aqi_diff1": "AQI trend", "pm2_5_roll7": "7-day PM2.5 average", "temp_change": "temperature change",
    "diwali_days": "Diwali timing", "diwali_window": "Diwali", "stubble_season": "crop-burning season",
    "doy_sin": "season", "doy_cos": "season", "weekday": "day of week", "is_weekend": "weekend",
}
WEATHER_LABELS = {"temp": "temperature", "temp_max": "max temperature", "temp_min": "min temperature",
                  "humidity": "humidity", "wind_speed": "wind speed", "wind_u": "wind (east-west)",
                  "wind_v": "wind (north-south)", "precip": "rain", "pressure": "pressure", "blh": "mixing height"}


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("next_") and name[5:] in WEATHER_LABELS:
        return "forecast " + WEATHER_LABELS[name[5:]]
    if name in WEATHER_LABELS:
        return "today's " + WEATHER_LABELS[name]
    for stem, label in (("aqi_lag", "AQI {} day(s) ago"), ("pm2_5_lag", "PM2.5 {} day(s) ago"), ("pm10_lag", "PM10 {} day(s) ago")):
        if name.startswith(stem):
            return label.format(name[len(stem):])
    return name


@lru_cache(maxsize=1)
def load_bundle(path=None) -> dict:
    b = joblib.load(path or MODELS_DIR / "aqi_models.joblib")
    for hb in b["horizons"].values():
        if "lstm" in hb:
            hb["lstm_model"] = LSTMForecaster.from_state(hb["lstm"])
    return b


def horizons(bundle: dict) -> list[int]:
    return sorted(bundle["horizons"])


def model_names(bundle: dict, horizon: int = 1) -> list[str]:
    hb = bundle["horizons"][horizon]
    return list(hb["sklearn"]) + (["LSTM"] if "lstm_model" in hb else [])


def recent_frame(city: str, past_days: int = 30) -> pd.DataFrame:
    """Daily pollutants, weather and AQI for the last `past_days` days through three days ahead."""
    daily = to_daily(fetch_recent(city, past_days)).reset_index()
    daily.insert(1, "city", city)
    return F.add_aqi(F.clean(daily))


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)


def _feature_row(daily, city, bundle, horizon):
    clipped, _ = F.clip_outliers(daily, {k: v for k, v in bundle["fences"].items() if k[0] == city})
    feats = F.build_features(clipped, horizon).sort_values("date").reset_index(drop=True)
    cols = bundle["horizons"][horizon]["selected"]
    pos = feats.index[feats["date"] == _today()]
    if len(pos) == 0:
        raise ValueError(f"No data for today in {city}")
    missing = feats.loc[pos[0], cols].isna()
    if missing.any():
        raise ValueError(f"Not enough recent data for {city} (missing: {missing[missing].index.tolist()})")
    return feats, int(pos[0]), cols


def forecast(city: str, model: str | None = None, bundle: dict | None = None,
             daily: pd.DataFrame | None = None, horizon: int = 1) -> dict:
    """Forecast the AQI `horizon` days ahead from today's conditions and the weather forecast."""
    bundle = bundle or load_bundle()
    hb = bundle["horizons"][horizon]
    model = model or hb["best_model"]
    if daily is None:
        daily = recent_frame(city)
    feats, pos, cols = _feature_row(daily, city, bundle, horizon)
    X = feats.loc[[pos], cols]

    if model == "LSTM":
        lstm = hb["lstm_model"]
        if pos not in set(sequence_index(feats, lstm.seq_len)):
            raise ValueError("LSTM needs 14 consecutive days of history")
        pred = float(lstm.predict(feats, np.array([pos]))[0])
    else:
        pred = float(hb["sklearn"][model].predict(X)[0])
    pred = float(np.clip(pred, 0, 500))
    q = hb["quantile"].predict(X)[0]
    m = hb.get("conformal_margin", 0.0)
    lo, hi = float(np.clip(q[0] - m, 0, 500)), float(np.clip(q[2] + m, 0, 500))

    today = _today()
    today_row = daily[daily["date"] == today].iloc[0]
    return {
        "city": city,
        "model": model,
        "horizon": horizon,
        "date": today.date().isoformat(),
        "target_date": (today + pd.Timedelta(days=horizon)).date().isoformat(),
        "today_aqi": float(today_row["aqi"]),
        "today_category": category(float(today_row["aqi"])),
        "today_dominant": today_row["dominant"],
        "predicted_aqi": round(pred, 1),
        "predicted_category": category(pred),
        "low": round(min(lo, pred), 1),
        "high": round(max(hi, pred), 1),
        "advice": HEALTH_ADVICE[category_index(pred)],
        "alert": pred > 200,
        "possible_alert": pred <= 200 < hi,
        "daily": daily,
    }


def explain(city: str, bundle: dict | None = None, daily: pd.DataFrame | None = None,
            horizon: int = 1, top: int = 6) -> pd.DataFrame:
    """Top SHAP contributions (XGBoost) to the forecast change from today's AQI, in AQI points."""
    bundle = bundle or load_bundle()
    if daily is None:
        daily = recent_frame(city)
    feats, pos, cols = _feature_row(daily, city, bundle, horizon)
    c = contributions(bundle["horizons"][horizon]["sklearn"]["XGBoost"], feats.loc[[pos], cols]).iloc[0]
    c = c.drop("bias")
    c = c[~c.index.str.startswith("city_")]
    out = c.reindex(c.abs().sort_values(ascending=False).index).head(top).to_frame("contribution")
    out["factor"] = [feature_label(i) for i in out.index]
    out["value"] = feats.loc[pos, out.index].to_numpy()
    return out.reset_index(names="feature")


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "Chennai"
    b = load_bundle()
    daily = recent_frame(city)
    for h in horizons(b):
        r = forecast(city, None, b, daily, h)
        print(f"+{h}d {r['target_date']} {r['city']}: AQI {r['predicted_aqi']:6.1f} ({r['predicted_category']}) "
              f"80% range {r['low']:.0f}-{r['high']:.0f} [{r['model']}] | today {r['today_aqi']:.0f}")
    print("\nWhy (next day, XGBoost SHAP, AQI points vs today):")
    print(explain(city, b, daily).to_string(index=False))


if __name__ == "__main__":
    main()
