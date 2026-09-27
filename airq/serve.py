"""Torch-free forecaster for serverless hosting (Vercel).

Loads the XGBoost point and quantile boosters exported to models/xgb/ and reproduces
airq.predict.forecast/explain with the XGBoost model only. The LSTM, which needs
PyTorch, stays in the full training pipeline.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import xgboost as xgb

from . import features as F
from .aqi import HEALTH_ADVICE, category, category_index
from .config import ROOT
from .data import fetch_recent, to_daily

MODEL_DIR = ROOT / "models" / "xgb"
MODEL_NAME = "XGBoost"

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
    for stem, what in (("aqi_lag", "AQI"), ("pm2_5_lag", "PM2.5"), ("pm10_lag", "PM10")):
        if name.startswith(stem):
            days = int(name[len(stem):])
            return {1: f"yesterday's {what}", 7: f"{what} a week ago"}.get(days, f"{what} {days} days ago")
    return name


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    horizons = {}
    for h, hm in meta["horizons"].items():
        boosters = {}
        for name in ("point", "quantile"):
            bst = xgb.Booster()
            bst.load_model(MODEL_DIR / f"h{h}_{name}.ubj")
            boosters[name] = bst
        horizons[int(h)] = {**hm, **boosters}
    return {"fences": {tuple(k): tuple(v) for k, v in meta["fences"]}, "horizons": horizons}


def horizons(bundle: dict) -> list[int]:
    return sorted(bundle["horizons"])


def recent_frame(city: str, past_days: int = 30) -> pd.DataFrame:
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
    return feats.loc[[int(pos[0])], cols].astype(float), feats


def _delta(bst: xgb.Booster, X: pd.DataFrame, n_iter: int, **kw) -> np.ndarray:
    return bst.predict(xgb.DMatrix(X), iteration_range=(0, n_iter), **kw)


def forecast(city: str, bundle: dict, daily: pd.DataFrame, horizon: int) -> dict:
    hb = bundle["horizons"][horizon]
    X, _ = _feature_row(daily, city, bundle, horizon)
    base = float(X["aqi"].iloc[0])
    pred = float(np.clip(base + _delta(hb["point"], X, hb["point_iter"])[0], 0, 500))
    q = base + np.ravel(_delta(hb["quantile"], X, hb["quantile_iter"]))
    m = hb["conformal_margin"]
    lo, hi = float(np.clip(q[0] - m, 0, 500)), float(np.clip(q[2] + m, 0, 500))
    today = _today()
    today_row = daily[daily["date"] == today].iloc[0]
    return {
        "city": city, "model": MODEL_NAME, "horizon": horizon,
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
    }


def explain(city: str, bundle: dict, daily: pd.DataFrame, horizon: int = 1, top: int = 6) -> pd.DataFrame:
    """Top TreeSHAP contributions to the forecast change from today's AQI, in AQI points."""
    hb = bundle["horizons"][horizon]
    X, _ = _feature_row(daily, city, bundle, horizon)
    c = pd.Series(_delta(hb["point"], X, hb["point_iter"], pred_contribs=True)[0], index=list(X.columns) + ["bias"])
    c = c.drop("bias")
    c = c[~c.index.str.startswith("city_")]
    out = c.reindex(c.abs().sort_values(ascending=False).index).head(top).to_frame("contribution")
    out["factor"] = [feature_label(i) for i in out.index]
    return out.reset_index(names="feature")
