"""Preprocessing and feature engineering.

Pipeline (matches the block diagram):
  1. cleaning       - regular daily index per city, physically impossible values -> NaN
  2. missing values - short gaps (<= 3 days) interpolated in time, longer gaps dropped
  3. outliers       - IQR fences learned on the training period only, inputs clipped
  4. features       - lags, rolling stats, weather, tomorrow's weather forecast, calendar
  5. normalisation  - done inside each model pipeline (StandardScaler), fit on train only

Target: AQI of the NEXT day. Using same-day pollutants to "predict" same-day AQI
would be circular, because AQI is a fixed formula of those pollutants.
"""
import numpy as np
import pandas as pd

from .aqi import aqi_frame
from .config import CITIES, POLLUTANTS, TRAIN_END, VAL_END

# Upper limits beyond which a daily reading is a sensor or model fault, not air.
PHYSICAL_MAX = {"pm2_5": 1000, "pm10": 2000, "no2": 1000, "so2": 2000, "co": 50, "o3": 1000}
MAX_GAP_DAYS = 3
LAGS = (1, 2, 3, 7)
FORECAST_WEATHER = ["temp", "humidity", "wind_speed", "wind_u", "wind_v", "precip", "pressure", "blh"]
WEATHER_COLS = FORECAST_WEATHER + ["temp_max", "temp_min"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Regular daily index per city, impossible values removed, short gaps filled."""
    out = []
    for city, g in df.groupby("city", sort=False):
        g = g.drop_duplicates("date").set_index("date").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
        g["city"] = city
        for p, hi in PHYSICAL_MAX.items():
            g.loc[(g[p] < 0) | (g[p] > hi), p] = np.nan
        num = g.columns.drop("city")
        g[num] = g[num].interpolate(method="time", limit=MAX_GAP_DAYS, limit_area="inside")
        # Longer weather gaps (ERA5 has no boundary-layer height for Jan-Jun 2024) are
        # filled with that city's monthly median from the training period.
        train_part = g[g.index <= TRAIN_END]
        for w in WEATHER_COLS:
            if g[w].isna().any() and len(train_part):
                clim = train_part.groupby(train_part.index.month)[w].median()
                g[w] = g[w].fillna(pd.Series(g.index.month.map(clim), index=g.index))
        g.index.name = "date"
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def add_aqi(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, aqi_frame(df)], axis=1)


def outlier_fences(df: pd.DataFrame, k: float = 3.0) -> dict:
    """Per-city IQR fences on log concentrations, fit on the training period only.

    Log scale because pollution is right-skewed: festival and crop-burning spikes
    are real events we must keep, so only extreme outliers get clipped.
    """
    train = df[df["date"] <= TRAIN_END]
    fences = {}
    for city, g in train.groupby("city"):
        for p in POLLUTANTS:
            x = np.log1p(g[p].dropna())
            q1, q3 = x.quantile([0.25, 0.75])
            iqr = q3 - q1
            fences[(city, p)] = (float(np.expm1(q1 - k * iqr)), float(np.expm1(q3 + k * iqr)))
    return fences


def clip_outliers(df: pd.DataFrame, fences: dict) -> tuple[pd.DataFrame, int]:
    """Clip pollutant inputs to the fences; returns the frame and the number of clipped cells."""
    df = df.copy()
    n = 0
    for (city, p), (lo, hi) in fences.items():
        m = df["city"] == city
        before = df.loc[m, p]
        after = before.clip(lower=max(lo, 0), upper=hi)
        n += int((before.notna() & (before != after)).sum())
        df.loc[m, p] = after
    return df, n


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature table: one row per (city, day t) with target = AQI on day t+1.

    `df` must already contain the aqi column (computed on unclipped data) and the
    pollutant columns (optionally clipped).
    """
    rows = []
    for city, g in df.groupby("city", sort=False):
        g = g.sort_values("date").set_index("date")
        f = pd.DataFrame(index=g.index)
        f["city"] = city
        for p in POLLUTANTS:
            f[p] = g[p]
        f["aqi"] = g["aqi"]
        for lag in LAGS:
            f[f"aqi_lag{lag}"] = g["aqi"].shift(lag)
            f[f"pm2_5_lag{lag}"] = g["pm2_5"].shift(lag)
        f["pm10_lag1"] = g["pm10"].shift(1)
        f["aqi_roll3"] = g["aqi"].rolling(3, min_periods=2).mean()
        f["aqi_roll7"] = g["aqi"].rolling(7, min_periods=5).mean()
        f["aqi_std7"] = g["aqi"].rolling(7, min_periods=5).std()
        f["aqi_diff1"] = g["aqi"].diff()
        f["pm2_5_roll7"] = g["pm2_5"].rolling(7, min_periods=5).mean()
        for w in ("temp", "temp_max", "temp_min", "humidity", "wind_speed", "wind_u", "wind_v", "precip", "pressure", "blh"):
            f[w] = g[w]
        # Tomorrow's weather: in live use this comes from the numerical weather forecast.
        for w in FORECAST_WEATHER:
            f[f"next_{w}"] = g[w].shift(-1)
        f["temp_change"] = f["next_temp"] - f["temp"]
        doy = g.index.dayofyear
        f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        f["weekday"] = g.index.dayofweek
        f["is_weekend"] = (g.index.dayofweek >= 5).astype(int)
        f["target_aqi"] = g["aqi"].shift(-1)
        f["target_date"] = f.index + pd.Timedelta(days=1)
        rows.append(f.reset_index())
    feats = pd.concat(rows, ignore_index=True)
    for c in CITIES:
        feats[f"city_{c}"] = (feats["city"] == c).astype(int)
    return feats


def feature_columns(feats: pd.DataFrame) -> list[str]:
    exclude = {"date", "city", "target_aqi", "target_date"}
    return [c for c in feats.columns if c not in exclude]


def split(feats: pd.DataFrame):
    """Chronological train / validation / test split on the target date."""
    td = feats["target_date"]
    return (
        feats[td <= TRAIN_END],
        feats[(td > TRAIN_END) & (td <= VAL_END)],
        feats[td > VAL_END],
    )


def prepare(daily: pd.DataFrame):
    """Full preprocessing: returns (feature table, fences, report dict)."""
    report = {"raw_rows": len(daily), "raw_missing_pct": daily[POLLUTANTS].isna().mean().mul(100).round(2).to_dict()}
    cleaned = clean(daily)
    report["after_clean_missing_pct"] = cleaned[POLLUTANTS].isna().mean().mul(100).round(2).to_dict()
    with_aqi = add_aqi(cleaned)
    fences = outlier_fences(with_aqi)
    clipped, n_clipped = clip_outliers(with_aqi, fences)
    report["outlier_cells_clipped"] = n_clipped
    feats = build_features(clipped)
    before = len(feats)
    feats = feats.dropna().reset_index(drop=True)
    report["feature_rows"] = len(feats)
    report["rows_dropped_incomplete"] = before - len(feats)
    return feats, fences, report
