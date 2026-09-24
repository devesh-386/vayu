"""Data collection from Open-Meteo (free, no API key).

Pollutants come from the CAMS atmospheric-composition model via Open-Meteo's
air-quality API. Weather comes from the ERA5 reanalysis archive for history and
from the forecast API for live use. Hourly data is cached under data/raw/ and
aggregated to daily values using the CPCB averaging rules.
"""
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from .config import CITIES, RAW_DIR, START_DATE, TIMEZONE

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

AQ_VARS = {
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "carbon_monoxide": "co",
    "ozone": "o3",
}
WX_VARS = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_dir",
    "precipitation": "precip",
    "surface_pressure": "pressure",
    "boundary_layer_height": "blh",
}


def _get(url: str, params: dict, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 429:  # rate limited: back off
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo request failed: {url}")


def _hourly_frame(payload: dict, rename: dict) -> pd.DataFrame:
    h = payload["hourly"]
    df = pd.DataFrame({new: h[old] for old, new in rename.items()})
    df.index = pd.to_datetime(h["time"])
    df.index.name = "time"
    return df.astype(float)


def _year_chunks(start: str, end: str):
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    while s <= e:
        chunk_end = min(date(s.year, 12, 31), e)
        yield s.isoformat(), chunk_end.isoformat()
        s = chunk_end + timedelta(days=1)


def fetch_history(city: str, start: str, end: str) -> pd.DataFrame:
    """Hourly pollutants + weather for one city between two ISO dates (inclusive)."""
    lat, lon = CITIES[city]
    parts = []
    for s, e in _year_chunks(start, end):
        base = {"latitude": lat, "longitude": lon, "start_date": s, "end_date": e, "timezone": TIMEZONE}
        aq = _hourly_frame(_get(AQ_URL, {**base, "hourly": ",".join(AQ_VARS)}), AQ_VARS)
        wx = _hourly_frame(_get(ARCHIVE_URL, {**base, "hourly": ",".join(WX_VARS)}), WX_VARS)
        parts.append(aq.join(wx, how="outer"))
    return pd.concat(parts)


def fetch_recent(city: str, past_days: int = 30) -> pd.DataFrame:
    """Hourly data for the last `past_days` days through three days ahead (live inference).

    Today's and tomorrow's pollutant hours beyond "now" are CAMS forecasts, and
    tomorrow's weather is a numerical weather forecast. Those are exactly the
    inputs available to an operational forecaster.
    """
    lat, lon = CITIES[city]
    base = {"latitude": lat, "longitude": lon, "timezone": TIMEZONE, "past_days": past_days, "forecast_days": 4}
    aq = _hourly_frame(_get(AQ_URL, {**base, "hourly": ",".join(AQ_VARS)}), AQ_VARS)
    wx = _hourly_frame(_get(FORECAST_URL, {**base, "hourly": ",".join(WX_VARS)}), WX_VARS)
    return aq.join(wx, how="outer")


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly data to daily values following CPCB averaging periods."""
    h = hourly.copy()
    h["co"] = h["co"] / 1000.0  # ug/m3 -> mg/m3, the unit CPCB breakpoints use
    # CO and O3 use the maximum 8-hour running mean of the day.
    for p in ("co", "o3"):
        h[f"{p}_8h"] = h[p].rolling(8, min_periods=6).mean()
    rad = np.deg2rad(h["wind_dir"])
    # Wind direction is circular; average it as u/v components instead of degrees.
    h["wind_u"] = -h["wind_speed"] * np.sin(rad)
    h["wind_v"] = -h["wind_speed"] * np.cos(rad)

    g = h.groupby(h.index.normalize())
    daily = pd.DataFrame({
        "pm2_5": g["pm2_5"].mean(),
        "pm10": g["pm10"].mean(),
        "no2": g["no2"].mean(),
        "so2": g["so2"].mean(),
        "co": g["co_8h"].max(),
        "o3": g["o3_8h"].max(),
        "temp": g["temp"].mean(),
        "temp_max": g["temp"].max(),
        "temp_min": g["temp"].min(),
        "humidity": g["humidity"].mean(),
        "wind_speed": g["wind_speed"].mean(),
        "wind_u": g["wind_u"].mean(),
        "wind_v": g["wind_v"].mean(),
        "precip": g["precip"].sum(min_count=1),
        "pressure": g["pressure"].mean(),
        "blh": g["blh"].mean(),
        "n_hours": g["pm2_5"].count(),
    })
    # A day needs >= 16 hourly pollutant readings to count (CPCB uses the same 2/3 rule).
    thin = daily["n_hours"] < 16
    daily.loc[thin, ["pm2_5", "pm10", "no2", "so2", "co", "o3"]] = np.nan
    daily.index.name = "date"
    return daily.drop(columns="n_hours")


def history_end() -> str:
    # ERA5 lags real time by about five days.
    return (date.today() - timedelta(days=7)).isoformat()


def load_dataset(refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Daily multi-city dataset, fetched once and cached as data/raw/<city>.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    end = history_end()
    frames = []
    for city in CITIES:
        path = RAW_DIR / f"{city.lower()}_hourly.csv"
        if path.exists() and not refresh:
            hourly = pd.read_csv(path, index_col="time", parse_dates=True)
        else:
            if verbose:
                print(f"  fetching {city} {START_DATE}..{end}")
            hourly = fetch_history(city, START_DATE, end)
            hourly.to_csv(path)
        daily = to_daily(hourly)
        daily.insert(0, "city", city)
        frames.append(daily.reset_index())
    return pd.concat(frames, ignore_index=True)
