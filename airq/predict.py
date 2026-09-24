"""Live next-day AQI forecast for a city using the trained models.

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
from .models import LSTMForecaster, sequence_index


@lru_cache(maxsize=1)
def load_bundle(path=None) -> dict:
    b = joblib.load(path or MODELS_DIR / "aqi_models.joblib")
    if "lstm" in b:
        b["lstm_model"] = LSTMForecaster.from_state(b["lstm"])
    return b


def model_names(bundle: dict) -> list[str]:
    return list(bundle["sklearn"]) + (["LSTM"] if "lstm_model" in bundle else [])


def recent_frame(city: str, past_days: int = 30) -> pd.DataFrame:
    """Daily pollutants, weather and AQI for the last `past_days` days through tomorrow."""
    daily = to_daily(fetch_recent(city, past_days)).reset_index()
    daily.insert(1, "city", city)
    return F.add_aqi(F.clean(daily))


def forecast(city: str, model: str | None = None, bundle: dict | None = None, daily: pd.DataFrame | None = None) -> dict:
    """Forecast tomorrow's AQI for `city` from today's conditions and tomorrow's weather forecast."""
    bundle = bundle or load_bundle()
    model = model or bundle["best_model"]
    if daily is None:
        daily = recent_frame(city)
    today = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)

    clipped, _ = F.clip_outliers(daily, {k: v for k, v in bundle["fences"].items() if k[0] == city})
    feats = F.build_features(clipped).sort_values("date").reset_index(drop=True)
    cols = bundle["selected"]
    row_pos = feats.index[feats["date"] == today]
    if len(row_pos) == 0 or feats.loc[row_pos[0], cols].isna().any():
        missing = [] if len(row_pos) == 0 else feats.loc[row_pos[0], cols][feats.loc[row_pos[0], cols].isna()].index.tolist()
        raise ValueError(f"Not enough recent data for {city} (missing: {missing or 'today'})")
    pos = int(row_pos[0])

    if model == "LSTM":
        lstm = bundle["lstm_model"]
        if pos not in set(sequence_index(feats, lstm.seq_len)):
            raise ValueError("LSTM needs 14 consecutive days of history")
        pred = float(lstm.predict(feats, np.array([pos]))[0])
    else:
        pred = float(bundle["sklearn"][model].predict(feats.loc[[pos], cols])[0])
    pred = float(np.clip(pred, 0, 500))

    today_row = daily[daily["date"] == today].iloc[0]
    ci = category_index(pred)
    return {
        "city": city,
        "model": model,
        "date": today.date().isoformat(),
        "target_date": (today + pd.Timedelta(days=1)).date().isoformat(),
        "today_aqi": float(today_row["aqi"]),
        "today_category": category(float(today_row["aqi"])),
        "today_dominant": today_row["dominant"],
        "predicted_aqi": round(pred, 1),
        "predicted_category": category(pred),
        "advice": HEALTH_ADVICE[ci],
        "alert": pred > 200,
        "daily": daily,
    }


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "Chennai"
    b = load_bundle()
    daily = recent_frame(city)
    for name in model_names(b):
        r = forecast(city, name, b, daily)
        print(f"{name:18s} {r['city']} {r['target_date']}: AQI {r['predicted_aqi']:6.1f} ({r['predicted_category']})"
              f"  | today {r['today_aqi']:.0f} ({r['today_category']})")


if __name__ == "__main__":
    main()
