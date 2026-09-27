"""Vercel Python function: live AQI forecasts for the web frontend.

Runs the torch-free XGBoost forecaster (airq.serve). Responses are cached for 30 minutes
in the function instance and on Vercel's CDN.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response

from airq.aqi import HEALTH_ADVICE, category_index
from airq.config import CITIES, REPORTS_DIR
from airq.serve import explain, forecast, horizons, load_bundle, recent_frame

CACHE_TTL = 30 * 60
CDN_CACHE = f"public, s-maxage={CACHE_TTL}, stale-while-revalidate=3600"

app = FastAPI(title="Vayu AQI forecast API")
_cache: dict[str, tuple[float, dict]] = {}
_lock = Lock()


def _num(x, nd=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _city_payload(city: str) -> dict:
    b = load_bundle()
    daily = recent_frame(city)
    outlook = [forecast(city, b, daily, h) for h in horizons(b)]
    today = pd.Timestamp(outlook[0]["date"])
    row = daily[daily["date"] == today].iloc[0]
    hist = daily[daily["date"] <= today].tail(30)
    why = explain(city, b, daily)
    lat, lon = CITIES[city]
    return {
        "city": city, "lat": lat, "lon": lon,
        "updated": pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%dT%H:%M:%S"),
        "today": {
            "date": outlook[0]["date"],
            "aqi": _num(row["aqi"], 0),
            "category": outlook[0]["today_category"],
            "dominant": row["dominant"],
            "advice": HEALTH_ADVICE[category_index(float(row["aqi"]))],
            "pollutants": {p: {"value": _num(row[p], 2 if p == "co" else 1), "sub_index": _num(row[f"si_{p}"], 0)}
                           for p in ("pm2_5", "pm10", "no2", "so2", "co", "o3")},
            "weather": {"temp": _num(row["temp"]), "humidity": _num(row["humidity"], 0),
                        "wind_speed": _num(row["wind_speed"]), "precip": _num(row["precip"])},
        },
        "outlook": [{
            "horizon": r["horizon"], "date": r["target_date"], "aqi": _num(r["predicted_aqi"], 0),
            "low": _num(r["low"], 0), "high": _num(r["high"], 0), "category": r["predicted_category"],
            "model": r["model"], "alert": bool(r["alert"]), "possible_alert": bool(r["possible_alert"]),
        } for r in outlook],
        "history": [{"date": d.strftime("%Y-%m-%d"), "aqi": _num(a, 0)} for d, a in zip(hist["date"], hist["aqi"])],
        "why": [{"factor": f, "contribution": _num(c)} for f, c in zip(why["factor"], why["contribution"])],
    }


def city_payload(city: str) -> dict:
    now = time.time()
    with _lock:
        hit = _cache.get(city)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1]
    data = _city_payload(city)
    with _lock:
        _cache[city] = (now, data)
    return data


@app.get("/api/city/{city}")
def city(city: str, response: Response):
    match = next((c for c in CITIES if c.lower() == city.lower()), None)
    if match is None:
        raise HTTPException(404, f"Unknown city: {city}")
    try:
        data = city_payload(match)
    except ValueError as e:
        raise HTTPException(503, str(e))
    response.headers["Cache-Control"] = CDN_CACHE
    return data


@app.get("/api/overview")
def overview(response: Response):
    def one(c):
        try:
            d = city_payload(c)
            return {"city": c, "today": {k: d["today"][k] for k in ("date", "aqi", "category", "dominant")},
                    "outlook": d["outlook"]}
        except Exception as e:  # one failing city should not break the board
            return {"city": c, "error": str(e)}
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(one, CITIES))
    if not any("error" in r for r in rows):
        response.headers["Cache-Control"] = CDN_CACHE
    return rows


@app.get("/api/model")
def model(response: Response):
    r = json.loads((REPORTS_DIR / "metrics.json").read_text())
    response.headers["Cache-Control"] = "public, s-maxage=86400"
    return {
        "generated": r["generated"],
        "data": r["data"],
        "split": r["split"],
        "best_model": r["best_model"],
        "results": {n: v["test"] for n, v in r["results"].items()},
        "horizons": r["horizons"],
        "test_mae_by_city": r["test_mae_by_city"],
        "shap_top": r.get("shap_top15", {}),
        "classification": r.get("classification"),
        "features": {"all": r["features"]["all"], "selected": len(r["features"]["selected"])},
    }
