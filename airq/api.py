"""JSON API for the web frontend, plus static hosting of the built site.

    uvicorn airq.api:app --port 8000
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .aqi import CATEGORIES, CATEGORY_COLORS, HEALTH_ADVICE, category_index
from .config import CITIES, REPORTS_DIR, ROOT
from .predict import explain, forecast, horizons, load_bundle, recent_frame

CACHE_TTL = 30 * 60
WEB_DIST = ROOT / "web" / "dist"

app = FastAPI(title="Vayu AQI forecast API")
_cache: dict[str, tuple[float, dict]] = {}
_lock = Lock()


def _num(x, nd=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _city_payload(city: str) -> dict:
    b = load_bundle()
    daily = recent_frame(city)
    outlook = [forecast(city, None, b, daily, h) for h in horizons(b)]
    today = pd.Timestamp(outlook[0]["date"])
    row = daily[daily["date"] == today].iloc[0]
    hist = daily[daily["date"] <= today].tail(30)
    why = explain(city, b, daily)
    lat, lon = CITIES[city]
    return {
        "city": city,
        "lat": lat,
        "lon": lon,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
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


@app.get("/api/meta")
def meta():
    return {"cities": [{"name": c, "lat": la, "lon": lo} for c, (la, lo) in CITIES.items()],
            "categories": [{"name": n, "color": c, "max": m} for n, c, m in
                           zip(CATEGORIES, CATEGORY_COLORS, (50, 100, 200, 300, 400, 500))]}


@app.get("/api/city/{city}")
def city(city: str):
    match = next((c for c in CITIES if c.lower() == city.lower()), None)
    if match is None:
        raise HTTPException(404, f"Unknown city: {city}")
    try:
        return city_payload(match)
    except ValueError as e:
        raise HTTPException(503, str(e))


@app.get("/api/overview")
def overview():
    def one(c):
        try:
            d = city_payload(c)
            return {"city": c, "today": {k: d["today"][k] for k in ("date", "aqi", "category", "dominant")},
                    "outlook": d["outlook"]}
        except Exception as e:  # one failing city should not break the board
            return {"city": c, "error": str(e)}
    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(one, CITIES))


@app.get("/api/model")
def model():
    r = json.loads((REPORTS_DIR / "metrics.json").read_text())
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


# Built frontend (npm run build in web/). Registered last so /api routes win.
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = WEB_DIST / path
        return FileResponse(f if path and f.is_file() else WEB_DIST / "index.html")
