"""Indian National Air Quality Index (CPCB, 2014).

Each pollutant's concentration is mapped to a sub-index by linear interpolation
between breakpoints. The AQI is the worst (maximum) sub-index, and is only valid
when at least three pollutants are available, one of which is PM2.5 or PM10.

Averaging periods: PM2.5, PM10, NO2, SO2 use 24-hour means; CO and O3 use the
maximum 8-hour mean. CO is in mg/m3, everything else in ug/m3.
"""
import math

import numpy as np
import pandas as pd

AQI_BANDS = [0, 50, 100, 200, 300, 400, 500]

# Concentration breakpoints matching AQI_BANDS. The last band is open-ended in the
# standard; its upper edge here follows the common CPCB calculator convention.
BREAKPOINTS = {
    "pm2_5": [0, 30, 60, 90, 120, 250, 380],
    "pm10":  [0, 50, 100, 250, 350, 430, 510],
    "no2":   [0, 40, 80, 180, 280, 400, 520],
    "so2":   [0, 40, 80, 380, 800, 1600, 2400],
    "co":    [0, 1.0, 2.0, 10, 17, 34, 51],
    "o3":    [0, 50, 100, 168, 208, 748, 1288],
}

CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
CATEGORY_COLORS = ["#009966", "#8bc34a", "#ffd500", "#ff8c00", "#e53935", "#7e0023"]
HEALTH_ADVICE = [
    "Minimal impact.",
    "Minor breathing discomfort to sensitive people.",
    "Breathing discomfort to people with lung or heart disease, children and older adults.",
    "Breathing discomfort to most people on prolonged exposure.",
    "Respiratory illness on prolonged exposure. Avoid outdoor activity.",
    "Affects healthy people and seriously impacts those with existing diseases. Stay indoors.",
]


def sub_index(pollutant: str, conc: float) -> float:
    """Sub-index for one pollutant concentration; NaN if the value is missing."""
    if conc is None or (isinstance(conc, float) and math.isnan(conc)) or conc < 0:
        return float("nan")
    bp = BREAKPOINTS[pollutant]
    if conc >= bp[-1]:
        return 500.0
    for i in range(1, len(bp)):
        if conc <= bp[i]:
            lo_c, hi_c = bp[i - 1], bp[i]
            lo_i, hi_i = AQI_BANDS[i - 1], AQI_BANDS[i]
            return lo_i + (conc - lo_c) * (hi_i - lo_i) / (hi_c - lo_c)
    return 500.0  # unreachable


def compute_aqi(concs: dict) -> float:
    """AQI from a {pollutant: concentration} mapping, applying the CPCB validity rule."""
    subs = {p: sub_index(p, concs.get(p)) for p in BREAKPOINTS}
    valid = {p: v for p, v in subs.items() if not math.isnan(v)}
    if len(valid) < 3 or ("pm2_5" not in valid and "pm10" not in valid):
        return float("nan")
    return round(max(valid.values()), 1)


def aqi_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised sub-indices and AQI for a frame with one column per pollutant."""
    out = pd.DataFrame(index=df.index)
    for p, bp in BREAKPOINTS.items():
        c = df[p].to_numpy(dtype=float)
        idx = np.interp(np.clip(c, 0, bp[-1]), bp, AQI_BANDS)
        idx[np.isnan(c) | (c < 0)] = np.nan
        out[f"si_{p}"] = idx
    si = out[[f"si_{p}" for p in BREAKPOINTS]]
    n_valid = si.notna().sum(axis=1)
    has_pm = out["si_pm2_5"].notna() | out["si_pm10"].notna()
    out["aqi"] = si.max(axis=1).where((n_valid >= 3) & has_pm).round(1)
    out["dominant"] = si.idxmax(axis=1).str.removeprefix("si_").where(out["aqi"].notna())
    return out


def category_index(aqi: float) -> int:
    """0..5 index into CATEGORIES."""
    if aqi is None or math.isnan(aqi):
        return -1
    for i, upper in enumerate(AQI_BANDS[1:]):
        if aqi <= upper:
            return i
    return len(CATEGORIES) - 1


def categorize(aqi) -> np.ndarray:
    """Vectorised category index (0..5) for an array of AQI values."""
    a = np.asarray(aqi, dtype=float)
    return np.clip(np.searchsorted(AQI_BANDS[1:-1], a, side="left"), 0, 5)


def category(aqi: float) -> str:
    i = category_index(aqi)
    return CATEGORIES[i] if i >= 0 else "Unknown"
