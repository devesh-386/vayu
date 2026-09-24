import numpy as np
import pandas as pd

from airq import features as F
from airq.config import TRAIN_END, VAL_END


def synthetic_daily(n=60, cities=("Chennai", "Delhi")):
    rng = np.random.default_rng(1)
    frames = []
    for c in cities:
        d = pd.DataFrame({"date": pd.date_range("2024-12-01", periods=n, freq="D"), "city": c})
        for p, scale in [("pm2_5", 60), ("pm10", 100), ("no2", 30), ("so2", 10), ("co", 1), ("o3", 60)]:
            d[p] = rng.uniform(0.2, 1.8, n) * scale
        for w in ("temp", "temp_max", "temp_min", "humidity", "wind_speed", "wind_u", "wind_v", "precip", "pressure", "blh"):
            d[w] = rng.normal(size=n)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def test_target_is_next_day_aqi_of_same_city():
    daily = F.add_aqi(F.clean(synthetic_daily()))
    feats = F.build_features(daily)
    for city, g in feats.groupby("city"):
        g = g.sort_values("date")
        np.testing.assert_allclose(g["target_aqi"].iloc[:-1].to_numpy(), g["aqi"].iloc[1:].to_numpy())
        assert np.isnan(g["target_aqi"].iloc[-1])


def test_no_same_day_target_information_leaks_into_features():
    """Changing tomorrow's pollutants must not change today's features."""
    daily = F.add_aqi(F.clean(synthetic_daily()))
    base = F.build_features(daily)
    tampered = daily.copy()
    last = tampered["date"].max()
    tampered.loc[tampered["date"] == last, ["pm2_5", "pm10", "no2", "so2", "co", "o3", "aqi"]] *= 10
    changed = F.build_features(tampered)
    cols = F.feature_columns(base)
    day_before = last - pd.Timedelta(days=1)
    a = base[base["date"] == day_before][cols].to_numpy(dtype=float)
    b = changed[changed["date"] == day_before][cols].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b)


def test_short_gaps_filled_long_gaps_kept():
    d = synthetic_daily(n=30, cities=("Chennai",))
    d.loc[5:6, "pm2_5"] = np.nan      # 2-day gap -> interpolated
    d.loc[12:17, "pm2_5"] = np.nan    # 6-day gap -> only edges filled up to the limit
    out = F.clean(d)
    assert out.loc[5:6, "pm2_5"].notna().all()
    assert out.loc[12:17, "pm2_5"].isna().any()


def test_missing_dates_are_reindexed():
    d = synthetic_daily(n=20, cities=("Chennai",)).drop(index=[4, 5])
    out = F.clean(d)
    assert len(out) == 20


def test_split_is_chronological():
    feats = pd.DataFrame({"target_date": pd.date_range("2024-06-01", "2025-12-31", freq="D")})
    tr, va, te = F.split(feats)
    assert tr["target_date"].max() <= pd.Timestamp(TRAIN_END) < va["target_date"].min()
    assert va["target_date"].max() <= pd.Timestamp(VAL_END) < te["target_date"].min()
    assert len(tr) + len(va) + len(te) == len(feats)


def test_outlier_fences_fit_on_train_only():
    d = F.add_aqi(F.clean(synthetic_daily(n=90)))
    d.loc[d["date"] > TRAIN_END, "pm2_5"] = 10_000  # absurd values after the train period
    fences = F.outlier_fences(d)
    assert fences[("Chennai", "pm2_5")][1] < 10_000
    clipped, n = F.clip_outliers(d, fences)
    assert n > 0 and clipped["pm2_5"].max() < 10_000


def test_long_weather_gap_filled_with_training_climatology():
    d = synthetic_daily(n=40, cities=("Chennai",))
    d.loc[10:30, "blh"] = np.nan  # 21-day gap, too long to interpolate
    out = F.clean(d)
    assert out["blh"].notna().all()
    assert out["pm2_5"].notna().all()
