import math

import numpy as np
import pandas as pd
import pytest

from airq.aqi import aqi_frame, categorize, category, compute_aqi, sub_index


@pytest.mark.parametrize("pollutant, conc, expected", [
    ("pm2_5", 0, 0), ("pm2_5", 30, 50), ("pm2_5", 45, 75), ("pm2_5", 60, 100),
    ("pm2_5", 90, 200), ("pm2_5", 120, 300), ("pm2_5", 250, 400), ("pm2_5", 1000, 500),
    ("pm10", 100, 100), ("pm10", 250, 200), ("no2", 80, 100), ("so2", 380, 200),
    ("co", 2.0, 100), ("co", 10, 200), ("o3", 168, 200),
])
def test_sub_index_breakpoints(pollutant, conc, expected):
    assert sub_index(pollutant, conc) == pytest.approx(expected)


def test_aqi_is_worst_sub_index():
    concs = {"pm2_5": 45, "pm10": 60, "no2": 20, "so2": 10, "co": 0.5, "o3": 30}
    assert compute_aqi(concs) == pytest.approx(75.0)


def test_aqi_requires_three_pollutants_including_pm():
    assert math.isnan(compute_aqi({"pm2_5": 45, "no2": 20}))
    assert math.isnan(compute_aqi({"no2": 20, "so2": 10, "co": 0.5, "o3": 30}))
    assert compute_aqi({"pm10": 80, "no2": 20, "so2": 10}) == pytest.approx(80.0)


def test_vectorised_matches_scalar():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "pm2_5": rng.uniform(0, 300, 200), "pm10": rng.uniform(0, 500, 200),
        "no2": rng.uniform(0, 200, 200), "so2": rng.uniform(0, 100, 200),
        "co": rng.uniform(0, 5, 200), "o3": rng.uniform(0, 200, 200),
    })
    df.loc[::7, "pm2_5"] = np.nan
    vec = aqi_frame(df)["aqi"].to_numpy()
    scal = np.array([compute_aqi(r) for r in df.to_dict("records")])
    np.testing.assert_allclose(vec, scal, atol=0.06)


def test_categories_at_band_edges():
    assert category(50) == "Good"
    assert category(51) == "Satisfactory"
    assert category(200) == "Moderate"
    assert category(201) == "Poor"
    assert category(450) == "Severe"
    np.testing.assert_array_equal(categorize([0, 50, 50.5, 100, 150, 250, 350, 499]), [0, 0, 1, 1, 2, 3, 4, 5])
