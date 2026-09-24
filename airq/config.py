"""Project-wide settings: cities, date ranges, paths and the train/val/test split."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TIMEZONE = "Asia/Kolkata"

# Open-Meteo's CAMS pollutant history starts on 2022-08-04; the first full day is the 5th.
START_DATE = "2022-08-05"

# Major Indian cities with distinct pollution regimes (Indo-Gangetic plain, coastal, plateau).
CITIES = {
    "Chennai":   (13.0827, 80.2707),
    "Delhi":     (28.6139, 77.2090),
    "Mumbai":    (19.0760, 72.8777),
    "Kolkata":   (22.5726, 88.3639),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Lucknow":   (26.8467, 80.9462),
}

# Chronological split: the model never sees the future during training.
TRAIN_END = "2024-12-31"
VAL_END = "2025-06-30"
# everything after VAL_END is the held-out test set

POLLUTANTS = ["pm2_5", "pm10", "no2", "so2", "co", "o3"]
WEATHER = ["temp", "humidity", "wind_speed", "wind_u", "wind_v", "precip", "pressure", "blh"]

SEED = 42

# Parallel workers for sklearn/xgboost. Each worker copies the forest, so keep this modest.
N_JOBS = 4
