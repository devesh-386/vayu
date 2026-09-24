"""Regression, category and early-warning metrics."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)

from .aqi import CATEGORIES, categorize

ALERT_AQI = 200  # alert when AQI is forecast to be Poor (201+) or worse


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ct, cp = categorize(y_true), categorize(y_pred)
    at, ap = y_true > ALERT_AQI, y_pred > ALERT_AQI
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
        "category_accuracy": round(accuracy_score(ct, cp), 4),
        "category_within_one": round(float(np.mean(np.abs(ct - cp) <= 1)), 4),
        "category_macro_f1": round(f1_score(ct, cp, average="macro", labels=range(len(CATEGORIES)), zero_division=0), 4),
        "alert_precision": round(precision_score(at, ap, zero_division=0), 4),
        "alert_recall": round(recall_score(at, ap, zero_division=0), 4),
        "alert_days": int(at.sum()),
    }


def category_confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(categorize(y_true), categorize(y_pred), labels=range(len(CATEGORIES)))
