"""Direct AQI-category classification vs categories derived from the regression forecast.

Severe and Good days are rare, so the regression models (which minimise average error)
tend to pull forecasts toward the middle categories. A classifier trained with balanced
class weights trades some overall accuracy for better recall on the rare, extreme days.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .aqi import CATEGORIES, categorize
from .config import N_JOBS, SEED

LABELS = list(range(len(CATEGORIES)))


def category_scores(y_true_cat, y_pred_cat) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true_cat, y_pred_cat), 4),
        "macro_f1": round(f1_score(y_true_cat, y_pred_cat, average="macro", labels=LABELS, zero_division=0), 4),
        "recall_by_category": dict(zip(CATEGORIES, np.round(
            recall_score(y_true_cat, y_pred_cat, average=None, labels=LABELS, zero_division=0), 3).tolist())),
    }


def train_classifier(train: pd.DataFrame, val: pd.DataFrame, cols: list[str]) -> XGBClassifier:
    y_tr, y_va = categorize(train["target_aqi"]), categorize(val["target_aqi"])
    clf = XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(CATEGORIES), early_stopping_rounds=100,
        eval_metric="mlogloss", random_state=SEED, n_jobs=N_JOBS,
    )
    clf.fit(train[cols], y_tr, sample_weight=compute_sample_weight("balanced", y_tr),
            eval_set=[(val[cols], y_va)], sample_weight_eval_set=[compute_sample_weight("balanced", y_va)],
            verbose=False)
    return clf


def compare(train, val, test, cols, regression_pred: np.ndarray) -> tuple[XGBClassifier, dict]:
    """Scores on the test set: categories from the regression forecast vs the balanced classifier."""
    clf = train_classifier(train, val, cols)
    y_te = categorize(test["target_aqi"])
    y_tr = categorize(train["target_aqi"])
    return clf, {
        "train_days_by_category": dict(zip(CATEGORIES, np.bincount(y_tr, minlength=len(CATEGORIES)).tolist())),
        "from_regression": category_scores(y_te, categorize(regression_pred)),
        "balanced_classifier": category_scores(y_te, clf.predict(test[cols])),
        "test_days_by_category": dict(zip(CATEGORIES, np.bincount(y_te, minlength=len(CATEGORIES)).tolist())),
    }
