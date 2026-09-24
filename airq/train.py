"""End-to-end training: data -> preprocessing -> feature selection -> models -> evaluation.

    python -m airq.train            # use cached data
    python -m airq.train --refresh  # re-download from Open-Meteo
"""
import argparse
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

from . import features as F
from .aqi import AQI_BANDS, CATEGORIES, CATEGORY_COLORS
from .classify import compare as compare_classifier
from .config import CITIES, DATA_DIR, FIGURES_DIR, MODELS_DIR, N_JOBS, POLLUTANTS, REPORTS_DIR, SEED
from .data import load_dataset
from .evaluate import category_confusion, metrics
from .models import (LSTMForecaster, Persistence, contributions, make_quantile_xgb, make_rf, make_ridge, make_xgb,
                     sequence_index)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_features(train, val, cols):
    """Rank features by permutation importance of a Random Forest on the validation set.

    Keeps features whose shuffling increases validation MAE by at least 0.05 AQI points.
    Today's AQI is always kept: the tree models and the LSTM predict the change from it.
    """
    rf = make_rf(n_estimators=300).fit(train[cols], train["target_aqi"])
    imp = permutation_importance(rf, val[cols], val["target_aqi"], scoring="neg_mean_absolute_error",
                                 n_repeats=5, random_state=SEED, n_jobs=N_JOBS)
    ranking = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)
    selected = ranking[ranking >= 0.05].index.tolist()
    if "aqi" not in selected:
        selected.insert(0, "aqi")
    return selected, ranking


def tune_rf(train, val, cols):
    best = None
    for leaf in (1, 3, 5, 10):
        for mf in (0.3, 0.5, 0.8):
            m = make_rf(min_samples_leaf=leaf, max_features=mf).fit(train[cols], train["target_aqi"])
            mae = mean_absolute_error(val["target_aqi"], m.predict(val[cols]))
            if best is None or mae < best[0]:
                best = (mae, {"min_samples_leaf": leaf, "max_features": mf})
    return best[1]


def tune_xgb(train, val, cols):
    best = None
    for depth in (3, 5, 7):
        for lr in (0.02, 0.05):
            m = make_xgb(max_depth=depth, learning_rate=lr)
            m.fit(train[cols], train["target_aqi"], eval_set=[(val[cols], val["target_aqi"])], verbose=False)
            mae = mean_absolute_error(val["target_aqi"], m.predict(val[cols]))
            if best is None or mae < best[0]:
                best = (mae, {"max_depth": depth, "learning_rate": lr})
    return best[1]


# ---------------------------------------------------------------- figures

def save_eda_figures(daily_aqi: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    order = daily_aqi.groupby("city")["aqi"].median().sort_values().index

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(data=daily_aqi, x="city", y="aqi", order=order, ax=ax, color="#9ecae1", fliersize=1.5)
    for lo, hi, c in zip(AQI_BANDS[:-1], AQI_BANDS[1:], CATEGORY_COLORS):
        ax.axhspan(lo, hi, color=c, alpha=0.08, lw=0)
    ax.set(title="Daily AQI distribution by city (Aug 2022 onward)", xlabel="", ylabel="AQI")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "aqi_by_city.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for city in ("Delhi", "Lucknow", "Kolkata", "Chennai", "Bengaluru"):
        s = daily_aqi[daily_aqi["city"] == city].set_index("date")["aqi"].rolling(7, min_periods=3).mean()
        ax.plot(s.index, s.values, lw=1.2, label=city)
    ax.set(title="AQI over time (7-day rolling mean)", ylabel="AQI")
    ax.legend(ncol=5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "aqi_timeseries.png", dpi=130)
    plt.close(fig)

    cols = POLLUTANTS + ["temp", "humidity", "wind_speed", "precip", "pressure", "blh", "aqi"]
    corr = daily_aqi[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax, square=True)
    ax.set_title("Spearman correlation: pollutants, weather and AQI")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation.png", dpi=130)
    plt.close(fig)

    dom = daily_aqi.dropna(subset=["dominant"]).groupby("city")["dominant"].value_counts(normalize=True).unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    (dom.loc[order] * 100).plot.bar(stacked=True, ax=ax, colormap="tab10", width=0.75)
    ax.set(title="Which pollutant drives the AQI (share of days)", ylabel="% of days", xlabel="")
    ax.legend(title="dominant", bbox_to_anchor=(1.01, 1), frameon=False)
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dominant_pollutant.png", dpi=130)
    plt.close(fig)


def save_result_figures(test, preds, ranking, best_name):
    top = ranking.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color="#3182bd")
    ax.set(title="Permutation importance (validation MAE increase)", xlabel="AQI points")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=130)
    plt.close(fig)

    for city in ("Delhi", "Chennai"):
        m = (test["city"] == city).to_numpy()
        d = test.loc[m, "target_date"]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(d, test.loc[m, "target_aqi"], color="black", lw=1.4, label="Actual")
        ax.plot(d, preds["Persistence (baseline)"][m], color="#bbbbbb", lw=1, label="Persistence")
        ax.plot(d, preds[best_name][m], color="#e6550d", lw=1.2, label=best_name)
        ax.set(title=f"{city}: next-day AQI on the held-out test period", ylabel="AQI")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"test_{city.lower()}.png", dpi=130)
        plt.close(fig)

    cm = category_confusion(test["target_aqi"], preds[best_name])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax, cbar=False)
    ax.set(title=f"AQI category confusion matrix ({best_name}, test)", xlabel="Predicted", ylabel="Actual")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main

def save_extra_figures(summary: dict, shap_mean: pd.Series):
    fig, ax = plt.subplots(figsize=(7, 4))
    hs = sorted(summary)
    for name in summary[hs[0]]["test_mae"]:
        ax.plot(hs, [summary[h]["test_mae"][name] for h in hs], marker="o",
                lw=2.2 if name.startswith("Persistence") else 1.4,
                color="#999999" if name.startswith("Persistence") else None, label=name)
    ax.set(title="Forecast error grows with lead time (test MAE)", xlabel="Days ahead", ylabel="MAE (AQI points)",
           xticks=hs)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_by_horizon.png", dpi=130)
    plt.close(fig)

    top = shap_mean.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.index, top.values, color="#6a51a3")
    ax.set(title="SHAP: mean |contribution| to next-day forecast (XGBoost, test)", xlabel="AQI points")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_importance.png", dpi=130)
    plt.close(fig)


def train_horizon(feats: pd.DataFrame, h: int, skip_lstm: bool):
    """Select features, tune, train and evaluate every model for one lead time (days ahead)."""
    all_cols = F.feature_columns(feats)
    train, val, test = F.split(feats)
    log(f"  [{h}d] {len(all_cols)} features | train {len(train)} / val {len(val)} / test {len(test)}")

    selected, ranking = select_features(train, val, all_cols)
    rf_params = tune_rf(train, val, selected)
    xgb_params = tune_xgb(train, val, selected)
    log(f"  [{h}d] kept {len(selected)} features | RF {rf_params} | XGB {xgb_params}")

    y_tr = train["target_aqi"]
    models = {
        "Persistence (baseline)": Persistence(),
        "Linear Regression": make_ridge().fit(train[selected], y_tr),
        "Random Forest": make_rf(**rf_params).fit(train[selected], y_tr),
    }
    xgb = make_xgb(**xgb_params)
    xgb.fit(train[selected], y_tr, eval_set=[(val[selected], val["target_aqi"])], verbose=False)
    models["XGBoost"] = xgb

    def predict(name, frame):
        m = models[name]
        return m.predict(frame) if name.startswith("Persistence") else m.predict(frame[selected])

    val_preds = {n: predict(n, val) for n in models}
    test_preds = {n: predict(n, test) for n in models}

    lstm_info = None
    if not skip_lstm:
        seq_rows = sequence_index(feats)
        tr_rows = np.intersect1d(seq_rows, train.index.to_numpy())
        va_rows = np.intersect1d(seq_rows, val.index.to_numpy())
        te_rows = np.intersect1d(seq_rows, test.index.to_numpy())
        lstm = LSTMForecaster(selected).fit(feats, tr_rows, va_rows)
        models["LSTM"] = lstm
        # LSTM needs 14 consecutive days of history; score it on the rows where that exists
        # and fall back to persistence on the few that lack it, so every model is scored on the same rows.
        vp = val["aqi"].to_numpy().astype(float).copy()
        vp[np.searchsorted(val.index.to_numpy(), va_rows)] = lstm.predict(feats, va_rows)
        tp = test["aqi"].to_numpy().astype(float).copy()
        tp[np.searchsorted(test.index.to_numpy(), te_rows)] = lstm.predict(feats, te_rows)
        val_preds["LSTM"], test_preds["LSTM"] = vp, tp
        lstm_info = {"device": lstm.device, "best_epoch": lstm.best_epoch,
                     "test_rows_with_full_history": int(len(te_rows)), "test_rows": int(len(test))}

    # 80% prediction interval from quantile XGBoost.
    qxgb = make_quantile_xgb(max_depth=xgb_params["max_depth"], learning_rate=xgb_params["learning_rate"])
    qxgb.fit(train[selected], y_tr, eval_set=[(val[selected], val["target_aqi"])], verbose=False)
    # Conformalised quantile regression: widen the band by the validation-set error margin so
    # that it really covers ~80% of outcomes (raw quantile models are usually over-confident).
    qv, yv = qxgb.predict(val[selected]), val["target_aqi"].to_numpy()
    scores = np.maximum(qv[:, 0] - yv, yv - qv[:, 2])
    margin = float(np.quantile(scores, min(1.0, 0.8 * (1 + 1 / len(yv)))))
    q = qxgb.predict(test[selected])
    y = test["target_aqi"].to_numpy()
    lo, hi = q[:, 0] - margin, q[:, 2] + margin
    interval = {"coverage_80_raw": round(float(np.mean((y >= q[:, 0]) & (y <= q[:, 2]))), 4),
                "coverage_80": round(float(np.mean((y >= lo) & (y <= hi))), 4),
                "mean_width": round(float(np.mean(hi - lo)), 1), "conformal_margin": round(margin, 2)}

    results = {n: {"validation": metrics(val["target_aqi"], val_preds[n]),
                   "test": metrics(test["target_aqi"], test_preds[n])} for n in models}
    # Choose the deployed model on validation MAE only; the test set stays untouched until reporting.
    candidates = [n for n in models if not n.startswith("Persistence")]
    best_name = min(candidates, key=lambda n: results[n]["validation"]["MAE"])
    table = pd.DataFrame({n: r["test"] for n, r in results.items()}).T
    log(f"  [{h}d] best on validation: {best_name} | 80% interval coverage {interval['coverage_80']:.0%}, "
        f"width {interval['mean_width']}\n" + table[["MAE", "RMSE", "R2", "category_accuracy", "alert_recall"]].to_string())

    per_city = {}
    for city in CITIES:
        m = (test["city"] == city).to_numpy()
        per_city[city] = {n: metrics(test.loc[m, "target_aqi"], test_preds[n][m])["MAE"] for n in models}

    report = {
        "split": {"train": [f"{train['target_date'].min():%Y-%m-%d}", f"{train['target_date'].max():%Y-%m-%d}", len(train)],
                  "validation": [f"{val['target_date'].min():%Y-%m-%d}", f"{val['target_date'].max():%Y-%m-%d}", len(val)],
                  "test": [f"{test['target_date'].min():%Y-%m-%d}", f"{test['target_date'].max():%Y-%m-%d}", len(test)]},
        "features": {"all": len(all_cols), "selected": selected,
                     "importance_top20": ranking.head(20).round(3).to_dict()},
        "hyperparameters": {"random_forest": rf_params, "xgboost": {**xgb_params, "best_iteration": int(xgb.best_iteration)}},
        "lstm": lstm_info,
        "results": results,
        "test_mae_by_city": per_city,
        "best_model": best_name,
        "interval": interval,
    }
    hbundle = {"selected": selected, "best_model": best_name, "quantile": qxgb, "conformal_margin": margin,
               "sklearn": {n: m for n, m in models.items() if n != "LSTM" and not n.startswith("Persistence")}}
    if "LSTM" in models:
        hbundle["lstm"] = models["LSTM"].state()
    return report, hbundle, (test, test_preds, ranking, table)


def main(refresh: bool = False, skip_lstm: bool = False):
    for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log("1/4 data collection")
    daily = load_dataset(refresh=refresh)
    daily.to_csv(DATA_DIR / "daily.csv", index=False)
    log(f"    {len(daily)} city-days, {daily['city'].nunique()} cities, "
        f"{daily['date'].min():%Y-%m-%d} .. {daily['date'].max():%Y-%m-%d}")

    log("2/4 preprocessing + exploratory figures")
    save_eda_figures(F.add_aqi(F.clean(daily)))

    log(f"3/4 training for horizons {F.HORIZONS} (feature selection, tuning, 5 models each)")
    reports, bundles, fences, prep_report = {}, {}, None, None
    for h in F.HORIZONS:
        feats, fences, prep = F.prepare(daily, horizon=h)
        prep_report = prep_report or prep
        feats = feats.sort_values(["city", "date"]).reset_index(drop=True)
        reports[h], bundles[h], out = train_horizon(feats, h, skip_lstm)
        if h == 1:
            test, test_preds, ranking, table = out
            save_result_figures(test, test_preds, ranking, reports[1]["best_model"])
            table.to_csv(REPORTS_DIR / "model_comparison_test.csv")
            pd.DataFrame({"date": test["target_date"], "city": test["city"], "actual": test["target_aqi"],
                          **{n: test_preds[n] for n in test_preds}}).to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
            xgb1, sel1 = bundles[1]["sklearn"]["XGBoost"], bundles[1]["selected"]
            tr1, va1, _ = F.split(feats)
            _, classification = compare_classifier(tr1, va1, test, sel1, test_preds[reports[1]["best_model"]])
            log(f"  [1d] category macro-F1: regression {classification['from_regression']['macro_f1']} "
                f"vs balanced classifier {classification['balanced_classifier']['macro_f1']}")
            shap_mean = contributions(xgb1, test[sel1]).drop(columns="bias").abs().mean().sort_values(ascending=False)

    log("4/4 reports")
    summary = {h: {"best_model": r["best_model"], "interval": r["interval"],
                   "test_mae": {n: v["test"]["MAE"] for n, v in r["results"].items()}} for h, r in reports.items()}
    save_extra_figures(summary, shap_mean)
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "data": {"source": "Open-Meteo (CAMS air quality + ERA5 weather)", "cities": list(CITIES),
                 "start": f"{daily['date'].min():%Y-%m-%d}", "end": f"{daily['date'].max():%Y-%m-%d}",
                 "city_days": len(daily)},
        "preprocessing": prep_report,
        **reports[1],  # headline numbers are for the next-day forecast
        "shap_top15": shap_mean.head(15).round(2).to_dict(),
        "classification": classification,
        "horizons": summary,
        "horizon_reports": reports,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    joblib.dump({"fences": fences, "horizons": bundles}, MODELS_DIR / "aqi_models.joblib")
    log("saved models/aqi_models.joblib and reports/ | " +
        ", ".join(f"{h}d: {s['best_model']} MAE {s['test_mae'][s['best_model']]}" for h, s in summary.items()))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download data from Open-Meteo")
    ap.add_argument("--skip-lstm", action="store_true")
    a = ap.parse_args()
    main(refresh=a.refresh, skip_lstm=a.skip_lstm)
