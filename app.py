"""Streamlit dashboard: live AQI, 1-3 day forecast with uncertainty and explanations, alerts, model performance.

    streamlit run app.py
"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from airq import alerts
from airq.aqi import AQI_BANDS, BREAKPOINTS, CATEGORIES, CATEGORY_COLORS, category_index
from airq.config import CITIES, FIGURES_DIR, REPORTS_DIR
from airq.predict import explain, forecast, horizons, load_bundle, model_names, recent_frame

st.set_page_config(page_title="Air Quality Prediction", page_icon="🌫️", layout="wide")

POLLUTANT_LABELS = {"pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO₂", "so2": "SO₂", "co": "CO", "o3": "O₃"}
UNITS = {"pm2_5": "µg/m³", "pm10": "µg/m³", "no2": "µg/m³", "so2": "µg/m³", "co": "mg/m³", "o3": "µg/m³"}


@st.cache_resource
def bundle():
    return load_bundle()


@st.cache_data(ttl=1800, show_spinner=False)
def recent(city: str) -> pd.DataFrame:
    return recent_frame(city)


@st.cache_data
def report() -> dict:
    return json.loads((REPORTS_DIR / "metrics.json").read_text())


def color_for(aqi: float) -> str:
    return CATEGORY_COLORS[max(category_index(aqi), 0)]


def aqi_card(label: str, aqi: float, cat: str, note: str = ""):
    c = color_for(aqi)
    text = "#1a1a1a" if c in ("#ffd500", "#8bc34a") else "#ffffff"
    st.markdown(
        f"""<div style="background:{c};color:{text};border-radius:14px;padding:18px 20px;min-height:150px">
        <div style="font-size:0.85rem;opacity:.85;text-transform:uppercase;letter-spacing:.04em">{label}</div>
        <div style="font-size:2.9rem;font-weight:700;line-height:1.1;margin-top:4px">{aqi:.0f}</div>
        <div style="font-size:1.15rem;font-weight:600">{cat}</div>
        <div style="font-size:.85rem;opacity:.85;margin-top:4px">{note}</div></div>""",
        unsafe_allow_html=True,
    )


def history_chart(daily: pd.DataFrame, outlook: list[dict]) -> go.Figure:
    today = pd.Timestamp(outlook[0]["date"])
    past = daily[daily["date"] <= today].tail(30)
    fig = go.Figure()
    for lo, hi, c, name in zip(AQI_BANDS[:-1], AQI_BANDS[1:], CATEGORY_COLORS, CATEGORIES):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=c, opacity=0.10, line_width=0,
                      annotation_text=name, annotation_position="right", annotation_font_size=10)
    fig.add_trace(go.Scatter(x=past["date"], y=past["aqi"], mode="lines+markers", name="Observed AQI",
                             line=dict(color="#3182bd", width=2), marker=dict(size=5)))
    xs = [today] + [pd.Timestamp(r["target_date"]) for r in outlook]
    last = past["aqi"].iloc[-1]
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[last] + [r["high"] for r in outlook] + [r["low"] for r in outlook][::-1] + [last],
                             fill="toself", fillcolor="rgba(230,85,13,0.18)", line=dict(width=0), mode="lines",
                             name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[last] + [r["predicted_aqi"] for r in outlook], mode="lines+markers",
                             name="Forecast", line=dict(color="#e6550d", dash="dash", width=2),
                             marker=dict(size=[0] + [11] * len(outlook), symbol="diamond")))
    ymax = max(past["aqi"].max(), max(r["high"] for r in outlook)) * 1.15
    fig.update_layout(height=380, margin=dict(l=10, r=90, t=10, b=10), yaxis=dict(title="AQI", range=[0, max(ymax, 120)]),
                      legend=dict(orientation="h", y=1.08), hovermode="x unified")
    return fig


def explain_chart(why: pd.DataFrame) -> go.Figure:
    w = why.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=w["contribution"], y=w["factor"], orientation="h",
        marker_color=["#e53935" if v > 0 else "#2e7d32" for v in w["contribution"]],
        text=[f"{v:+.0f}" for v in w["contribution"]], textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(height=330, margin=dict(l=10, r=40, t=10, b=10), xaxis_title="AQI points vs today")
    return fig


def subindex_chart(row: pd.Series) -> go.Figure:
    names = list(BREAKPOINTS)
    vals = [row[f"si_{p}"] for p in names]
    fig = go.Figure(go.Bar(
        x=vals, y=[POLLUTANT_LABELS[p] for p in names], orientation="h",
        marker_color=[color_for(v) for v in vals],
        text=[f"{v:.0f}  ({row[p]:.1f} {UNITS[p]})" for p, v in zip(names, vals)], textposition="auto",
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Sub-index (AQI scale)",
                      yaxis=dict(autorange="reversed"))
    return fig


# ---------------------------------------------------------------- layout

b = bundle()
rep = report()
HS = horizons(b)

with st.sidebar:
    st.header("Settings")
    city = st.selectbox("City", list(CITIES), index=0)
    names = model_names(b)
    best1 = b["horizons"][1]["best_model"]
    choice = st.selectbox("Model", ["Best per horizon"] + names, index=0,
                          help=f"'Best per horizon' uses the model with the lowest validation error for each "
                               f"lead time (next day: {best1}).")
    st.divider()
    st.caption("Data: Open-Meteo (CAMS air quality, ERA5 / forecast weather). "
               "AQI: CPCB National AQI. Refreshes every 30 minutes.")

st.title("Intelligent Air Quality Prediction System")
st.caption("1–3 day AQI forecasting from pollutant and weather data using Machine Learning")

tab_live, tab_models, tab_data, tab_how = st.tabs(["Live forecast", "Model performance", "Data analysis", "How it works"])

with tab_live:
    try:
        with st.spinner(f"Fetching latest data for {city}..."):
            daily = recent(city)
            model_for = (lambda h: None) if choice == "Best per horizon" else (lambda h: choice)
            outlook = [forecast(city, model_for(h), b, daily, h) for h in HS]
    except Exception as e:  # network failure or incomplete data
        st.error(f"Could not produce a forecast for {city}: {e}")
        st.stop()

    res = outlook[0]
    today_row = daily[daily["date"] == pd.Timestamp(res["date"])].iloc[0]

    worst = max(outlook, key=lambda r: r["predicted_aqi"])
    maybe = [r for r in outlook if r["possible_alert"]]
    if worst["alert"]:
        st.error(f"⚠️ **Early warning:** air quality in {city} is forecast to be **{worst['predicted_category']}** "
                 f"(AQI {worst['predicted_aqi']:.0f}) on {worst['target_date']}. {worst['advice']}")
    elif maybe:
        st.warning(f"Watch: the forecast range for {city} on {maybe[0]['target_date']} reaches "
                   f"AQI {maybe[0]['high']:.0f} (Poor). Most likely value is {maybe[0]['predicted_aqi']:.0f}.")
    else:
        st.success(f"No alert: no Poor-or-worse day expected in {city} over the next {len(HS)} days "
                   f"(worst: {worst['predicted_category']}, AQI {worst['predicted_aqi']:.0f}).")

    cols = st.columns(len(HS) + 1)
    with cols[0]:
        aqi_card(f"Today · {res['date']}", res["today_aqi"], res["today_category"],
                 f"Dominant: {POLLUTANT_LABELS.get(res['today_dominant'], res['today_dominant'])}")
    for col, r in zip(cols[1:], outlook):
        with col:
            label = "Tomorrow" if r["horizon"] == 1 else f"+{r['horizon']} days"
            aqi_card(f"{label} · {r['target_date']}", r["predicted_aqi"], r["predicted_category"],
                     f"80% range {r['low']:.0f}–{r['high']:.0f} · {r['model']}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Last 30 days and the 3-day outlook")
        st.plotly_chart(history_chart(daily, outlook), width="stretch")
    with right:
        st.subheader("Why this forecast?")
        try:
            why = explain(city, b, daily)
            st.plotly_chart(explain_chart(why), width="stretch")
            st.caption("SHAP contributions (XGBoost, next day): how much each factor pushes tomorrow's AQI "
                       "up (red) or down (green) compared with today.")
        except Exception as e:
            st.caption(f"Explanation unavailable: {e}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Today's pollutant sub-indices")
        st.plotly_chart(subindex_chart(today_row), width="stretch")
        st.caption("The AQI is the highest sub-index (CPCB method).")
    with right:
        st.subheader("Health advice")
        st.write(res["advice"])
        st.markdown("**Weather**")
        w1, w2 = st.columns(2)
        w1.metric("Temp today", f"{today_row['temp']:.1f} °C")
        w2.metric("Humidity", f"{today_row['humidity']:.0f} %")
        tomorrow_row = daily[daily["date"] == pd.Timestamp(res["target_date"])]
        if len(tomorrow_row):
            t = tomorrow_row.iloc[0]
            w1.metric("Wind tomorrow", f"{t['wind_speed']:.0f} km/h")
            w2.metric("Rain tomorrow", f"{t['precip']:.1f} mm")

    with st.expander("All models, all horizons"):
        rows = []
        for n in names:
            row = {"Model": n}
            for h in HS:
                try:
                    r = forecast(city, n, b, daily, h)
                    row[f"+{h} day"] = f"{r['predicted_aqi']:.0f} ({r['predicted_category']})"
                except ValueError as e:
                    row[f"+{h} day"] = str(e)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("Email alert"):
        st.code(alerts.build_message(worst), language=None)
        if alerts.email_configured():
            if st.button("Send this alert by email"):
                st.success("Sent." if alerts.send_email(worst) else "Email not configured.")
        else:
            st.caption("Set AQI_SMTP_HOST, AQI_SMTP_USER, AQI_SMTP_PASSWORD and AQI_ALERT_TO to enable email alerts.")

with tab_models:
    split = rep["split"]
    st.markdown(
        f"Trained on **{split['train'][2]:,}** city-days ({split['train'][0]} → {split['train'][1]}), "
        f"tuned on **{split['validation'][2]:,}** ({split['validation'][0]} → {split['validation'][1]}), "
        f"tested on **{split['test'][2]:,}** unseen city-days ({split['test'][0]} → {split['test'][1]}). "
        "The split is chronological, so every model is scored on days after its training data."
    )
    tbl = pd.DataFrame({n: r["test"] for n, r in rep["results"].items()}).T
    tbl = tbl.rename(columns={"category_accuracy": "Category acc.", "category_within_one": "Within 1 category",
                              "category_macro_f1": "Category F1", "alert_precision": "Alert precision",
                              "alert_recall": "Alert recall", "alert_days": "Alert days"})
    st.subheader("Test-set results")
    best_cell = "background-color:#2e7d32;color:#ffffff;font-weight:600"
    st.dataframe(tbl.style.highlight_min(subset=["MAE", "RMSE"], props=best_cell)
                 .highlight_max(subset=["R2", "Category acc.", "Alert recall"], props=best_cell)
                 .format({c: "{:.3f}" for c in tbl.columns} | {"MAE": "{:.2f}", "RMSE": "{:.2f}", "Alert days": "{:.0f}"}),
                 width="stretch")
    st.caption("Best value in each key column is highlighted. Category = CPCB band (Good … Severe). "
               "Alert = forecast AQI above 200.")
    base = rep["results"]["Persistence (baseline)"]["test"]["MAE"]
    best = rep["best_model"]
    gain = 100 * (1 - rep["results"][best]["test"]["MAE"] / base)
    st.info(f"**{best}** (chosen on validation) cuts the error of the naive 'tomorrow = today' baseline by "
            f"**{gain:.0f}%** (MAE {rep['results'][best]['test']['MAE']} vs {base}). "
            "Alerts fire when forecast AQI exceeds 200 (Poor or worse).")

    st.subheader("Forecasting further ahead")
    hz = rep["horizons"]
    htbl = pd.DataFrame({f"+{h} day": s["test_mae"] for h, s in hz.items()})
    htbl.loc["80% range: actual inside"] = [f"{s['interval']['coverage_80']:.0%}" for s in hz.values()]
    htbl.loc["80% range: avg width"] = [s["interval"]["mean_width"] for s in hz.values()]
    a, bcol = st.columns([2, 3])
    a.dataframe(htbl.astype(str), width="stretch")
    a.caption("Test MAE per lead time. The range comes from quantile XGBoost, calibrated on the validation set "
              "(conformal prediction).")
    bcol.image(str(FIGURES_DIR / "error_by_horizon.png"))

    if "classification" in rep:
        st.subheader("Category forecast: regression vs direct classifier")
        c = rep["classification"]
        ctbl = pd.DataFrame({
            f"From regression ({best})": c["from_regression"]["recall_by_category"],
            "Balanced XGBoost classifier": c["balanced_classifier"]["recall_by_category"],
            "Training days": c["train_days_by_category"],
            "Test days": c["test_days_by_category"],
        })
        st.dataframe(ctbl, width="stretch")
        st.caption(f"Recall per category on the test set. Macro-F1: regression {c['from_regression']['macro_f1']:.2f}, "
                   f"classifier {c['balanced_classifier']['macro_f1']:.2f}. The classifier catches more Good and "
                   "Very Poor days but cannot learn Severe (only "
                   f"{c['train_days_by_category']['Severe']} Severe day in training), so the dashboard derives "
                   "categories from the regression forecast.")

    st.subheader("Mean absolute error by city (test)")
    st.dataframe(pd.DataFrame(rep["test_mae_by_city"]).T.round(1), width="stretch")

    a, bcol = st.columns(2)
    a.image(str(FIGURES_DIR / "test_delhi.png"), caption="Delhi: forecast vs actual (test period)")
    bcol.image(str(FIGURES_DIR / "test_chennai.png"), caption="Chennai: forecast vs actual (test period)")
    a.image(str(FIGURES_DIR / "confusion_matrix.png"), caption="AQI category confusion matrix")
    bcol.image(str(FIGURES_DIR / "feature_importance.png"), caption="Most useful features (permutation importance)")
    a.image(str(FIGURES_DIR / "shap_importance.png"), caption="SHAP: average impact of each feature on the forecast")

with tab_data:
    d = rep["data"]
    st.markdown(f"**{d['city_days']:,}** city-days for **{len(d['cities'])}** cities, {d['start']} → {d['end']}. "
                f"Source: {d['source']}.")
    a, bcol = st.columns(2)
    a.image(str(FIGURES_DIR / "aqi_by_city.png"))
    bcol.image(str(FIGURES_DIR / "dominant_pollutant.png"))
    st.image(str(FIGURES_DIR / "aqi_timeseries.png"))
    st.image(str(FIGURES_DIR / "correlation.png"), width=700)
    p = rep["preprocessing"]
    st.markdown(f"Preprocessing: {p['outlier_cells_clipped']} outlier readings clipped, "
                f"{p['rows_dropped_incomplete']} incomplete rows dropped, {p['feature_rows']:,} rows used for modelling. "
                f"Feature selection kept **{len(rep['features']['selected'])} of {rep['features']['all']}** features.")

with tab_how:
    st.markdown("""
| Stage | What happens |
|---|---|
| **Data collection** | Hourly PM2.5, PM10, CO, NO₂, SO₂, O₃ (CAMS) and temperature, humidity, wind, rain, pressure, boundary-layer height (ERA5 / weather forecast) for 8 Indian cities from Open-Meteo. |
| **Preprocessing** | Daily aggregation with CPCB averaging rules (24-h means; 8-h max for CO and O₃). Short gaps interpolated, impossible values removed, outliers clipped with IQR fences learned on training data only. |
| **AQI calculation** | CPCB National AQI: each pollutant → sub-index by breakpoint interpolation; AQI = worst sub-index. |
| **Feature engineering** | Lags (1, 2, 3, 7 days), rolling means, today's weather, forecast weather for the target day, season, weekday, Diwali proximity, crop-burning season, city. |
| **Feature selection** | Permutation importance of a Random Forest on the validation set; weak features dropped. |
| **Models** | Persistence baseline, Linear Regression, Random Forest, XGBoost, LSTM. Trees and LSTM learn the change from today's AQI. |
| **Prediction** | AQI 1, 2 and 3 days ahead with CPCB category (Good → Severe). |
| **Uncertainty** | 80% range from quantile XGBoost, calibrated on validation data (conformal prediction). |
| **Explanation** | SHAP values from XGBoost show which factors push the forecast up or down. |
| **Alerts** | Early warning when the forecast is Poor (AQI > 200) or worse: dashboard banner, optional email. |
""")
    st.caption("Limitations: pollutant data is from the CAMS atmospheric model (about 40 km grid), not ground monitoring "
               "stations, so it can differ from CPCB station readings; a model trained on station data would be needed for street-level accuracy.")
