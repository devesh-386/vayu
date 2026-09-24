"""Streamlit dashboard: live AQI, next-day forecast, alerts and model performance.

    streamlit run app.py
"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from airq import alerts
from airq.aqi import AQI_BANDS, BREAKPOINTS, CATEGORIES, CATEGORY_COLORS, category_index
from airq.config import CITIES, FIGURES_DIR, REPORTS_DIR
from airq.predict import forecast, load_bundle, model_names, recent_frame

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


def history_chart(daily: pd.DataFrame, result: dict) -> go.Figure:
    today = pd.Timestamp(result["date"])
    past = daily[daily["date"] <= today].tail(30)
    fig = go.Figure()
    for lo, hi, c, name in zip(AQI_BANDS[:-1], AQI_BANDS[1:], CATEGORY_COLORS, CATEGORIES):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=c, opacity=0.10, line_width=0,
                      annotation_text=name, annotation_position="right", annotation_font_size=10)
    fig.add_trace(go.Scatter(x=past["date"], y=past["aqi"], mode="lines+markers", name="Observed AQI",
                             line=dict(color="#3182bd", width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=[today, pd.Timestamp(result["target_date"])],
                             y=[past["aqi"].iloc[-1], result["predicted_aqi"]], mode="lines+markers",
                             name=f"Forecast ({result['model']})", line=dict(color="#e6550d", dash="dash", width=2),
                             marker=dict(size=[0, 12], symbol="diamond")))
    ymax = max(past["aqi"].max(), result["predicted_aqi"]) * 1.2
    fig.update_layout(height=380, margin=dict(l=10, r=90, t=10, b=10), yaxis=dict(title="AQI", range=[0, max(ymax, 120)]),
                      legend=dict(orientation="h", y=1.08), hovermode="x unified")
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

with st.sidebar:
    st.header("Settings")
    city = st.selectbox("City", list(CITIES), index=0)
    names = model_names(b)
    model = st.selectbox("Model", names, index=names.index(b["best_model"]),
                         help=f"Default is the model with the lowest validation error ({b['best_model']}).")
    st.divider()
    st.caption("Data: Open-Meteo (CAMS air quality, ERA5 / forecast weather). "
               "AQI: CPCB National AQI. Refreshes every 30 minutes.")

st.title("Intelligent Air Quality Prediction System")
st.caption("Next-day AQI forecasting from pollutant and weather data using Machine Learning")

tab_live, tab_models, tab_data, tab_how = st.tabs(["Live forecast", "Model performance", "Data analysis", "How it works"])

with tab_live:
    try:
        with st.spinner(f"Fetching latest data for {city}..."):
            daily = recent(city)
            res = forecast(city, model, b, daily)
    except Exception as e:  # network failure or incomplete data
        st.error(f"Could not produce a forecast for {city}: {e}")
        st.stop()

    today_row = daily[daily["date"] == pd.Timestamp(res["date"])].iloc[0]
    tomorrow_row = daily[daily["date"] == pd.Timestamp(res["target_date"])]

    if res["alert"]:
        st.error(f"⚠️ **Early warning:** air quality in {city} is forecast to be **{res['predicted_category']}** "
                 f"(AQI {res['predicted_aqi']:.0f}) on {res['target_date']}. {res['advice']}")
    else:
        st.success(f"No alert: tomorrow's air in {city} is forecast to be {res['predicted_category']}.")

    c1, c2, c3 = st.columns(3)
    with c1:
        aqi_card(f"Today · {res['date']}", res["today_aqi"], res["today_category"],
                 f"Dominant pollutant: {POLLUTANT_LABELS.get(res['today_dominant'], res['today_dominant'])}")
    with c2:
        delta = res["predicted_aqi"] - res["today_aqi"]
        aqi_card(f"Forecast · {res['target_date']}", res["predicted_aqi"], res["predicted_category"],
                 f"{'▲' if delta > 0 else '▼'} {abs(delta):.0f} vs today · {res['model']}")
    with c3:
        st.markdown("**Health advice**")
        st.write(res["advice"])
        st.markdown("**Weather**")
        w1, w2 = st.columns(2)
        w1.metric("Temp today", f"{today_row['temp']:.1f} °C")
        w2.metric("Humidity", f"{today_row['humidity']:.0f} %")
        if len(tomorrow_row):
            t = tomorrow_row.iloc[0]
            w1.metric("Wind tomorrow", f"{t['wind_speed']:.0f} km/h")
            w2.metric("Rain tomorrow", f"{t['precip']:.1f} mm")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Last 30 days and tomorrow's forecast")
        st.plotly_chart(history_chart(daily, res), width="stretch")
    with right:
        st.subheader("Today's pollutant sub-indices")
        st.plotly_chart(subindex_chart(today_row), width="stretch")
        st.caption("The AQI is the highest sub-index (CPCB method).")

    with st.expander("All models for this city"):
        rows = []
        for n in names:
            try:
                r = forecast(city, n, b, daily)
                rows.append({"Model": n, "Forecast AQI": r["predicted_aqi"], "Category": r["predicted_category"]})
            except ValueError as e:
                rows.append({"Model": n, "Forecast AQI": None, "Category": str(e)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("Email alert"):
        st.code(alerts.build_message(res), language=None)
        if alerts.email_configured():
            if st.button("Send this alert by email"):
                st.success("Sent." if alerts.send_email(res) else "Email not configured.")
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

    st.subheader("Mean absolute error by city (test)")
    st.dataframe(pd.DataFrame(rep["test_mae_by_city"]).T.round(1), width="stretch")

    a, bcol = st.columns(2)
    a.image(str(FIGURES_DIR / "test_delhi.png"), caption="Delhi: forecast vs actual (test period)")
    bcol.image(str(FIGURES_DIR / "test_chennai.png"), caption="Chennai: forecast vs actual (test period)")
    a.image(str(FIGURES_DIR / "confusion_matrix.png"), caption="AQI category confusion matrix")
    bcol.image(str(FIGURES_DIR / "feature_importance.png"), caption="Most useful features")

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
| **Feature engineering** | Lags (1, 2, 3, 7 days), rolling means, today's weather, tomorrow's forecast weather, season and weekday, city. |
| **Feature selection** | Permutation importance of a Random Forest on the validation set; weak features dropped. |
| **Models** | Persistence baseline, Linear Regression, Random Forest, XGBoost, LSTM. Trees and LSTM learn the change from today's AQI. |
| **Prediction** | Next-day AQI and CPCB category (Good → Severe). |
| **Alerts** | Early warning when the forecast is Poor (AQI > 200) or worse: dashboard banner, optional email. |
""")
    st.caption("Limitations: pollutant data is from the CAMS atmospheric model (about 40 km grid), not ground monitoring "
               "stations, so it can differ from CPCB station readings; a model trained on station data would be needed for street-level accuracy.")
