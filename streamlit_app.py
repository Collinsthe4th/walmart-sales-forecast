"""Streamlit dashboard for the Walmart weekly sales forecast.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(page_title="Walmart Weekly Sales Forecast", layout="wide")

HEADER = "https://images.unsplash.com/photo-1591696205602-2f950c417cb9?auto=format&fit=crop&w=1600&h=340&q=80"
st.image(HEADER, use_container_width=True)


@st.cache_data
def load_results() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RESULTS / "holdout_predictions.csv.gz", parse_dates=["Date"])
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    return df, metrics


def wmae(actual: pd.Series, pred: pd.Series, holiday: pd.Series) -> float:
    w = holiday.astype(bool).map({True: 5.0, False: 1.0}).astype(float)
    return float((w * (actual - pred).abs()).sum() / w.sum())


df, metrics = load_results()

st.title("Walmart Weekly Sales Forecast")
st.caption(
    "I trained a gradient boosting model to forecast weekly sales for 3,331 store and department pairs "
    "across 45 Walmart stores. The results below come from the 39 weeks I held out "
    f"({metrics['validation_period']['start']} to {metrics['validation_period']['end']}), scored with WMAE "
    "(weighted mean absolute error, where holiday weeks count five times)."
)

c1, c2, c3 = st.columns(3)
scores = metrics["wmae"]
c1.metric("Gradient boosting WMAE", f"${scores['Gradient boosting (HistGBR)']:,.0f}")
c2.metric("Seasonal naive WMAE", f"${scores['Seasonal naive (same week last year)']:,.0f}")
c3.metric("Improvement", f"{metrics['improvement_over_seasonal_naive_pct']}%")

st.subheader("All stores")
total = df.groupby("Date")[["Weekly_Sales", "seasonal_naive", "gradient_boosting"]].sum() / 1e6
total.columns = ["Actual", "Seasonal naive", "Gradient boosting"]
st.line_chart(total, y_label="Total weekly sales (USD millions)")

st.subheader("One store and department")
left, right = st.columns([1, 3])
with left:
    store = st.selectbox("Store", sorted(df["Store"].unique()), index=0)
    depts = sorted(df.loc[df["Store"] == store, "Dept"].unique())
    dept = st.selectbox("Department", depts, index=0)
    sel = df[(df["Store"] == store) & (df["Dept"] == dept)].sort_values("Date")
    st.write(f"Store type **{sel['Type'].iloc[0]}**, {len(sel)} weeks")
    st.metric("Model WMAE here", f"${wmae(sel['Weekly_Sales'], sel['gradient_boosting'], sel['IsHoliday']):,.0f}")
    st.metric("Naive WMAE here", f"${wmae(sel['Weekly_Sales'], sel['seasonal_naive'], sel['IsHoliday']):,.0f}")
with right:
    chart = sel.set_index("Date")[["Weekly_Sales", "seasonal_naive", "gradient_boosting"]]
    chart.columns = ["Actual", "Seasonal naive", "Gradient boosting"]
    st.line_chart(chart, y_label="Weekly sales (USD)")

st.subheader("Error by store type")
by_type = pd.Series(metrics["gbr_mae_by_store_type"], name="Mean absolute error (USD)")
st.bar_chart(by_type)
st.markdown(
    "Every model misses the spike in early April 2012. That week is Easter, which fell on 24 April in 2011 "
    "but on 8 April in 2012, so last year's sales point at the wrong week. "
    "[Source code and README](https://github.com/Collinsthe4th/walmart-sales-forecast) · Header photo from Unsplash."
)
