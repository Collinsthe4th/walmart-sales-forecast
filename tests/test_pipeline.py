"""Unit tests that run on small synthetic frames, so they need no data files."""

import numpy as np
import pandas as pd
import pytest

from walmart_forecast.features import MODEL_FEATURES, build_features, time_split
from walmart_forecast.models import GradientBoostingForecaster, seasonal_naive, store_dept_mean, wmae


def make_frame(weeks: int = 110) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-02-05", periods=weeks, freq="7D")
    rows = []
    for store in (1, 2):
        for dept in (1, 2):
            base = 1000 * store + 300 * dept
            for i, d in enumerate(dates):
                rows.append(
                    {
                        "Store": store,
                        "Dept": dept,
                        "Date": d,
                        "Weekly_Sales": base + 200 * np.sin(2 * np.pi * i / 52) + rng.normal(0, 20),
                        "IsHoliday": d.isocalendar().week == 47,
                        "Type": "A" if store == 1 else "B",
                        "Size": 150000 if store == 1 else 90000,
                        "Temperature": 60.0,
                        "Fuel_Price": 3.0,
                        "MarkDown1": 0.0,
                        "MarkDown2": 0.0,
                        "MarkDown3": 0.0,
                        "MarkDown4": 0.0,
                        "MarkDown5": 0.0,
                        "CPI": 210.0,
                        "Unemployment": 8.0,
                    }
                )
    return pd.DataFrame(rows)


def test_wmae_weights_holidays_five_times():
    y = np.array([10.0, 10.0])
    p = np.array([0.0, 10.0])
    assert wmae(y, p, np.array([True, False])) == pytest.approx(50 / 6)
    assert wmae(y, p, np.array([False, False])) == pytest.approx(5.0)


def test_wmae_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        wmae(np.ones(3), np.ones(2), np.zeros(3))


def test_time_split_keeps_order_and_size():
    df = make_frame()
    train, valid = time_split(df, valid_weeks=39)
    assert train["Date"].max() < valid["Date"].min()
    assert valid["Date"].nunique() == 39


def test_features_have_no_missing_values_and_no_leakage():
    df = make_frame()
    train, valid = time_split(df, valid_weeks=39)
    feats = build_features(valid, history=train)
    assert feats[MODEL_FEATURES].isna().sum().sum() == 0
    # The level statistics must come from history only.
    hist_mean = train.groupby(["Store", "Dept"])["Weekly_Sales"].mean()
    row = feats.iloc[0]
    assert row["dept_store_mean"] == pytest.approx(hist_mean[(row["Store"], row["Dept"])])


def test_lag_52_matches_last_year():
    df = make_frame()
    feats = build_features(df, history=df)
    later = feats[feats["Date"] == df["Date"].unique()[60]].iloc[0]
    year_before = df[
        (df["Store"] == later["Store"])
        & (df["Dept"] == later["Dept"])
        & (df["Date"] == later["Date"] - pd.Timedelta(weeks=52))
    ]["Weekly_Sales"].iloc[0]
    assert later["lag_52"] == pytest.approx(year_before)


def test_model_beats_mean_on_seasonal_data():
    df = make_frame()
    train, valid = time_split(df, valid_weeks=39)
    tr = build_features(train, history=train)
    va = build_features(valid, history=train)
    model = GradientBoostingForecaster(max_iter=150, max_leaf_nodes=8).fit(tr)
    err_model = wmae(va["Weekly_Sales"], model.predict(va), va["IsHoliday"])
    err_mean = wmae(va["Weekly_Sales"], store_dept_mean(va), va["IsHoliday"])
    # On a seasonal series the model must at least beat a flat average.
    assert err_model < err_mean
    assert np.isfinite(seasonal_naive(va)).all()
