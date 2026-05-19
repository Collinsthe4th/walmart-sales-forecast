"""Feature engineering for the weekly sales model.

The forecast horizon in the Kaggle test set is up to 39 weeks, so the model can
only use information that is known at least that far ahead. That rules out
last week's sales. Sales from the same week one year earlier (``lag_52``) are
always known, and they carry most of the yearly pattern (holidays,
back-to-school, and so on).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS

MODEL_FEATURES = [
    "Store",
    "Dept",
    "type_code",
    "Size",
    "week",
    "month",
    "year",
    "IsHoliday",
    "lag_52",
    "dept_store_mean",
    "dept_store_median",
    *FEATURE_COLUMNS,
]

CATEGORICAL_FEATURES = ["Store", "Dept", "type_code"]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add ISO week, month and year columns."""
    out = df.copy()
    iso = out["Date"].dt.isocalendar()
    out["week"] = iso["week"].astype(int)
    out["month"] = out["Date"].dt.month
    out["year"] = out["Date"].dt.year
    out["type_code"] = out["Type"].map({"A": 0, "B": 1, "C": 2}).astype(int)
    return out


def add_lag_52(df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Attach the sales of the same store, department and week 52 weeks earlier.

    Args:
        df: rows to enrich (may be future weeks without sales).
        history: rows with known ``Weekly_Sales``.
    """
    past = history[["Store", "Dept", "Date", "Weekly_Sales"]].copy()
    past["Date"] = past["Date"] + pd.Timedelta(weeks=52)
    past = past.rename(columns={"Weekly_Sales": "lag_52"})
    return df.merge(past, on=["Store", "Dept", "Date"], how="left")


def add_level_stats(df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Attach the mean and median sales of each store and department.

    The statistics come only from ``history`` so that validation rows never see
    their own target.
    """
    stats = (
        history.groupby(["Store", "Dept"])["Weekly_Sales"]
        .agg(dept_store_mean="mean", dept_store_median="median")
        .reset_index()
    )
    return df.merge(stats, on=["Store", "Dept"], how="left")


def build_features(df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with every model feature added.

    ``lag_52`` falls back to the store-department median when the department did
    not exist a year earlier, and to 0 when it has no history at all.
    """
    out = add_calendar(df)
    out = add_lag_52(out, history)
    out = add_level_stats(out, history)
    out["lag_52"] = out["lag_52"].fillna(out["dept_store_median"])
    for col in ("lag_52", "dept_store_mean", "dept_store_median"):
        out[col] = out[col].fillna(0.0)
    out["IsHoliday"] = out["IsHoliday"].astype(int)
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].astype(float)
    return out


def time_split(df: pd.DataFrame, valid_weeks: int = 39) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date: the last ``valid_weeks`` weeks become the validation set.

    39 weeks matches the length of the Kaggle test period.
    """
    dates = np.sort(df["Date"].unique())
    if valid_weeks >= len(dates):
        raise ValueError("valid_weeks must be shorter than the history")
    cutoff = dates[-valid_weeks]
    return df[df["Date"] < cutoff].copy(), df[df["Date"] >= cutoff].copy()
