"""Baselines, the gradient boosting model and the competition metric."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import CATEGORICAL_FEATURES, MODEL_FEATURES

HOLIDAY_WEIGHT = 5.0


def wmae(y_true: np.ndarray, y_pred: np.ndarray, is_holiday: np.ndarray) -> float:
    """Weighted mean absolute error, the official Kaggle metric.

    Holiday weeks count five times as much as normal weeks, because a bad
    forecast during Thanksgiving or Christmas costs the business far more.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.where(np.asarray(is_holiday, dtype=bool), HOLIDAY_WEIGHT, 1.0)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))


def seasonal_naive(df: pd.DataFrame) -> np.ndarray:
    """Predict that sales repeat the same week of the previous year."""
    return df["lag_52"].to_numpy(dtype=float)


def store_dept_mean(df: pd.DataFrame) -> np.ndarray:
    """Predict the historical average of each store and department."""
    return df["dept_store_mean"].to_numpy(dtype=float)


@dataclass
class GradientBoostingForecaster:
    """Histogram gradient boosting with an absolute-error loss.

    Absolute error matches the WMAE metric and is less pulled by the few huge
    holiday spikes than squared error.
    """

    max_iter: int = 600
    learning_rate: float = 0.08
    max_leaf_nodes: int = 63
    random_state: int = 42

    def __post_init__(self) -> None:
        categorical = [f in CATEGORICAL_FEATURES for f in MODEL_FEATURES]
        self.model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            categorical_features=categorical,
            early_stopping=False,
            random_state=self.random_state,
        )

    def fit(self, df: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "GradientBoostingForecaster":
        self.model.fit(df[MODEL_FEATURES], df["Weekly_Sales"], sample_weight=sample_weight)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[MODEL_FEATURES])


def holiday_weights(df: pd.DataFrame) -> np.ndarray:
    """Training weights that mirror the WMAE metric."""
    return np.where(df["IsHoliday"].astype(bool), HOLIDAY_WEIGHT, 1.0)
