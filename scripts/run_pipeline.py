"""Train and evaluate every model on a 39-week hold-out and save the results.

Usage:
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from walmart_forecast.data import load_sales  # noqa: E402
from walmart_forecast.features import build_features, time_split  # noqa: E402
from walmart_forecast.models import (  # noqa: E402
    GradientBoostingForecaster,
    holiday_weights,
    seasonal_naive,
    store_dept_mean,
    wmae,
)

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"


def main() -> None:
    start = time.time()
    raw = load_sales("train")
    train_raw, valid_raw = time_split(raw, valid_weeks=39)

    train = build_features(train_raw, history=train_raw)
    valid = build_features(valid_raw, history=train_raw)

    y = valid["Weekly_Sales"].to_numpy()
    hol = valid["IsHoliday"].to_numpy()

    predictions = {
        "Seasonal naive (same week last year)": seasonal_naive(valid),
        "Store-department mean": store_dept_mean(valid),
    }

    model = GradientBoostingForecaster().fit(train, sample_weight=holiday_weights(train))
    predictions["Gradient boosting (HistGBR)"] = model.predict(valid)

    scores = {name: round(wmae(y, p, hol), 2) for name, p in predictions.items()}
    best = min(scores, key=scores.get)
    baseline = scores["Seasonal naive (same week last year)"]

    metrics = {
        "validation_period": {
            "start": str(valid_raw["Date"].min().date()),
            "end": str(valid_raw["Date"].max().date()),
            "weeks": int(valid_raw["Date"].nunique()),
            "rows": int(len(valid_raw)),
        },
        "train_rows": int(len(train_raw)),
        "wmae": scores,
        "best_model": best,
        "improvement_over_seasonal_naive_pct": round(100 * (baseline - scores[best]) / baseline, 1),
        "runtime_seconds": round(time.time() - start, 1),
    }

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Total weekly sales across all stores: actual against each forecast.
    weekly = valid[["Date"]].copy()
    weekly["Actual"] = y
    for name, p in predictions.items():
        weekly[name] = p
    weekly = weekly.groupby("Date").sum() / 1e6

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(weekly.index, weekly["Actual"], color="black", lw=2.2, label="Actual")
    for name in predictions:
        ax.plot(weekly.index, weekly[name], lw=1.3, label=f"{name} (WMAE {scores[name]:,.0f})")
    ax.set_ylabel("Total weekly sales (USD millions)")
    ax.set_title("Hold-out forecasts, all 45 stores")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "holdout_forecast.png", dpi=150)
    plt.close(fig)

    # Row-level hold-out predictions for the Streamlit app.
    holdout = valid[["Store", "Dept", "Date", "Type", "IsHoliday", "Weekly_Sales"]].copy()
    holdout["seasonal_naive"] = predictions["Seasonal naive (same week last year)"].round(2)
    holdout["gradient_boosting"] = predictions["Gradient boosting (HistGBR)"].round(2)
    holdout.to_csv(RESULTS / "holdout_predictions.csv.gz", index=False, compression="gzip")

    # Error by store type shows where the model is weakest.
    valid["abs_err"] = (valid["Weekly_Sales"] - predictions["Gradient boosting (HistGBR)"]).abs()
    by_type = valid.groupby("Type")["abs_err"].mean().round(2)
    metrics["gbr_mae_by_store_type"] = by_type.to_dict()
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    MODELS.mkdir(exist_ok=True)
    full = build_features(raw, history=raw)
    final = GradientBoostingForecaster().fit(full, sample_weight=holiday_weights(full))
    joblib.dump(final, MODELS / "gbr_full.joblib")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()
