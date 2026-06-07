"""Forecast the Kaggle test weeks with the model trained on all history.

Run ``scripts/run_pipeline.py`` first. Writes ``results/submission.csv`` in the
Kaggle format (``Id`` = Store_Dept_Date, ``Weekly_Sales``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from walmart_forecast.data import load_sales  # noqa: E402
from walmart_forecast.features import build_features  # noqa: E402


def main() -> None:
    model_path = ROOT / "models" / "gbr_full.joblib"
    if not model_path.exists():
        sys.exit("No trained model found. Run scripts/run_pipeline.py first.")
    model = joblib.load(model_path)

    history = load_sales("train")
    test = build_features(load_sales("test"), history=history)
    test["Weekly_Sales"] = model.predict(test)
    test["Id"] = (
        test["Store"].astype(str) + "_" + test["Dept"].astype(str) + "_" + test["Date"].dt.strftime("%Y-%m-%d")
    )
    out = ROOT / "results" / "submission.csv"
    test[["Id", "Weekly_Sales"]].to_csv(out, index=False)
    print(f"Wrote {len(test):,} forecasts to {out}")


if __name__ == "__main__":
    main()
