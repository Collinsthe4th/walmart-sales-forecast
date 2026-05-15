"""Loading and joining the raw Walmart files.

The Kaggle release ships four files:

* ``train.csv``    - weekly sales per store and department (the target)
* ``test.csv``     - the same keys without sales (the weeks to forecast)
* ``stores.csv``   - store type (A, B or C) and floor size
* ``features.csv`` - weekly regional data (temperature, fuel price, CPI,
  unemployment and five promotional markdowns)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

FEATURE_COLUMNS = [
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment",
]


def _read(name: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} was not found. Place the Kaggle files in {raw_dir} (see README)."
        )
    return pd.read_csv(path)


def load_sales(split: str = "train", raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return the train or test sales table with store and regional features joined.

    Args:
        split: ``"train"`` (has ``Weekly_Sales``) or ``"test"`` (no target).
        raw_dir: folder holding the four Kaggle CSV files.

    Returns:
        One row per store, department and week, sorted by those keys.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    sales = _read(f"{split}.csv", raw_dir)
    stores = _read("stores.csv", raw_dir)
    features = _read("features.csv", raw_dir)

    for frame in (sales, features):
        frame["Date"] = pd.to_datetime(frame["Date"])

    # IsHoliday appears in both sales and features; keep the sales copy.
    features = features.drop(columns=["IsHoliday"])

    df = (
        sales.merge(stores, on="Store", how="left", validate="many_to_one")
        .merge(features, on=["Store", "Date"], how="left", validate="many_to_one")
        .sort_values(["Store", "Dept", "Date"])
        .reset_index(drop=True)
    )

    # Missing markdowns mean "no promotion that week", so zero is the honest fill.
    markdowns = [c for c in FEATURE_COLUMNS if c.startswith("MarkDown")]
    df[markdowns] = df[markdowns].fillna(0.0)
    df["IsHoliday"] = df["IsHoliday"].astype(bool)
    return df
