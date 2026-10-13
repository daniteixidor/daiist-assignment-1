"""
Shared, non-training pieces of the pipeline: loading the raw data, rebuilding
the calendar, feature engineering, the time-based split and the business
economics. Both train.py and app.py import from here so the app sees exactly
the same features and split that the models were trained on — but nothing in
this file fits a model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
ARTIFACTS = ROOT / "artifacts"

# ---------------------------------------------------------------------------
# Business economics (see REPORT.md, "Business framing").
# Every contacted client costs agent time across the campaign's calls; every
# client who subscribes is worth the net margin on a term deposit.
# ---------------------------------------------------------------------------
COST_PER_CONTACT = 25.0   # EUR: ~2.5 calls x ~EUR 10 loaded agent cost per call
VALUE_PER_SUBSCRIBER = 80.0  # EUR: e.g. EUR 8k avg deposit x ~1% net interest margin

# ---------------------------------------------------------------------------
# Time-based split. Rows in the file are in chronological order
# (May 2008 -> Nov 2010); we rebuild the year from the month sequence.
# ---------------------------------------------------------------------------
TRAIN_END = 200910  # train: May 2008 .. Oct 2009
VAL_END = 201003    # val:   Nov 2009 .. Mar 2010   (test: Apr 2010 .. Nov 2010)

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

# Continuous features -> StandardScaler (fitted on train).
NUMERIC_FEATURES = [
    "log_campaign",     # log(1 + calls this campaign), capped at 20 calls
    "log_previous",     # log(1 + contacts in earlier campaigns)
    "euribor3m",        # single macro indicator (others are ~collinear with it)
]
# Already on a 0..1 scale -> passed through UNscaled. Standardising a feature
# that is 0 for 96% of rows turns the rare 1s into z-scores of 8-30, which made
# plain gradient descent crawl (see REPORT.md, three-way comparison).
PASSTHROUGH_FEATURES = [
    "prev_contacted",   # 1 if pdays is recorded (client reached in an earlier campaign)
    "pdays_recency",    # 1/(1+pdays) if recorded, 0 for the 999 "never" sentinel
    "default_unknown",  # 1 if default status is not 'no' (only 3 'yes' rows)
]
# Categorical -> one-hot.
CATEGORICAL_FEATURES = [
    "age_band",         # <25 / 25-34 / 35-49 / 50-59 / 60+: subscription is U-shaped in age
    "job", "marital", "education", "housing", "loan",
    "contact", "day_of_week", "poutcome",
]
AGE_BINS = [0, 25, 35, 50, 60, 200]
AGE_LABELS = ["<25", "25-34", "35-49", "50-59", "60+"]
# Deliberately NOT used: duration (only known after the call -> leakage),
# month (encodes the 2008 campaign calendar, not a stable effect),
# emp.var.rate / nr.employed (r > 0.95 with euribor3m), cons.price.idx /
# cons.conf.idx (monthly constants that act as time IDs; see REPORT.md).


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")
    month_idx = df["month"].map({m: i for i, m in enumerate(MONTHS, start=1)}).to_numpy()
    # Each time the month number goes down, a new calendar year started.
    year = 2008 + np.r_[0, np.cumsum(np.diff(month_idx) < 0)]
    df["year"] = year
    df["period"] = year * 100 + month_idx            # e.g. 200905
    df["date"] = pd.to_datetime(dict(year=year, month=month_idx, day=1))
    df["target"] = (df["y"] == "yes").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise transformations only: no statistics from other rows, so this
    can't leak information across the split. (Scaling/encoding, which DO learn
    from data, live in the sklearn ColumnTransformer fitted on train only.)"""
    X = pd.DataFrame(index=df.index)
    never = df["pdays"] == 999
    X["prev_contacted"] = (~never).astype(int)
    X["pdays_recency"] = np.where(never, 0.0, 1.0 / (1.0 + df["pdays"]))
    X["log_campaign"] = np.log1p(df["campaign"].clip(upper=20))
    X["log_previous"] = np.log1p(df["previous"])
    X["default_unknown"] = (df["default"] != "no").astype(int)
    X["euribor3m"] = df["euribor3m"]
    for c in CATEGORICAL_FEATURES[1:]:
        X[c] = df[c]
    X["age_band"] = pd.cut(df["age"], AGE_BINS, right=False, labels=AGE_LABELS).astype(str)
    # 'illiterate' has 18 rows in train (1 in val/test): its own coefficient
    # would be noise, so pool it with the nearest level, basic.4y.
    X["education"] = X["education"].replace({"illiterate": "basic.4y"})
    return X[NUMERIC_FEATURES + PASSTHROUGH_FEATURES + CATEGORICAL_FEATURES]


def split_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    p = df["period"]
    return {
        "train": (p <= TRAIN_END).to_numpy(),
        "val": ((p > TRAIN_END) & (p <= VAL_END)).to_numpy(),
        "test": (p > VAL_END).to_numpy(),
    }


def profit(y_true, contact, cost=COST_PER_CONTACT, value=VALUE_PER_SUBSCRIBER) -> float:
    """Net campaign profit (EUR) of contacting the clients flagged in `contact`."""
    y_true = np.asarray(y_true)
    contact = np.asarray(contact).astype(bool)
    tp = np.sum(contact & (y_true == 1))
    return float(tp * value - contact.sum() * cost)


class LogisticRegressionNet(torch.nn.Module):
    """Model definition for the standard-PyTorch version. Lives here (not in
    train.py) so app.py can rebuild the module and load its saved weights."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):            # returns logits; sigmoid applied outside
        return self.linear(x).squeeze(-1)
