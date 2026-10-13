"""
Assignment 1 — training pipeline (run with `uv run python main.py train`).

Bank telemarketing: predict whether a client will subscribe to a term deposit
*before* calling them, so the bank can decide whom to put on the call list.

Steps:
  1. load data, rebuild calendar, engineer features      (pipeline.py)
  2. time-based split: train <= Oct 2009 < val <= Mar 2010 < test
  3. fit preprocessing (scaling + one-hot) on TRAIN only
  4. tune C (L2 strength) on val with sklearn, then train the SAME model
     three ways with that C: sklearn / manual PyTorch / nn.Module + optim
  5. pick each model's decision threshold on VAL by max campaign profit
  6. evaluate on TEST vs naive baselines, compare the three implementations
  7. save models + preprocessor + metrics to artifacts/ for app.py
"""

import json
import time

import joblib
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline import (
    ARTIFACTS, CATEGORICAL_FEATURES, COST_PER_CONTACT, NUMERIC_FEATURES, PASSTHROUGH_FEATURES,
    VALUE_PER_SUBSCRIBER, LogisticRegressionNet, engineer_features, load_raw, profit, split_masks,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_default_dtype(torch.float64)  # float64 so all three can agree to ~1e-6
ARTIFACTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1-3. Data, split, preprocessing
# ---------------------------------------------------------------------------
df = load_raw()
X_all = engineer_features(df)
y_all = df["target"].to_numpy()
masks = split_masks(df)

for name, m in masks.items():
    span = f"{df.loc[m, 'date'].min():%Y-%m} .. {df.loc[m, 'date'].max():%Y-%m}"
    print(f"{name:5s}: {m.sum():6d} rows  {span}  subscribe rate {y_all[m].mean():.3f}")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("bin", "passthrough", PASSTHROUGH_FEATURES),
    # drop="first": without it, every categorical block's dummies sum to 1 (= the
    # intercept column), so w and b are not uniquely identified. See REPORT.md.
    # min_frequency=100: any level seen < 100 times in train (here only
    # marital=unknown, 77 rows) becomes an "infrequent" level, a safety net
    # against noisy, barely-identified coefficients.
    ("cat", OneHotEncoder(drop="first", min_frequency=100, handle_unknown="infrequent_if_exist",
                          sparse_output=False),
     CATEGORICAL_FEATURES),
])
X_tr = preprocessor.fit_transform(X_all[masks["train"]])   # fit on TRAIN only
X_va = preprocessor.transform(X_all[masks["val"]])
X_te = preprocessor.transform(X_all[masks["test"]])
y_tr, y_va, y_te = (y_all[masks[k]] for k in ("train", "val", "test"))
feature_names = list(preprocessor.get_feature_names_out())
n, d = X_tr.shape
print(f"design matrix: {n} x {d}")


def metrics(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "avg_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12))),
        "brier": float(brier_score_loss(y, p)),
        "mean_pred": float(np.mean(p)),
    }


# ---------------------------------------------------------------------------
# 4a. Tune regularisation strength C on the validation set (sklearn only;
#     the chosen C is then reused by both PyTorch versions).
# ---------------------------------------------------------------------------
C_GRID = [0.01, 0.1, 1.0, 10.0]
tuning_C = []
for C in C_GRID:
    m = LogisticRegression(C=C, max_iter=10_000).fit(X_tr, y_tr)
    pv = m.predict_proba(X_va)[:, 1]
    tuning_C.append({"C": C, **metrics(y_va, pv)})
    print(f"C={C:<5}  val AUC={tuning_C[-1]['roc_auc']:.4f}  val logloss={tuning_C[-1]['log_loss']:.4f}")
# Select on val AUC: the model's job is to RANK clients; the threshold is set
# separately below, so calibration matters less than ordering.
# Ties (val AUC barely moves with C) go to the STRONGEST regularisation within
# 0.0005 AUC of the best: no evidence the extra flexibility helps.
top_auc = max(r["roc_auc"] for r in tuning_C)
best_C = min(r["C"] for r in tuning_C if r["roc_auc"] >= top_auc - 0.0005)
print(f"chosen C = {best_C}")

# The objective all three implementations minimise (identical to sklearn's,
# divided by C*n so it reads as an average loss):
#     J(w, b) = mean BCE(sigmoid(Xw + b), y) + lam/2 * ||w||^2,   lam = 1/(C*n)
# The intercept b is NOT penalised (sklearn doesn't penalise it either).
lam = 1.0 / (best_C * n)

# ---------------------------------------------------------------------------
# 4b. scikit-learn
# ---------------------------------------------------------------------------
t0 = time.time()
sk_model = LogisticRegression(C=best_C, max_iter=10_000, tol=1e-10).fit(X_tr, y_tr)
sk_time = time.time() - t0
sk_w = sk_model.coef_.ravel()
sk_b = float(sk_model.intercept_[0])

Xt_tr = torch.tensor(X_tr)
yt_tr = torch.tensor(y_tr, dtype=torch.float64)


def objective(logits, w):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt_tr)
    return bce + 0.5 * lam * (w ** 2).sum()


# ---------------------------------------------------------------------------
# 4c. Manual PyTorch: raw tensors, autograd, hand-written gradient step.
#     Full-batch gradient descent (the data fits in memory; full batch removes
#     SGD noise so the result is directly comparable to sklearn's optimum).
# ---------------------------------------------------------------------------
def train_manual(lr, epochs=3000):
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for epoch in range(epochs):
        logits = Xt_tr @ w + b
        loss = objective(logits, w)
        loss.backward()                      # autograd fills w.grad, b.grad
        with torch.no_grad():                # the update itself is not tracked
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        if epoch % 50 == 0 or epoch == epochs - 1:
            history.append((epoch, loss.item()))
    return w.detach(), b.detach(), history


# Learning rate is an OPTIMISATION knob, so it's chosen by the final training
# objective (did we reach the optimum?), not by a validation metric.
tuning_lr_manual = []
best = None
for lr in [0.1, 0.5, 1.0, 2.0, 4.0]:
    w, b, hist = train_manual(lr, epochs=600)
    final = hist[-1][1]
    tuning_lr_manual.append({"lr": lr, "objective_after_600_epochs": final})
    print(f"manual lr={lr:<4} objective after 600 epochs = {final:.6f}")
    if np.isfinite(final) and (best is None or final < best[1]):
        best = (lr, final)
lr_manual = best[0]
t0 = time.time()
man_w, man_b, man_hist = train_manual(lr_manual, epochs=6000)
man_time = time.time() - t0


# ---------------------------------------------------------------------------
# 4d. Standard PyTorch: nn.Module + torch.optim.
# ---------------------------------------------------------------------------
def train_standard(lr, epochs=2000):
    torch.manual_seed(SEED)
    model = LogisticRegressionNet(d)
    torch.nn.init.zeros_(model.linear.weight)
    torch.nn.init.zeros_(model.linear.bias)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # L2 is added to the loss on the WEIGHT only (Adam's weight_decay would
    # also hit the bias and, with Adam, isn't the same as an L2 penalty).
    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = objective(model(Xt_tr), model.linear.weight)
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            history.append((epoch, loss.item()))
    return model, history


tuning_lr_std = []
best = None
for lr in [0.001, 0.01, 0.05, 0.1]:
    _, hist = train_standard(lr, epochs=600)
    final = hist[-1][1]
    tuning_lr_std.append({"lr": lr, "objective_after_600_epochs": final})
    print(f"adam   lr={lr:<5} objective after 600 epochs = {final:.6f}")
    if np.isfinite(final) and (best is None or final < best[1]):
        best = (lr, final)
lr_std = best[0]
t0 = time.time()
std_model, std_hist = train_standard(lr_std, epochs=3000)
std_time = time.time() - t0
std_model.eval()
std_w = std_model.linear.weight.detach().ravel()
std_b = std_model.linear.bias.detach()


# ---------------------------------------------------------------------------
# 5-6. Evaluate
# ---------------------------------------------------------------------------
def p_sklearn(X):
    return sk_model.predict_proba(X)[:, 1]


def p_manual(X):
    return torch.sigmoid(torch.tensor(X) @ man_w + man_b).numpy()


def p_standard(X):
    with torch.no_grad():
        return torch.sigmoid(std_model(torch.tensor(X))).numpy()


models = {"sklearn": p_sklearn, "torch_manual": p_manual, "torch_standard": p_standard}
with torch.no_grad():
    sk_obj = objective(torch.tensor(X_tr @ sk_w + sk_b), torch.tensor(sk_w)).item()
train_objective = {
    "sklearn": sk_obj,
    "torch_manual": man_hist[-1][1],
    "torch_standard": std_hist[-1][1],
}
THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.005), 3)


def best_threshold(y, p):
    profits = [profit(y, p >= t) for t in THRESHOLDS]
    i = int(np.argmax(profits))
    return float(THRESHOLDS[i]), profits


def decision_stats(y, p, t):
    contact = p >= t
    tp = int(np.sum(contact & (y == 1))); fp = int(np.sum(contact & (y == 0)))
    fn = int(np.sum(~contact & (y == 1))); tn = int(np.sum(~contact & (y == 0)))
    pr = profit(y, contact)
    return {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "contacted_share": float(contact.mean()),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "profit_eur": pr, "profit_per_client_eur": pr / len(y)}


results = {}
for name, f in models.items():
    pv, pt = f(X_va), f(X_te)
    t_star, _ = best_threshold(y_va, pv)
    results[name] = {
        "val": metrics(y_va, pv),
        "test": metrics(y_te, pt),
        "train_objective": train_objective[name],
        "threshold_chosen_on_val": t_star,
        "test_decision": decision_stats(y_te, pt, t_star),
    }

# Naive baselines: a constant prediction equal to the train base rate (it
# can't rank anyone: AUC 0.5) and the two "no-model" policies the bank could
# run: call everyone (what the original campaign did) or call no one.
prior = float(y_tr.mean())
baselines = {
    "constant_train_rate": metrics(y_te, np.full(len(y_te), prior)),
    "call_everyone": decision_stats(y_te, np.ones(len(y_te)), 0.5),
    "call_no_one": decision_stats(y_te, np.zeros(len(y_te)), 0.5),
}

# Do the three implementations agree?
P_te = {k: f(X_te) for k, f in models.items()}
weights = {"sklearn": sk_w, "torch_manual": man_w.numpy(), "torch_standard": std_w.numpy()}
biases = {"sklearn": sk_b, "torch_manual": float(man_b), "torch_standard": float(std_b)}
agreement = {}
for k in ("torch_manual", "torch_standard"):
    t_sk = results["sklearn"]["threshold_chosen_on_val"]
    agreement[k + "_vs_sklearn"] = {
        "max_abs_coef_diff": float(np.max(np.abs(weights[k] - sk_w))),
        "intercept_diff": float(biases[k] - sk_b),
        "max_abs_prob_diff_test": float(np.max(np.abs(P_te[k] - P_te["sklearn"]))),
        "same_decision_share_test_at_sklearn_threshold":
            float(np.mean((P_te[k] >= t_sk) == (P_te["sklearn"] >= t_sk))),
    }

# Diagnostics that justify the framing (sklearn only, not used by the app):
#   - what a RANDOM split would have reported (leaks the 2010 regime into train)
#   - what including `duration` would report (leaks the outcome of the call)
diag = {}
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2,
                                  stratify=y_all, random_state=SEED)
for label, use_duration in [("random_split", False), ("random_split_with_duration", True)]:
    Xd = X_all.copy()
    num = list(NUMERIC_FEATURES)
    if use_duration:
        Xd["duration"] = df["duration"]
        num.append("duration")
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("bin", "passthrough", PASSTHROUGH_FEATURES),
                             ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES)])
    A = pre.fit_transform(Xd.iloc[idx_tr]); B = pre.transform(Xd.iloc[idx_te])
    m = LogisticRegression(C=best_C, max_iter=10_000).fit(A, y_all[idx_tr])
    diag[label + "_test_auc"] = float(roc_auc_score(y_all[idx_te], m.predict_proba(B)[:, 1]))
diag["time_split_test_auc"] = results["sklearn"]["test"]["roc_auc"]
print("diagnostics:", {k: round(v, 4) for k, v in diag.items()})

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
joblib.dump(sk_model, ARTIFACTS / "sklearn_logreg.joblib")
torch.save({"w": man_w, "b": man_b}, ARTIFACTS / "torch_manual.pt")
torch.save(std_model.state_dict(), ARTIFACTS / "torch_standard.pt")

summary = {
    "data": {k: {"rows": int(m.sum()), "subscribe_rate": float(y_all[m].mean()),
                 "from": f"{df.loc[m, 'date'].min():%Y-%m}", "to": f"{df.loc[m, 'date'].max():%Y-%m}"}
             for k, m in masks.items()},
    "n_features_after_encoding": d,
    "feature_names": feature_names,
    "economics": {"cost_per_contact_eur": COST_PER_CONTACT,
                  "value_per_subscriber_eur": VALUE_PER_SUBSCRIBER,
                  "breakeven_probability": COST_PER_CONTACT / VALUE_PER_SUBSCRIBER},
    "tuning": {"C_grid_on_val": tuning_C, "chosen_C": best_C,
               "lr_manual": tuning_lr_manual, "chosen_lr_manual": lr_manual,
               "lr_adam": tuning_lr_std, "chosen_lr_adam": lr_std},
    "train_seconds": {"sklearn": sk_time, "torch_manual": man_time, "torch_standard": std_time},
    "loss_history": {"torch_manual": man_hist, "torch_standard": std_hist},
    "results": results,
    "baselines": baselines,
    "agreement": agreement,
    "coefficients": {k: dict(zip(feature_names, map(float, v))) for k, v in weights.items()},
    "intercepts": biases,
    "diagnostics": diag,
}
with open(ARTIFACTS / "metrics.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print("\n=== TEST SET (Apr-Nov 2010) ===")
print(f"{'model':16s} {'AUC':>6s} {'AP':>6s} {'logloss':>8s} {'thr':>6s} {'profit EUR':>11s}")
for k, r in results.items():
    print(f"{k:16s} {r['test']['roc_auc']:6.4f} {r['test']['avg_precision']:6.4f} "
          f"{r['test']['log_loss']:8.4f} {r['threshold_chosen_on_val']:6.3f} "
          f"{r['test_decision']['profit_eur']:11,.0f}")
print(f"{'call everyone':16s} {'0.5':>6s} {'':6s} {'':8s} {'':6s} {baselines['call_everyone']['profit_eur']:11,.0f}")
print("agreement:", json.dumps(agreement, indent=1))
print(f"saved artifacts to {ARTIFACTS}")
