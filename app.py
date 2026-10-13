"""
Assignment 1 — Gradio dashboard (run with `uv run python main.py app`).

Loads the artifacts written by train.py — preprocessor, the three trained
logistic regressions and metrics.json — and NEVER fits anything: it only
calls .transform() on the already-fitted preprocessor and runs forward passes.
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from pipeline import (
    ARTIFACTS, COST_PER_CONTACT, VALUE_PER_SUBSCRIBER, LogisticRegressionNet,
    engineer_features, load_raw, profit, split_masks,
)

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Load data + saved artifacts (no training here)
# ---------------------------------------------------------------------------
if not (ARTIFACTS / "metrics.json").exists():
    raise SystemExit("artifacts/ not found - run `uv run python main.py train` first.")

df = load_raw()
masks = split_masks(df)
df["split"] = np.select([masks["train"], masks["val"], masks["test"]], ["train", "val", "test"], default="")
summary = json.loads((ARTIFACTS / "metrics.json").read_text())

preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
sk_model = joblib.load(ARTIFACTS / "sklearn_logreg.joblib")
manual = torch.load(ARTIFACTS / "torch_manual.pt")
X_all = preprocessor.transform(engineer_features(df))
std_model = LogisticRegressionNet(X_all.shape[1])
std_model.load_state_dict(torch.load(ARTIFACTS / "torch_standard.pt"))
std_model.eval()

with torch.no_grad():
    Xt = torch.tensor(X_all)
    PROBS = {
        "sklearn": sk_model.predict_proba(X_all)[:, 1],
        "torch_manual": torch.sigmoid(Xt @ manual["w"] + manual["b"]).numpy(),
        "torch_standard": torch.sigmoid(std_model(Xt)).numpy(),
    }
for k, p in PROBS.items():
    df[f"p_{k}"] = p

MODELS = list(PROBS)
LABEL = {"sklearn": "scikit-learn", "torch_manual": "PyTorch (manual loop)",
         "torch_standard": "PyTorch (nn.Module + optim)"}
COLOR = {"sklearn": "#2a6fdb", "torch_manual": "#e07a1f", "torch_standard": "#1a9e6e"}
SPLIT_COLOR = {"train": "rgba(42,111,219,0.07)", "val": "rgba(224,122,31,0.12)",
               "test": "rgba(26,158,110,0.12)"}
Y = df["target"].to_numpy()


# ---------------------------------------------------------------------------
# Tab 1: model comparison
# ---------------------------------------------------------------------------
def metrics_table():
    rows = []
    for k in MODELS:
        r = summary["results"][k]
        rows.append({
            "model": LABEL[k],
            "val AUC": r["val"]["roc_auc"], "test AUC": r["test"]["roc_auc"],
            "test avg precision": r["test"]["avg_precision"],
            "test log loss": r["test"]["log_loss"],
            "train objective": r["train_objective"],
            "threshold (from val)": r["threshold_chosen_on_val"],
            "test profit EUR": r["test_decision"]["profit_eur"],
            "train seconds": summary["train_seconds"][k],
        })
    b = summary["baselines"]
    rows.append({"model": "Baseline: constant train rate", "test AUC": 0.5,
                 "test avg precision": b["constant_train_rate"]["avg_precision"],
                 "test log loss": b["constant_train_rate"]["log_loss"]})
    rows.append({"model": "Baseline: call everyone", "test profit EUR": b["call_everyone"]["profit_eur"]})
    rows.append({"model": "Baseline: call no one", "test profit EUR": 0.0})
    return pd.DataFrame(rows).round(4)


def agreement_md():
    a = summary["agreement"]
    lines = ["| vs scikit-learn | max abs coef diff | intercept diff | max abs prob diff (test) | same decision (test) |",
             "|---|---|---|---|---|"]
    for k, v in a.items():
        lines.append(f"| {LABEL[k.replace('_vs_sklearn', '')]} | {v['max_abs_coef_diff']:.2e} | "
                     f"{v['intercept_diff']:.2e} | {v['max_abs_prob_diff_test']:.2e} | "
                     f"{v['same_decision_share_test_at_sklearn_threshold']:.1%} |")
    d = summary["diagnostics"]
    lines += ["", "**Why a time split?** Same model, same features, different evaluation:",
              f"random split AUC **{d['random_split_test_auc']:.3f}** · random split + `duration` "
              f"(leaks the call's outcome) **{d['random_split_with_duration_test_auc']:.3f}** · "
              f"honest time split (test = Apr–Nov 2010) **{d['time_split_test_auc']:.3f}**."]
    return "\n".join(lines)


def calibration_plot(split):
    m = masks[split]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                             line=dict(dash="dot", color="grey")))
    for k in MODELS:
        p, y = PROBS[k][m], Y[m]
        bins = pd.qcut(p, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y, "b": bins}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        fig.add_trace(go.Scatter(x=g["p"], y=g["y"], mode="lines+markers", name=LABEL[k],
                                 line=dict(color=COLOR[k], width=2 if k == "sklearn" else 1,
                                           dash=None if k == "sklearn" else "dash"),
                                 customdata=g["n"], hovertemplate="predicted %{x:.3f}<br>actual %{y:.3f}<br>n=%{customdata}"))
    fig.update_layout(title=f"Predicted vs actual subscription rate, by predicted-probability decile ({split})",
                      xaxis_title="mean predicted probability", yaxis_title="actual subscription rate",
                      xaxis_range=[0, 1], yaxis_range=[0, 1], height=430, legend=dict(orientation="h", y=-0.2))
    return fig


def over_time_plot():
    g = df.groupby("date").agg(actual=("target", "mean"), n=("target", "size"),
                               **{k: (f"p_{k}", "mean") for k in MODELS})
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=g.index, y=g["n"], name="clients called", marker_color="rgba(120,120,120,0.25)"),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=g.index, y=g["actual"], name="actual rate", mode="lines+markers",
                             line=dict(color="black", width=3)))
    for k in MODELS:
        fig.add_trace(go.Scatter(x=g.index, y=g[k], name=f"predicted ({LABEL[k]})", mode="lines",
                                 line=dict(color=COLOR[k], dash=None if k == "sklearn" else "dash")))
    for split in ("train", "val", "test"):
        dates = df.loc[masks[split], "date"]
        fig.add_vrect(x0=dates.min() - pd.Timedelta(days=15), x1=dates.max() + pd.Timedelta(days=15),
                      fillcolor=SPLIT_COLOR[split], line_width=0, annotation_text=split,
                      annotation_position="top left")
    fig.update_layout(title="Monthly mean predicted probability vs actual subscription rate",
                      height=430, legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="subscription rate", range=[0, 1], secondary_y=False)
    fig.update_yaxes(title_text="clients called", secondary_y=True, showgrid=False)
    return fig


def agreement_scatter():
    m = masks["test"]
    fig = go.Figure()
    for k in ("torch_manual", "torch_standard"):
        fig.add_trace(go.Scattergl(x=PROBS["sklearn"][m], y=PROBS[k][m], mode="markers", name=LABEL[k],
                                   marker=dict(size=5, color=COLOR[k], opacity=0.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="grey"), name="y = x"))
    fig.update_layout(title="Test-set probabilities: PyTorch vs scikit-learn", height=430,
                      xaxis_title="scikit-learn probability", yaxis_title="PyTorch probability",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def coef_plot():
    coefs = pd.DataFrame(summary["coefficients"])
    coefs = coefs.reindex(coefs["sklearn"].abs().sort_values().index)
    fig = go.Figure()
    for k in MODELS:
        fig.add_trace(go.Bar(y=coefs.index, x=coefs[k], name=LABEL[k], orientation="h",
                             marker_color=COLOR[k]))
    fig.update_layout(title="Coefficients (log-odds per unit of the encoded feature)", barmode="group",
                      height=900, legend=dict(orientation="h", y=-0.05))
    return fig


def loss_plot():
    fig = go.Figure()
    for k in ("torch_manual", "torch_standard"):
        h = np.array(summary["loss_history"][k])
        fig.add_trace(go.Scatter(x=h[:, 0] + 1, y=h[:, 1], name=LABEL[k], line=dict(color=COLOR[k])))
    fig.add_hline(y=summary["results"]["sklearn"]["train_objective"], line_dash="dot",
                  annotation_text="scikit-learn optimum")
    fig.update_layout(title="Training objective (mean BCE + L2) per epoch", xaxis_title="epoch (log scale)",
                      yaxis_title="objective", xaxis_type="log", yaxis_type="log", height=380)
    return fig


# ---------------------------------------------------------------------------
# Tab 2: data distributions
# ---------------------------------------------------------------------------
RAW_NUMERIC = ["age", "campaign", "pdays", "previous", "duration", "euribor3m",
               "emp.var.rate", "cons.price.idx", "cons.conf.idx", "nr.employed"]
RAW_CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
                   "contact", "month", "day_of_week", "poutcome"]


def feature_plot(feature, split):
    d = df if split == "all" else df[df["split"] == split]
    if feature in RAW_CATEGORICAL:
        g = d.groupby(feature).agg(n=("target", "size"), rate=("target", "mean")).sort_values("n", ascending=False)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=g.index, y=g["n"], name="clients", marker_color="#9fb7e0"))
        fig.add_trace(go.Scatter(x=g.index, y=g["rate"], name="subscription rate", mode="markers+lines",
                                 marker=dict(color="#c0392b", size=9)), secondary_y=True)
        fig.update_yaxes(title_text="clients", secondary_y=False)
        fig.update_yaxes(title_text="subscription rate", secondary_y=True, range=[0, 1], showgrid=False)
    else:
        vals = d[feature]
        if feature == "pdays":
            vals = vals.replace(999, np.nan)  # 999 = never contacted; plotting it hides everything else
        fig = go.Figure()
        for t, name, c in ((0, "did not subscribe", "#9fb7e0"), (1, "subscribed", "#c0392b")):
            fig.add_trace(go.Histogram(x=vals[d["target"] == t], name=name, marker_color=c, opacity=0.65,
                                       histnorm="probability density", nbinsx=50))
        fig.update_layout(barmode="overlay", xaxis_title=feature, yaxis_title="density")
    note = " (999 'never contacted' rows hidden)" if feature == "pdays" else ""
    fig.update_layout(title=f"{feature} by outcome — {split} rows{note}", height=440,
                      legend=dict(orientation="h", y=-0.2))
    return fig


def target_plot():
    g = df.groupby(["split"]).agg(rows=("target", "size"), rate=("target", "mean")).reindex(["train", "val", "test"])
    fig = go.Figure(go.Bar(x=g.index, y=g["rate"], text=[f"{r:.1%}<br>n={n:,}" for r, n in zip(g["rate"], g["rows"])],
                           textposition="outside", marker_color=["#2a6fdb", "#e07a1f", "#1a9e6e"]))
    fig.update_layout(title="Target: share of clients who subscribed, per split", yaxis_range=[0, 0.7],
                      yaxis_title="subscription rate", height=380)
    return fig


# ---------------------------------------------------------------------------
# Tab 3: threshold, confusion matrix, business cost
# ---------------------------------------------------------------------------
def threshold_view(model, split, threshold, cost, value):
    m = masks[split]
    p, y = PROBS[model][m], Y[m]
    contact = p >= threshold
    tp = int(np.sum(contact & (y == 1))); fp = int(np.sum(contact & (y == 0)))
    fn = int(np.sum(~contact & (y == 1))); tn = int(np.sum(~contact & (y == 0)))

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]], x=["predicted: don't call", "predicted: call"],
        y=["actual: no", "actual: subscribed"], colorscale="Blues", showscale=False,
        text=[[f"TN<br>{tn}<br>saved call", f"FP<br>{fp}<br>wasted call: -€{cost:.0f}"],
              [f"FN<br>{fn}<br>missed sale", f"TP<br>{tp}<br>sale: +€{value - cost:.0f}"]],
        texttemplate="%{text}", textfont=dict(size=15)))
    cm.update_layout(title=f"Confusion matrix — {LABEL[model]}, {split}, threshold {threshold:.3f}",
                     height=400, yaxis_autorange="reversed")

    pr = profit(y, contact, cost, value)
    everyone = profit(y, np.ones_like(y), cost, value)
    ideal = profit(y, y == 1, cost, value)
    # Profit across all thresholds, with the current one and the val-chosen one marked
    ts = np.linspace(0, 1, 201)
    curve = [profit(y, p >= t, cost, value) for t in ts]
    t_val = summary["results"][model]["threshold_chosen_on_val"]
    fig = go.Figure(go.Scatter(x=ts, y=curve, name="profit", line=dict(color=COLOR[model])))
    fig.add_hline(y=everyone, line_dash="dot", line_color="grey", annotation_text="call everyone")
    fig.add_vline(x=threshold, line_color="black", annotation_text="current")
    fig.add_vline(x=t_val, line_dash="dash", line_color="#c0392b", annotation_text="chosen on val",
                  annotation_position="bottom right")
    fig.add_vline(x=cost / value, line_dash="dot", line_color="#7d3c98", annotation_text="cost/value",
                  annotation_position="top left")
    fig.update_layout(title=f"Campaign profit vs threshold ({split})", xaxis_title="threshold",
                      yaxis_title="profit (€)", height=400)

    md = f"""
### Net campaign profit: **€{pr:,.0f}**  ({pr / len(y):+.2f} € per client in the pool)
| | |
|---|---|
| Clients in pool | {len(y):,} |
| Called | {tp + fp:,} ({(tp + fp) / len(y):.1%}) |
| Precision (subscribed / called) | {tp / max(tp + fp, 1):.1%} |
| Recall (subscribers reached) | {tp / max(tp + fn, 1):.1%} |
| vs. **call everyone** (€{everyone:,.0f}) | **{pr - everyone:+,.0f} €** |
| vs. perfect foresight (€{ideal:,.0f}) | {pr - ideal:+,.0f} € |
| Breakeven probability cost/value | {cost / value:.3f} |
"""
    return cm, fig, md


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
data = summary["data"]
intro = f"""
# Who should the bank call? — term-deposit telemarketing
Logistic regression trained three ways (scikit-learn · manual PyTorch loop · nn.Module + optim) on
**{data['train']['rows']:,}** calls from {data['train']['from']}–{data['train']['to']}; threshold chosen on
**{data['val']['rows']:,}** calls ({data['val']['from']}–{data['val']['to']}); evaluated on
**{data['test']['rows']:,}** unseen calls ({data['test']['from']}–{data['test']['to']}).
All models are loaded from `artifacts/` — nothing is retrained here.
"""

with gr.Blocks(title="Bank telemarketing — Assignment 1") as demo:
    gr.Markdown(intro)
    with gr.Tab("1 · Compare the three models"):
        gr.Dataframe(metrics_table(), label="Same split, same features, same objective")
        gr.Markdown(agreement_md())
        with gr.Row():
            cal_split = gr.Radio(["train", "val", "test"], value="test", label="Split for calibration plot")
        with gr.Row():
            cal_plot = gr.Plot(calibration_plot("test"))
            gr.Plot(agreement_scatter())
        gr.Plot(over_time_plot())
        with gr.Row():
            gr.Plot(loss_plot())
        gr.Plot(coef_plot())
        cal_split.change(calibration_plot, cal_split, cal_plot)

    with gr.Tab("2 · Features & target"):
        gr.Plot(target_plot())
        with gr.Row():
            feat = gr.Dropdown(RAW_CATEGORICAL + RAW_NUMERIC, value="poutcome", label="Raw feature")
            feat_split = gr.Radio(["all", "train", "val", "test"], value="all", label="Rows")
        feat_plot = gr.Plot(feature_plot("poutcome", "all"))
        gr.Markdown("`duration`, `month` and four of the five macro series are shown for context but are "
                    "**not** model inputs — see REPORT.md for why.")
        feat.change(feature_plot, [feat, feat_split], feat_plot)
        feat_split.change(feature_plot, [feat, feat_split], feat_plot)

    with gr.Tab("3 · Threshold & business cost"):
        with gr.Row():
            mdl = gr.Dropdown(MODELS, value="sklearn", label="Model")
            spl = gr.Radio(["val", "test"], value="test", label="Evaluate on")
        with gr.Row():
            thr = gr.Slider(0, 1, value=summary["results"]["sklearn"]["threshold_chosen_on_val"],
                            step=0.005, label="Decision threshold: call if P(subscribe) ≥ threshold")
            cost = gr.Number(value=COST_PER_CONTACT, label="Cost per contacted client (€)")
            value = gr.Number(value=VALUE_PER_SUBSCRIBER, label="Value per subscriber (€)")
        with gr.Row():
            cm_plot = gr.Plot()
            kpis = gr.Markdown()
        profit_plot = gr.Plot()
        inputs = [mdl, spl, thr, cost, value]
        for comp in inputs:
            comp.change(threshold_view, inputs, [cm_plot, profit_plot, kpis])
        demo.load(threshold_view, inputs, [cm_plot, profit_plot, kpis])

if __name__ == "__main__":
    demo.launch()
