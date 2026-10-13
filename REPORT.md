# Assignment 1 Report

- **Name**: Daniel Teixidor
- **Student ID**: 19971
- **Email**: dteixidor.ieu2022@student.ie.edu
- **Group**: BBADBA 5A
## Dataset

**Bank Marketing (additional, full)**: Moro, Cortez & Rita (2014), UCI ML Repository, dataset 222. File:
`data/bank-additional-full.csv` (5.8 MB, `;`-separated). Note: I downloaded it from a GitHub mirror
(`selva86/datasets`) because UCI was unreachable. It has the original shape: 41,188 rows × 21 columns
and 4,640 "yes" answers.

- **One row = one client contacted** in a Portuguese bank's term-deposit telemarketing campaign
  (May 2008 – Nov 2010). Columns are client attributes (age, job, marital, education, default/housing/
  personal-loan status), current-campaign contact info (channel, month, weekday, number of calls,
  call `duration`), history from earlier campaigns (`pdays`, `previous`, `poutcome`) and five monthly
  macro indicators (Euribor 3m, employment variation rate, CPI, consumer confidence, number employed).
- **Target** `y`: did the client subscribe to a term deposit (11.3% yes overall).
- It has more rows than the ~20k guide. The brief says that's not a hard cutoff, the file is well under
  20 MB, and the time split below needs the long history. I kept every row.

**Why this dataset.** Feature engineering isn't trivial here. There's a leakage trap (`duration`), a
sentinel value (`pdays = 999`), many "unknown" levels, collinear macro series, and, most importantly,
a strong change over time. The rows are in chronological order but have **no year column**, so the
pipeline has to rebuild the calendar before it can split by time.

## Business / real-life framing

**Decision supported:** *before* each campaign, the bank decides **which clients go on the call list**.
The model scores every client with P(subscribe). We call a client if that score is ≥ a threshold.

Economics (constants in `pipeline.py`, editable live in the dashboard):

- **Cost per contacted client c = €25.** A contacted client gets about 2.5 calls on average
  (`campaign` mean), at roughly €10 loaded agent cost per call.
- **Value per subscriber V = €80.** That's net margin on a new term deposit, e.g. about €8k average
  deposit × about 1% net interest margin.
- Profit = TP·(V − c) − FP·c. Missed subscribers (FN) earn nothing; skipped non-subscribers (TN) save c.

How this framing drives the pipeline:

1. **Target and features: drop `duration`.** Call length is only known *after* the call, and a call of
   0 seconds is always a "no". The decision is made *before* dialling, so `duration` can't be an input.
   As a diagnostic, `train.py` shows that adding it inflates AUC from 0.786 to **0.930** on a random
   split. That number is useless for this decision.
2. **Split by time, not randomly.** The model is used on *future* campaigns, so evaluation must be on
   later months than training. I rebuild the year from the month sequence: each time the month number
   drops, a new year starts. That gives May 2008 → Nov 2010, with 2 wraps.
   - **train** May 2008 – Oct 2009 (38,601 rows, 8.6% yes)
   - **val** Nov 2009 – Mar 2010 (793 rows, 51.2% yes), used to pick C and the threshold
   - **test** Apr – Nov 2010 (1,794 rows, 51.4% yes), touched once at the end

   The split matters a lot. After 2008 the bank switched from mass calling (thousands of calls a month,
   about 5% success) to much smaller, targeted campaigns (about 50% success), while Euribor fell from
   4.9% to 0.7%. A random split mixes 2010 into training and reports AUC 0.786. The honest
   out-of-time AUC is **0.716**.
3. **Threshold metric = campaign profit, chosen on val.** In theory you call when P ≥ c/V = 0.3125.
   That only holds if probabilities are calibrated. Because of the shift above, they aren't: the model
   predicts about 28% on average for a population that converts at 51%. So I pick the threshold that
   maximises profit on the val set, which is from the *new* regime. The pick is **0.155**. It is not
   the training-set optimum, and it is not the theoretical 0.3125. (On test, 0.3125 would make €21.6k
   against €30.2k at 0.155.)
4. **Ranking metric = ROC AUC** for choosing C. The model's job is to *order* clients, and the threshold
   is set separately afterwards.

## Data preparation & feature engineering

All row-wise transforms are in `pipeline.engineer_features` (no cross-row statistics, so no leakage
across the split). Anything learned from data (scaling, one-hot vocabulary) is a `ColumnTransformer`
**fitted on train only**.

| Feature | What / why |
|---|---|
| `prev_contacted` | 1 if `pdays != 999`. 999 is a "never contacted" sentinel. Used as a raw number it would say "999 days ago", which is nonsense. |
| `pdays_recency` | 1/(1+pdays) for clients with a recorded `pdays`, else 0. Recent previous contact is a stronger signal, and the sentinel maps to 0 instead of a huge number. |
| `log_campaign` | log(1 + calls this campaign), capped at 20. Very right-skewed (max 56), and diminishing effect. |
| `log_previous` | log(1 + earlier contacts). Same reason. |
| `default_unknown` | `default` has only 3 "yes" rows, so it's really "no" vs "unknown" (people who won't say). |
| `euribor3m` | **One** macro indicator instead of five. Euribor, employment variation rate and number employed are correlated > 0.9. Euribor is the rate a deposit competes with. |
| `age_band` | <25 / 25-34 / 35-49 / 50-59 / 60+, one-hot. Subscription is U-shaped in age (students and retirees), which a single linear age term can't capture. |
| `job, marital, education, housing, loan, contact, day_of_week, poutcome` | One-hot with `drop="first"`. "unknown" is kept as its own level because refusing to answer is informative. `education = illiterate` (18 train rows) is pooled into `basic.4y`, and `min_frequency=100` pools any other rare level (here `marital = unknown`, 77 rows). |

Scaling: the three continuous features are standardised. The three 0..1 features are **passed through
unscaled** (see the comparison section for why). That gives 41 columns after encoding.

**Deliberately excluded:** `duration` (leakage, above). `month`: in train it mostly encodes the 2008
mass-campaign calendar (e.g. May = huge volume, low success), not a stable seasonal effect. In my
first prototype, including it didn't help out-of-time. Also excluded: the four other macro series.
`emp.var.rate` and `nr.employed` are near-duplicates of Euribor (r = 0.97, 0.95). CPI and consumer
confidence are less correlated (0.69, 0.28), but they're monthly constants: with only about 18 training
months they mostly act as month-identifiers, and they're the features most likely to be extrapolated
out-of-time. I didn't test them one at a time on val; that's a fair next step.

**What didn't work / what I changed:**

- Keeping all five macro series + `month` was no better on val and less stable than the single-Euribor version.
- Dropping `prev_contacted` because it overlaps with `poutcome = success` (1,373 of its 1,515 rows)
  cost about 0.004 val AUC, so I kept it.
- My first version used `age` and `age²` standardised. The squared term had z-scores up to 22. I
  replaced it with age bands, which are better conditioned and easier to explain.

## Modeling: three implementations, one model

**Model:** L2-regularised logistic regression, since the target is binary. All three implementations
minimise the **same objective**, which is sklearn's divided by C·n:

  J(w, b) = mean BCE(σ(Xw + b), y) + (λ/2)·‖w‖²,  λ = 1/(C·n), intercept not penalised.

- **scikit-learn:** `LogisticRegression(C, lbfgs, tol=1e-10)`.
- **Manual PyTorch:** raw tensors `w`, `b` with `requires_grad`, `loss.backward()`, a hand-written
  `w -= lr * w.grad` under `torch.no_grad()`, then zero the grads. Full-batch gradient descent, 6,000 epochs.
- **Standard PyTorch:** `nn.Module` wrapping `nn.Linear`, `BCEWithLogitsLoss`-equivalent, `torch.optim.Adam`,
  full batch, 3,000 epochs. L2 is added to the loss on the **weight only**. Adam's `weight_decay`
  would also shrink the bias, and with Adam it isn't equivalent to an L2 penalty.
- Everything runs in float64, and both PyTorch models start at w = 0, b = 0 (the problem is convex, so
  the start point only affects speed).

**Tuning (sparingly):**

- C ∈ {0.01, 0.1, 1, 10} was chosen on val AUC. Val AUC is almost flat (0.669–0.672), so ties go to
  the strongest regularisation. **C = 0.01.**
- Learning rates were chosen by the **training objective after 600 epochs**, not by a val metric. The
  learning rate only decides whether we reach the optimum, not which optimum.
  - Manual GD over {0.1, 0.5, 1, 2, 4}: picked 4.0. It's at the edge of the grid but still converges.
    The logistic loss's curvature is at most 0.25·λmax(XᵀX/n), so fairly large steps are stable.
  - Adam over {0.001, 0.01, 0.05, 0.1}: picked 0.1.

**Results (test = Apr–Nov 2010, 1,794 clients; threshold chosen on val):**

| Model | val AUC | test AUC | test avg precision | test log loss | threshold | test profit |
|---|---|---|---|---|---|---|
| scikit-learn | 0.672 | **0.716** | 0.718 | 0.751 | 0.155 | **€30,155** |
| PyTorch manual loop | 0.672 | **0.716** | 0.718 | 0.751 | 0.155 | **€30,155** |
| PyTorch nn.Module + Adam | 0.672 | **0.716** | 0.718 | 0.751 | 0.155 | **€30,155** |
| Baseline: constant = train rate (8.6%) | – | 0.500 | 0.514 | 1.307 | – | – |
| Baseline: call everyone (what the bank did) | – | – | – | – | – | €28,990 |
| Baseline: call no one | – | – | – | – | – | €0 |

**Do the three agree?** Yes, to optimiser tolerance:

- Max |coefficient difference| vs sklearn: **3.9e-6**.
- Intercept difference: 6e-6.
- Max |probability difference| on test: 1.2e-6.
- Identical decisions for 100% of test clients.
- Final objective: 0.264051 for all three.

The dashboard's first tab plots the three probability sets against each other, the three coefficient
sets side by side, and both PyTorch loss curves against the sklearn optimum.

**They didn't agree at first. Here's what I found.**

1. *First run (full one-hot, no `drop`).* Predicted probabilities agreed (max diff 0.0015, same
   decisions), but **coefficients differed by up to 0.43 and intercepts by 1.8–2.3**. Cause: within
   each categorical block the dummies sum to 1, which is exactly the intercept column. So infinitely
   many (w, b) give the same predictions. Only the weak L2 term (C = 10 at that point) picks one, and
   each optimiser had drifted to a different point along that nearly flat valley. Fix:
   `drop="first"`, which makes the model identifiable.
2. *Second run.* Adam now matched sklearn (2e-3), but **manual gradient descent was off by up to 0.22
   in probability** for some test clients. I traced the worst one: a client with `pdays = 0`. The
   standardised `pdays_recency` was **32.6** for that client, and `prev_contacted` was 7.7. Standardising
   a feature that's 0 for 96% of rows creates huge values for the rare non-zeros. That makes the
   problem badly conditioned: plain GD moves very slowly along that direction, while Adam's
   per-parameter step sizes cope. Same story for `education = illiterate` (18 rows, poorly identified
   coefficient). Fixes: pass 0..1 features through unscaled, pool rare levels, replace age² with
   bands. After that, all three match to 1e-6.

Honest caveat: the final C = 0.01 is fairly strong regularisation, which also improves conditioning.
Part of the tight agreement comes from that, not only from the feature fixes. Plain GD is still the
slowest: about 10 s against 0.6 s for lbfgs and 5.6 s for Adam.

**Is the model worth it?** Modestly. On test it beats calling everyone by **€1,165 (+4%)**. It skips
161 non-subscribers at the cost of 52 missed subscribers, while calling 88% of the pool. In the 2010
regime the bank was already calling a pre-selected list where half of people say yes. With these
economics, most clients are worth calling and there's little left for a model to filter. The value
would be much larger in the 2008-style regime (5% base rate), and the profit curve in the dashboard
shows how quickly this changes if c/V changes.

## Limitations & next steps

- **Calibration under the regime shift.** Mean predicted 28% against 51% actual on test (see the
  calibration and over-time plots). Ranking survives, probabilities don't. *Fix:* re-estimate the
  intercept on the most recent months (prior-shift correction / Platt scaling on val). Even better,
  retrain monthly on a rolling window, which is how a bank would actually run this.
- **Train is dominated by 2008.** 2008 is 27.7k of the 38.6k training rows, and its mass-calling
  population doesn't look like 2010's. *Fix:* recency weights in the loss (same weights in all three
  implementations), or train only on post-2009 data once there's enough of it.
- **Small val set (793 rows).** C and the threshold are noisy choices: val AUC differs by < 0.003
  across C, and the profit curve is flat between thresholds 0.12 and 0.17. *Fix:* rolling-origin
  validation over several months, choosing the threshold that's best on average.
- **The economics are assumptions.** c and V are illustrative, not the bank's numbers. That's why
  they're live inputs in the dashboard. *Fix:* get real cost per call and deposit margins. V should
  also depend on the client (deposit size), which would make this a value-weighted decision.
- **Selection bias.** We only see clients the bank chose to call, so the model learns who subscribes
  *among those called*, not across the whole customer base. *Fix:* a small randomised holdout of
  called and not-called clients.
- **Linear model limits.** Interactions (e.g. Euribor × age band, job × previous outcome) aren't
  modelled. That's acceptable for this assignment; a gradient-boosted model would be the natural
  comparison.
- **Dataset provenance.** The file came from a GitHub mirror, not UCI directly. I checked shape and
  class counts against the published description, but not a checksum.

## Generative AI use disclosure

I used **Claude (Anthropic)** as instructed for this assignment. It helped me:

- choose and frame the dataset;
- write `pipeline.py`, `train.py` and `app.py`;
- investigate the three-way disagreement (the identifiability and conditioning issues above);
- draft this report.

I reviewed and ran all of the code, and I checked every number in this report against the
`artifacts/metrics.json` produced by `uv run python main.py train`.
