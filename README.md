# Assignment 1

Pick a tabular dataset, frame a realistic business problem it would serve,
and train the **same** simple model (linear regression for a continuous
target, logistic regression for a binary one) three ways: scikit-learn, a
from-scratch PyTorch loop (Session 5's manual approach), and the standard
`torch.nn.Module` + `torch.optim` workflow. Compare all three and ship the
result behind an interactive Gradio dashboard.

The point isn't the dataset or getting the best score. It's showing you
understand every step well enough to explain it later, without your code
in front of you, in the written comprehension check (see Grading below).
Pick a dataset interesting enough that feature engineering actually
matters, and be honest in REPORT.md about what didn't work and why.

## The task

1. **Pick a dataset.** Tabular, with a clear regression or classification
   target (or one you can build non-trivially from the data), interesting
   enough that feature engineering isn't trivial. Large enough to train on,
   small enough to commit to this repo and retrain from scratch: a rough
   guide is under ~20k rows / ~20MB, not a strict cutoff.
2. **Frame the business problem** *before* writing any pipeline code: what
   hypothetical (or real) decision does this model support? Write it in
   REPORT.md, and make sure it drives later decisions: does target
   definition need care, would a random split leak the future into
   training, which metric should set the decision threshold?
3. **Preprocess, engineer features, and split** with numpy/pandas/sklearn,
   consistent with what you decided in step 2.
4. **Train the same model three ways** on the same split:
   - scikit-learn linear or logistic regression
   - manual PyTorch loop like we saw in the PyTorch introduction notebook
   - standard PyTorch workflow

   Compare all three against each other and against a naive baseline. Save
   each trained model to disk: training is the only stage that trains
   anything, and the Gradio app must load these saved models, never
   retrain them.
5. **Tune sparingly.** The only hyperparameters worth touching are
   regularization strength and, for the two PyTorch versions, the learning
   rate. Put your effort into features, not a grid search.
6. **Write REPORT.md** (skeleton already in this repo): dataset, business
   framing, your process, a three-method comparison table, and honest
   limitations.
7. **Build the Gradio dashboard** from your trained models. It should
   never retrain anything at startup. At minimum, let you compare the
   three models with a prediction vs. actual plot, see feature/target
   distributions, and, for classification, move a threshold slider to
   watch the confusion matrix and a business-cost number change.
8. **Submit**: see Submission below.

Everything else (how you structure your code, what you name things beyond
what `main.py` requires, how you organize your pipeline) is your call to
make and be able to explain.

## Setup

Same environment workflow as Session 2's environment check.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS
```
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

Fork this repo (https://github.com/ami232/daiist-assignment-1), then clone
**your fork**:

```bash
git clone https://github.com/<your-github-username>/<your-fork-name>.git
cd <your-fork-name>
uv sync
```

> If `uv sync` can't install PyTorch locally (older Intel Macs), use GitHub
> Codespaces on your fork instead: same environment, runs in the browser.

### Adding dependencies

The starting dependencies (numpy, pandas, scikit-learn, torch, gradio,
plotly, matplotlib, ...) cover everything the assignment needs, but if you
want something else:

```bash
uv add <package-name>
```

This updates `pyproject.toml` and `uv.lock`. Commit both so `uv sync`
reproduces your exact environment for anyone else, including CI (below).

## Running your work

```bash
uv run python main.py train    # runs train.py, or train.ipynb if you used a notebook
uv run python main.py app      # runs app.py, or app.ipynb if you used a notebook, opens in your browser
```

`main.py` looks for `<stage>.py` first, then `<stage>.ipynb`, so name your
training code `train.py` or `train.ipynb`, and your app `app.py` or
`app.ipynb`, at the repo root. Pick whichever format fits each stage (a
notebook for one and a script for the other is fine). Beyond that naming
requirement, how you build them is up to you, but each must run start to
finish with no manual steps.

A GitHub Actions workflow (`.github/workflows/verify.yml`) runs both
commands automatically on every push and pull request. It's a sanity check
that your submission runs, not a grading mechanism, and it fails if
`main.py train` errors or `main.py app` doesn't come up and respond within
its startup window.

## Submission

1. Push your finished branch to your fork.
2. Open a pull request from your fork to the original repo:
   https://github.com/ami232/daiist-assignment-1
3. Submit the PR link on Blackboard.
4. Also upload a zipped copy of your repo to Blackboard, as a backup in
   case your fork or the PR becomes unavailable.

## Grading

**Hard requirement:** your training pipeline and Gradio app must each run
end-to-end via `main.py` with no manual intervention. A submission that
fails this can't pass the assignment, regardless of everything else below.

| Component | Weight | What it checks |
|---|---|---|
| Dataset & business framing | 20% | Is the framing sensible, and does it actually drive concrete pipeline decisions (split strategy, threshold metric, etc.) rather than just narrating them? |
| Feature engineering & preprocessing | 30% | Quality and justification of what you built, not just its presence. This is the largest component, since most of your effort should go here. |
| Three-method comparison | 30% | Do scikit-learn / manual PyTorch / standard PyTorch agree on the same split? If not, is that investigated and explained honestly rather than hidden? |
| Gradio dashboard | 20% | All required views present, working off your pipeline's real artifacts, nothing retrained at startup. |
| **× Written Comprehension Check** | **0–100%** | Multiplies the subtotal |

REPORT.md isn't graded as its own row: it's where the other four
components' content lives, so its quality is already captured by them. The
four weighted rows above sum to 100% and form your subtotal; the Written
Comprehension Check then multiplies that subtotal. A strong submission
paired with a weak comprehension-check score gets scaled down accordingly,
since the check verifies the understanding this brief keeps asking for.

Coefficients above are a starting proposal sized to relative workload
(feature engineering and the three-method comparison carry the most work),
not settled policy. Expect them to be confirmed before the deadline.

## Generative AI use

Per the syllabus AI Policy: disclosed AI use is fine and must be stated in
REPORT.md's disclosure section. Within that policy, here's how it applies
to this assignment specifically:

- **Fine to use AI for**: boilerplate and common operations, like loading
  a dataset, saving a trained model, and especially plotting and
  presenting results in the Gradio dashboard.
- **Use your own judgement for**: the decisions that are the point of this
  assignment, like which features to design, how to frame the business
  problem, and the implications of your design choices. AI can write the
  code for a decision, but the decision itself has to be yours.
- **REPORT.md**: the ideas and findings in it must be your own. AI may
  help with formatting, not with generating the analysis or conclusions.

None of this changes what's expected of you: you have to be able to
explain every decision in your submission as if you made it yourself,
because you did. Using a tool to help write it doesn't transfer the
understanding requirement to the tool.
