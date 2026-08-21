# Loan Default Prediction

A mortgage loan default prediction system, built as a working API rather than a
notebook. It takes a loan application and returns the probability the borrower
defaults, a risk grade, the expected loss in currency terms, an approve /
review / decline recommendation, and the reasons behind it.

Dataset: 148,670 mortgage applications.

[![CI](https://github.com/your-username/loan-default-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/loan-default-prediction/actions/workflows/ci.yml)

---

## Start here: the model used to score 100%

The first working version of this project reported a ROC-AUC of **1.0000**.
Perfect. Logistic regression, random forest and XGBoost all scored exactly 1.0
on a held-out test set of 29,734 loans, with a confusion matrix that had zeros
off the diagonal.

That is not a good model. It is a broken one, and finding out why is the most
useful thing in this repository.

The dataset has a column called `Interest_rate_spread`, which is blank for about
a quarter of the rows. It turns out those blanks are not random:

```
Interest_rate_spread.isna()  ==  Status     on 148,670 of 148,670 rows
```

Every single row. The column is only populated for loans that were actually
originated and priced, so its absence is a consequence of the loan defaulting,
not a predictor of it. The original feature engineering built a
`_missing` indicator from that column and handed it to the model, which is to
say it handed the model the answer.

Two of those indicators carried **97.5% of the model's feature importance**.
Eighty-two of the eighty-seven features had importance of exactly zero —
including credit score, LTV, debt-to-income and income. It was not a credit
model. It was a lookup table on one NaN pattern.

Removing the leakage properly (there is a second, subtler layer of it) gives a
model that scores **0.8244**. That is the honest number this project reports,
and the full investigation is in
[docs/LEAKAGE_INVESTIGATION.md](docs/LEAKAGE_INVESTIGATION.md).

Training now refuses to save a model that scores above 0.95, and there is a test
that fails the build if one appears.

---

## Results

Held-out test set of 24,910 loans, 16.32% default rate.

| Metric | Value |
|---|---|
| ROC-AUC | 0.8244 (95% CI 0.8166–0.8320) |
| Gini | 0.6489 |
| PR-AUC | 0.6104, against a 0.1632 baseline |
| KS | 0.4892 |
| Brier score | 0.0933 |
| Max calibration error | 0.0136 |

Risk grades rank-order correctly, which is the thing that actually matters if
you intend to lend against them:

| Grade | Loans | Mean predicted PD | Observed default rate |
|---|---|---|---|
| A | 88 | 1.44% | 0.00% |
| B | 5,898 | 3.48% | 3.42% |
| C | 5,961 | 6.79% | 5.89% |
| D | 8,318 | 13.21% | 13.54% |
| E | 1,784 | 26.50% | 26.18% |
| F | 1,077 | 43.38% | 42.34% |
| G | 1,784 | 82.43% | 82.06% |

Predicted and observed track each other closely all the way up. That is
calibration, and it is what lets the PD be multiplied by an exposure to get a
loss figure in pounds rather than just used as a ranking.

Full breakdown, including performance by region and loan purpose, is in
[reports/validation_report.md](reports/validation_report.md).

---

## Try it

```bash
git clone https://github.com/your-username/loan-default-prediction.git
cd loan-default-prediction
make install
make serve
```

Then open <http://localhost:8000/docs> and hit `POST /v1/risk/assess` with the
example payload that is already filled in.

The trained model is committed to the repository, so this works immediately.
You do not need the dataset and you do not need to train anything.

A request looks like a loan application. The response looks like this:

```json
{
  "probability_of_default": 0.0578,
  "risk_grade": "C",
  "grade_description": "Modest risk",
  "loss": {
    "pd": 0.0578,
    "lgd": 0.100,
    "ead": 296500.0,
    "expected_loss": 1713.57,
    "lgd_method": "collateral_proxy"
  },
  "decision": {
    "decision": "APPROVE",
    "reason": "PD 5.78% is below the break-even PD of 58.33% by more than the 5% review band.",
    "break_even_pd": 0.5833
  },
  "explanation": {
    "risk_drivers": [
      {"label": "Application channel", "value": "to_inst", "contribution": 0.2079},
      {"label": "Loan purpose", "value": "p3", "contribution": 0.1189}
    ],
    "risk_reducers": [
      {"label": "Loan-to-value ratio", "value": 70.9, "contribution": -0.5478},
      {"label": "Debt-to-income ratio", "value": 39.0, "contribution": -0.3811}
    ]
  },
  "model_version": "v20260821T011832Z",
  "assumptions_version": "1.0.0",
  "request_id": "3f8a1c2e-5b7d-4a91-8e6f-1d2c3b4a5e6f"
}
```

---

## How the decision gets made

The model produces a probability of default. Turning that into a lending
decision takes three more steps, and each one is an assumption I had to make
explicit rather than a number I could estimate from the data.

**Exposure at default** is the loan amount. These are fully-drawn term loans
being scored at origination, so there is no undrawn commitment to convert and
credit conversion factor modelling does not apply.

**Loss given default** is a collateral proxy, not a model. The dataset has no
recovery cash flows, no workout costs and no resolution times, so an LGD model
cannot be fitted from it — claiming otherwise would be inventing a capability.
What it does have is property values, and 99.98% of the book is secured on
residential property. So:

```
LGD = clip(1 − property_value × (1 − haircut) × (1 − costs) / EAD, floor, 1)
```

with a 25% distressed-sale haircut, 10% workout costs and a 10% floor. Those
three numbers live in [config/risk_policy.yaml](config/risk_policy.yaml) and are
returned with every response, so nobody has to guess what the loss figure rests
on.

**Expected loss** is then just `PD × LGD × EAD`.

**The approve/decline threshold** is derived rather than picked. It solves for
the PD at which expected margin equals expected loss:

```
(1 − PD) · EAD · margin · life  =  PD · LGD · EAD
⇒  PD* = revenue_rate / (LGD + revenue_rate)
```

Because it depends on LGD, a well-secured borrower is tolerated at a much higher
PD than a thinly-secured one, which falls out of the arithmetic rather than
being coded as a rule:

| LGD | Break-even PD |
|---|---|
| 0.10 | 58.3% |
| 0.25 | 35.9% |
| 0.45 | 23.7% |
| 1.00 | 12.3% |

The earlier version of this project used hardcoded bands of 0.3 and 0.6 with no
stated reason. This is the same idea with the reasoning put back in.

---

## Fair lending

Gender and age are not model inputs, and the API will not accept them — send
either field and the request is rejected. The service cannot use what it never
receives, which is a stronger position than filtering after the fact.

This costs **4.6 basis points of ROC-AUC** (0.8258 with them, 0.8253 without).
I measured it rather than assuming it, because "we removed it and performance
was fine" is a claim someone will ask you to back up.

Excluding the fields is not the end of it, since a model can still produce
uneven outcomes through correlated proxies. Approval rates by group are measured
in [reports/fairness_analysis.md](reports/fairness_analysis.md); the lowest
adverse impact ratio is 0.859, above the four-fifths threshold conventionally
used as a review trigger.

Region *is* a model input, and coarse geography is a plausible proxy. It is kept
because the levels here are very broad and it carries real signal, but it is
flagged and monitored rather than waved through.

To be clear: this is a design decision motivated by fair-lending principles. It
is not a claim of regulatory compliance, which would need legal review this
project has not had.

---

## What's in here

```
src/loan_default/
  data/          contract, loading, quality profiling
  features/      one feature engineering implementation, shared by train and serve
  models/        training, evaluation, calibration, SHAP, segments, registry
  risk/          grades, LGD/EAD/EL, decision policy, portfolio, stress
  monitoring/    PSI and calibration drift
  api/           routers, dependencies, middleware, error handlers
config/          feature contract, credit policy, stress scenarios
tests/           unit, api, model, integration
notebooks/       EDA, leakage investigation, model development
reports/         generated validation, fairness and calibration output
artifacts/       versioned model, metadata and accepted metrics
```

The three YAML files in `config/` are worth a look. Every assumption that could
change a lending decision is in them rather than buried in code, which makes
them reviewable by someone who does not read Python.

### The API

| Method | Endpoint | |
|---|---|---|
| POST | `/v1/risk/assess` | Score one application |
| POST | `/v1/risk/batch` | Score many, with portfolio aggregates |
| GET | `/v1/model/metadata` | Version, data hash, features, limitations |
| GET | `/v1/model/metrics` | Accepted performance metrics |
| GET | `/v1/model/policy` | Grade scale and active assumptions |
| POST | `/v1/portfolio/stress-test` | Scenario analysis |
| GET | `/v1/portfolio/summary` | Exposure, expected loss, concentration |
| GET | `/health/live` | Process is up |
| GET | `/health/ready` | Model loaded and scoring |

Readiness actually scores a canary record at startup. Checking that a file
exists on disk tells you nothing about whether it loads.

Batch scoring validates row by row. One malformed application in a file of five
thousand returns an error for that row and scores the rest, which is the
behaviour you want at 2am when a nightly job hits a bad record.

---

## Stress testing

```bash
curl -X POST http://localhost:8000/v1/portfolio/stress-test \
  -H 'Content-Type: application/json' \
  -d '{"sample_size": 5000}'
```

Three scenarios, defined in
[config/stress_scenarios.yaml](config/stress_scenarios.yaml). Income falls,
debt burden rises, property values drop. A property shock recomputes LTV, so it
raises PD and the collateral LGD together, which is the channel that matters in
a mortgage book.

This is sensitivity analysis under stated assumptions. It is **not** CCAR or
DFAST and I have not called it that anywhere. The dataset has no macroeconomic
variables and no time dimension at all — `year` is 2019 for every row — so there
is nothing to condition a macro path on.

More importantly, the results come with a confidence rating, because the
headline numbers are large and should not be taken at face value:

| Scenario | Weighted PD | Change in EL | Book outside observed range | Confidence |
|---|---|---|---|---|
| Base | 0.1478 | — | 0.9% | High |
| Moderate | 0.5218 | +509% | 23.1% | Low |
| Severe | 0.8262 | +1175% | 58.7% | Low |

The reason the moderate scenario looks so violent is worth understanding. A 25%
income shock only moves mean PD by a factor of 1.24. A 15% *property* shock
moves it by 2.64, because it drives LTV up — and the training data contained
only 1.2% of loans above 100% LTV. A modest-sounding house price fall pushes a
quarter of the book into a region the model has barely seen. So the model
extrapolates, and the endpoint says so rather than quietly reporting a number.

---

## Testing

```bash
make test
```

180 tests, 85% coverage. The ones that earn their place:

| Test | What it stops |
|---|---|
| `test_no_leakage.py` | Fails the build if test AUC exceeds 0.95 |
| `test_schema_roundtrip.py` | Real dataset rows must validate against the API |
| `test_train_serve_parity.py` | Training and serving must agree to 1e-12 |
| `test_training.py` | Two runs with the same seed produce identical metrics |
| `test_grades_and_stress.py` | Grade monotonicity; shocks move risk the right way |
| `test_data_quality.py` | Duplicate IDs, impossible values, leaky missingness |

The schema test exists because the earlier API rejected 148,111 of the 148,670
records it was trained on — 99.6% — while accepting eleven category values that
appear nowhere in the data. The request schema is now generated from the dataset
rather than typed by hand, and the test checks that real rows still pass.

CI runs lint, mypy, the test suite, and a deploy smoke test that installs
exactly what Railway installs and scores a live request.

---

## Deployment

```
GitHub → Railway → Python 3.11 → FastAPI → model
```

Push to GitHub, point Railway at the repo, and it builds from
[nixpacks.toml](nixpacks.toml) and starts the app from
[railway.toml](railway.toml). Traffic is only routed once `/health/ready`
returns 200, so a deploy with a broken artifact fails the health check instead
of serving 500s to users.

No environment variables are required; everything has a working default. Full
walkthrough in [docs/RAILWAY_DEPLOYMENT.md](docs/RAILWAY_DEPLOYMENT.md).

---

## Limitations

Written out properly in [MODEL_CARD.md](MODEL_CARD.md). The ones that matter
most:

**Credit score is noise in this dataset.** Univariate AUC of 0.5030, with a flat
default rate across all ten deciles. In real bureau data this would be the
single strongest predictor. Here it appears to be randomly generated, and its
range (500–900) is not FICO's. Do not read anything into its SHAP value. There
is a test asserting this stays true, so if the data changes the model card gets
revisited.

**Only originated loans are in the data.** Everyone the original lender declined
is missing, so the model is trained on an accepted population but would score
the full through-the-door population. Correcting for that properly needs
reject-inference data this dataset does not contain.

**Complete-case training.** 16.2% of rows are dropped because their missingness
was itself predictive. Those rows are not missing at random, so the model is
calibrated to a 16.3% default rate population rather than the full book's 24.6%.

**No time dimension**, so no out-of-time validation, no vintage analysis, and no
real production drift to observe. The drift detectors are demonstrated against a
deliberately perturbed sample, clearly labelled as a simulation.

**Grade A is thin** — 88 loans, zero observed defaults. Zero defaults out of 88
supports no PD estimate worth quoting.

This is a portfolio project. It has not been through model validation or
fair-lending review and is not fit for real lending decisions.

---

## Built with

Python 3.11, FastAPI, Pydantic, Pandera, scikit-learn, XGBoost, SHAP, pytest,
Ruff, mypy, GitHub Actions, Railway.

MIT licensed. See [LICENSE](LICENSE).

**Abubakar Sadiq Muhammad**
