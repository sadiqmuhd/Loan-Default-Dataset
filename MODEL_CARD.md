# Model Card — Mortgage Probability of Default

| | |
|---|---|
| **Model version** | `v20260821T011832Z` |
| **Model type** | XGBoost classifier + isotonic calibration |
| **Task** | Binary probability of default at loan origination |
| **Owner** | Abubakar Sadiq Muhammad |
| **Training data SHA-256** | `4234b122f463ff4d563de600ade5ec347a9ab8f02cfc204535f7c0d4929bfe70` |
| **Status** | Portfolio / demonstration project. **Not approved for production lending.** |

---

## 1. Intended use

**In scope.** Estimating the probability that a residential mortgage application
will default, at the point of origination, to support an approve / refer /
decline recommendation and a portfolio expected-loss calculation.

**Out of scope.**

- Any real lending decision. This model is trained on a public dataset with
  known quality problems (§6) and has not been through model validation,
  independent review, or fair-lending legal sign-off.
- Pricing. The economics in `config/risk_policy.yaml` are illustrative.
- Non-mortgage lending. 99.98% of the training book is secured on residential
  property; the model has never seen unsecured exposure.
- Behavioural or through-the-life scoring. This is an at-origination model and
  the data has no time dimension.

---

## 2. What is modelled and what is assumed

This distinction is the single most important thing on this card.

| Component | Status | Basis |
|---|---|---|
| **PD** | **Modelled** | XGBoost on 124,547 loans, isotonically calibrated |
| **LGD** | **Assumed** | Collateral proxy under stated haircuts — *not an estimated model* |
| **EAD** | **Assumed** | Origination amount (fully-drawn term loan) |
| **EL** | **Derived** | `PD × LGD × EAD` |

The dataset contains **no recovery cash flows, no workout costs, no
time-to-resolution and no balance history**, so LGD and EAD cannot be estimated
from it. Presenting the LGD proxy as an LGD model would be fabrication. Every
assumption is externalised in `config/risk_policy.yaml` and echoed in the
`assumptions` block of every API response that quotes a loss figure.

**LGD proxy formula**

```
LGD = clip(1 − property_value × (1 − haircut) × (1 − cost) / EAD, floor, 1)
```

with a 25% distressed-sale haircut, 10% workout cost, and a 10% downturn floor.
Where `property_value` is unavailable, a flat 45% fallback applies.

**Why CCF modelling is absent:** these are fully-drawn term loans. There is no
undrawn commitment to convert, so a credit conversion factor is not applicable.

---

## 3. Training data

| | |
|---|---|
| Source | `Loan_Default.csv`, 148,670 records |
| Rows used | **124,547** after filters |
| Rows dropped | 24,123 (16.2%) |
| Default rate | **16.32%** |
| Vintage | 2019 — **a single cross-section** |

### Exclusions and why

**Target leakage — removed.** Measured across all 148,670 rows:

| Column | Agreement of `.isna()` with `Status` |
|---|---|
| `Interest_rate_spread` | **100.000%** (148,670 / 148,670) |
| `rate_of_interest` | 99.865% |
| `Upfront_charges` | 97.774% |

These fields are populated only for originated, performing loans, so their
presence is an *outcome* of the event being predicted. An earlier version of this
project engineered missingness indicators from exactly these columns and scored
ROC-AUC 1.0000. See `docs/LEAKAGE_INVESTIGATION.md`.

**Second-order leakage — controlled by complete-case filtering.** Missingness of
`property_value`, `LTV` and `dtir1` also predicts default: a model using *only*
those three missingness flags scores ROC-AUC **0.7155**. Median imputation leaves
a recoverable signature at the imputed value, so rows missing them are dropped
instead. This is the 16.2% of rows removed above.

**Protected characteristics — removed.** `Gender` and `age` are excluded as a
fair-lending safeguard, motivated by the principles behind ECOA / Regulation B.
This is a model-design decision, not a claim of regulatory compliance. Measured
cost: **4.6 basis points of ROC-AUC** (0.8258 → 0.8253).

**Identifiers — removed.** `ID` (unique per row) and `year` (constant).

---

## 4. Performance

Held-out test set, n = 24,910, base rate 16.32%.

| Metric | Calibrated | Uncalibrated |
|---|---|---|
| **ROC-AUC** | **0.8244** *(95% CI 0.8166–0.8320)* | 0.8249 |
| **Gini** | 0.6489 | 0.6498 |
| **PR-AUC** | **0.6104** *(95% CI 0.5955–0.6241)* | 0.6260 |
| **KS** | 0.4892 | 0.4901 |
| Brier score | 0.0933 | 0.0932 |
| Log loss | 0.3188 | 0.3183 |
| Mean predicted PD | 0.1654 | 0.1640 |
| **Mean calibration error** | **0.0060** | 0.0088 |
| **Max calibration error** | **0.0136** | 0.0309 |

PR-AUC of 0.6104 against a 0.1632 base rate is a **3.7× lift**.

**On calibration.** Both candidate calibrators were fitted on the held-out
calibration slice and measured on the test set:

| Method | Brier | Mean cal. error | Max cal. error | ROC-AUC |
|---|---|---|---|---|
| Uncalibrated | 0.09324 | 0.00884 | 0.03091 | 0.8249 |
| **Isotonic (selected)** | 0.09333 | **0.00598** | **0.01355** | 0.8244 |
| Platt (sigmoid) | 0.09382 | 0.02360 | 0.03612 | 0.8249 |

Isotonic roughly halves calibration error. Platt scaling makes it *worse than
doing nothing*, because its sigmoid assumption does not fit this model's score
distribution.

Selection is on calibration error rather than Brier score. Brier mixes
discrimination and calibration together, so it barely moves when only the latter
improves — it would have hidden the difference being measured. Since the PD is
multiplied by LGD and EAD to produce a currency figure, being right in absolute
terms matters more here than a marginal gain in ranking.

### Model selection

Chosen on 5-fold cross-validated PR-AUC:

| Candidate | CV PR-AUC |
|---|---|
| Logistic regression | 0.3897 |
| Random forest | 0.5596 |
| **XGBoost** | **0.6191** |

`scale_pos_weight` is deliberately **not** set. It distorts the predicted
probability distribution, which is incoherent with serving the output as a
probability of default. Imbalance is handled by metric choice and post-hoc
calibration.

### Rank ordering

Observed default rate is monotonic across grades (asserted by
`tests/model/test_grades_and_stress.py`):

| Grade | n | Share | Mean PD | Observed default rate |
|---|---|---|---|---|
| A | 88 | 0.4% | 0.0144 | 0.0000 |
| B | 5,898 | 23.7% | 0.0348 | 0.0342 |
| C | 5,961 | 23.9% | 0.0679 | 0.0589 |
| D | 8,318 | 33.4% | 0.1321 | 0.1354 |
| E | 1,784 | 7.2% | 0.2650 | 0.2618 |
| F | 1,077 | 4.3% | 0.4338 | 0.4234 |
| G | 1,784 | 7.2% | 0.8243 | 0.8206 |

---

## 5. Fair lending

`Gender` and `age` are not model inputs and are **not accepted by the API at
all** — the request schema rejects them, so the service cannot receive a
protected characteristic even by accident. This is a design decision taken on
fair-lending grounds; it is not a compliance assessment, which would require
legal review this project has not had.

Outcomes are nonetheless measured across groups, because exclusion does not rule
out disparate impact through correlated proxies. Adverse impact ratios against
the four-fifths threshold (0.80):

| Attribute | Lowest group | Adverse impact ratio | Flagged |
|---|---|---|---|
| Gender | Sex Not Available | 0.859 | No |
| Age | `<25` | 0.869 | No |
| Region | North-East | 0.861 | No |

No group falls below the 0.80 threshold. Predicted PD tracks observed default
rate closely within every group, so the model is not systematically
mis-estimating any of them.

**`Region` is a model input and is a known proxy risk.** Coarse geography can
encode redlining. It is retained because the levels here are very coarse
(four regions), it carries genuine risk signal, and it is monitored in the
fairness report — but this is a judgement call that a real model risk function
would need to review.

Regenerate with `make report`.

### Performance by segment

A single headline AUC can hide a segment where the model does not discriminate.
Measured across region, loan purpose, occupancy type and credit bureau, every
segment above 200 loans clears a 0.65 ROC-AUC floor and a 3pp calibration
tolerance. The weakest is `loan_purpose = p2` (ROC-AUC 0.734 on 574 loans).
Full breakdown in `reports/validation_report.md`.

---

## 6. Limitations

**1. `Credit_Score` is not predictive and appears to be synthetic.** Univariate
ROC-AUC **0.5030**, with a flat default rate across all ten deciles
(0.166, 0.163, 0.163, 0.158, 0.161, 0.156, 0.164, 0.164, 0.168, 0.170). In a
real bureau dataset this would be the single strongest predictor. Do not
interpret its SHAP value. Asserted by `test_credit_score_is_non_predictive_noise`.

**2. No time dimension.** `year` is 2019 for every row. Therefore: no
out-of-time validation, no vintage or cohort analysis, no through-the-cycle
versus point-in-time distinction, no macroeconomic conditioning, and no genuine
production drift to observe.

**3. Stress results are extrapolations, and the collateral channel dominates
them.** Measured single-variable sensitivity of mean PD: income −25% gives
1.24×, dtir1 +35% gives 2.62×, but property value −15% gives **2.64×** and −30%
gives **4.46×**. The property channel is amplified through LTV, and the training
data held only **1.2% of loans above LTV 100** (p99 = 102.4). A −15% property
shock moves 22.6% of the book beyond that point, and −30% moves 58.1%. The model
therefore extrapolates heavily under stress. Each scenario reports an
`extrapolation` block with the share of the book outside the observed range and a
confidence rating; the moderate and severe scenarios are both rated **low
confidence** and should be read as directional. These are **not CCAR or DFAST**.

**4. No interest-rate sensitivity.** The rate variables are excluded as leakage,
so the model cannot be stressed on rates.

**5. Only originated loans are observed (through-the-door bias).** The dataset
contains loans that were approved and funded. Every applicant the original
lender declined is absent, so the model is fitted on an accepted population but
would, in use, score the full through-the-door population — including applicants
that the policy generating this data would have rejected. Its behaviour on that
unobserved region is unknown.

This is what reject inference exists to address, and it cannot be done here:
correcting for it requires data on declined applicants, which this dataset does
not contain. A production deployment would need either that data or a monitored
champion/challenger arrangement to learn the missing region safely. It also
means the approval rates in the fairness analysis describe outcomes on an
already-filtered population, so any bias in the original credit policy is
invisible to this analysis.

**6. Complete-case bias.** The 16.2% of rows dropped are not missing at random —
their missingness is precisely what predicted default. The model is therefore
calibrated to the complete-case population (16.3% default rate), not to the full
book (24.6%). A production system would need an explicit policy for applications
with missing collateral data.

**7. `Status` has no stated observation window.** It is treated as a 12-month
default flag; the dataset does not confirm this.

**8. `income` is interpreted as monthly.** Median income 5,760 against a median
loan of 296,500. Read as annual this implies a 52× loan-to-income ratio, which is
not a real mortgage; read as monthly it gives 4.36×, which is. Cross-checked
against a reported median DTI of 39% versus an implied principal-only burden of
15.9%.

---

## 7. Explainability

SHAP `TreeExplainer` produces per-decision reason codes, required for adverse
action notices under ECOA (12 CFR 1002.9).

**Important caveat, surfaced in every API response:** SHAP is computed on the
**uncalibrated** model score in log-odds space. Isotonic calibration is
monotonic, so the sign and ranking of contributions carry over to the calibrated
PD — but the magnitudes are **not** percentage-point contributions to PD.

---

## 8. Governance and reproducibility

- Every artifact records model version, training timestamp, training data
  SHA-256, seed, full feature contract, exclusions, hyperparameters, and library
  versions. Retrievable at `GET /v1/model/metadata`.
- Every assessment response carries `model_version` and `request_id`, and every
  decision is logged as a structured JSON audit record.
- Loading an artifact warns loudly if runtime library versions differ from the
  versions it was trained under.
- All dependencies pinned; `pyproject.toml` and `requirements.txt` are asserted
  to agree by `tests/unit/test_dependency_pins.py`.
- Training fails with `LeakageGuardError` if test ROC-AUC exceeds 0.95.

## 9. Monitoring

PSI on features and on the predicted PD distribution, calibration drift
(observed versus expected by decile), and contract-based data-quality checks.

Because the dataset has no time axis, there is no real drift to observe. The
detectors are demonstrated against a **deliberately perturbed sample, clearly
labelled as a simulation** — the detectors themselves are production-shaped and
work unchanged against a real scoring window.

---

## 10. Maintenance

| | |
|---|---|
| Retraining | `make train` — writes a new versioned artifact |
| Review triggers | PSI > 0.25 on any input; calibration gap > 2pp; AUC below 0.75 |
| Contact | Repository issues |
