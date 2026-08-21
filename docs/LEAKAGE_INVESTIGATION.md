# Target Leakage Investigation

> An earlier version of this project achieved **ROC-AUC 1.0000** on a held-out
> test set. This document is the investigation into why, what was wrong, and
> what the honest performance turned out to be.
>
> The original notebook is preserved unmodified at
> `notebooks/archive/original_EDA_and_Modeling_LEAKING.ipynb` as evidence.

---

## 1. The symptom

The archived notebook's committed output, cell 25:

```
                     Accuracy  Precision  Recall  F1 Score  ROC-AUC
Logistic Regression       1.0        1.0     1.0       1.0      1.0
Random Forest             1.0        1.0     1.0       1.0      1.0
XGBoost                   1.0        1.0     1.0       1.0      1.0

Confusion Matrix:
[[22406     0]
 [    0  7328]]
```

A perfectly diagonal confusion matrix, on 29,734 held-out rows, from three
different model families.

**Logistic regression reaching AUC 1.0 is the tell.** A linear model in the
original feature space cannot separate a real credit population perfectly. When
a linear model and a tree ensemble both hit exactly 1.0, the cause is almost
never the model — it is the data.

---

## 2. Locating the leak

The hypothesis was that some feature was a transformation of the label. Testing
every column whose missingness pattern could carry information:

```python
for col in df.columns:
    agreement = (df[col].isna().astype(int) == df["Status"]).mean()
```

| Column | % missing | P(default \| missing) | P(default \| present) | Agreement with `Status` |
|---|---|---|---|---|
| **`Interest_rate_spread`** | 24.64% | **1.0000** | 0.0000 | **100.000%** |
| `rate_of_interest` | 24.51% | 1.0000 | 0.0018 | 99.865% |
| `Upfront_charges` | 26.66% | 0.9204 | 0.0014 | 97.774% |
| `LTV` | 10.16% | 0.9999 | 0.1613 | 85.508% |
| `property_value` | 10.16% | 0.9999 | 0.1613 | 85.508% |
| `dtir1` | 16.22% | 0.6762 | 0.1632 | 81.072% |

**`Interest_rate_spread.isna()` is an exact copy of the target — 148,670 of
148,670 rows, no exceptions.**

Cross-tabulated:

```
Status                 0      1
rate_of_interest
False             112031    200
True                   0  36439
```

---

## 3. How it reached the model

The original feature engineering:

```python
# modeltraining.py — the original code
for col in ['Upfront_charges', 'Interest_rate_spread', 'rate_of_interest',
            'dtir1', 'LTV', 'property_value']:
    df[col + '_missing'] = df[col].isnull().astype(int)
```

This constructs `Interest_rate_spread_missing`, which *is* the label, and hands
it to the classifier as a feature.

Feature importances of the shipped model confirm it:

```
0.49691  bin__rate_of_interest_missing
0.47780  bin__Interest_rate_spread_missing
0.01121  cat__age_nan
0.00857  cat__credit_type_EQUI
0.00552  bin__Upfront_charges_missing
0.00000  num__Credit_Score
0.00000  num__LTV
0.00000  num__dtir1
0.00000  num__income
...
```

**97.5% of total importance sat on two leakage indicators. 82 of 87 features had
exactly zero importance** — including every genuine underwriting variable.

The model was not a credit model. It was a lookup table on one NaN pattern.

### Why the fields are missing

The three worst offenders — `rate_of_interest`, `Interest_rate_spread`,
`Upfront_charges` — are **loan pricing terms**. They exist only for loans that
were actually originated and priced. A defaulted record in this dataset has them
blank. Their presence is an *outcome* of the process being predicted, not an
input available at decision time. This is classic temporal leakage: information
from after the decision point leaking backwards into it.

---

## 4. The second-order leak

Dropping the three pricing columns is not sufficient. `LTV`, `property_value`
and `dtir1` also leak through missingness — weaker, but material.

A model trained on **nothing but the missingness flags** of those three columns:

```
ROC-AUC = 0.7155
```

And median imputation does not fix it: every imputed row lands on the *same*
exact value, which a tree can isolate as cleanly as a NaN flag.

This is why the naive fix still looked too good:

| Configuration | ROC-AUC | PR-AUC |
|---|---|---|
| Original (all leakage present) | 1.0000 | 1.0000 |
| Drop `ID` only | 1.0000 | 1.0000 |
| Drop 3 pricing columns, median-impute the rest | 0.8980 | 0.8482 |
| **Complete cases, no protected attributes** | **0.8269** | **0.6331** |

The resolution adopted here is **complete-case training**: rows missing
`property_value`, `LTV` or `dtir1` are excluded rather than imputed. Cost: 24,123
rows (16.2%). Benefit: the leakage channel is closed entirely.

---

## 5. Honest performance

Final leakage-free model, held-out test set of 24,910 rows at a 16.32% base rate:

| Metric | Value |
|---|---|
| ROC-AUC | **0.8244** (95% CI 0.8166–0.8320) |
| Gini | 0.6489 |
| PR-AUC | **0.6104** vs 0.1632 baseline — a **3.7× lift** |
| KS | 0.4892 |
| Brier | 0.0933 |
| Max calibration error | 0.0136 |

For a retail PD model this is a good, defensible result. Production retail
scorecards commonly land in this range.

### Where the real signal is

With the leakage gone, the drivers are economically interpretable:

| Feature | Default rate | vs baseline |
|---|---|---|
| `lump_sum_payment = lpsm` | **66.2%** | 15.5% |
| `Neg_ammortization = neg_amm` | **33.1%** | 14.5% |
| `business_or_commercial = b/c` | **28.3%** | 14.7% |
| `submission_of_application = to_inst` | 19.9% | 10.0% |
| `loan_purpose = p2` | 25.2% | ~15% |

Balloon payments, negative amortisation and non-owner-occupied lending — the
non-traditional mortgage features that drove losses in 2007–08. That is a
credible risk story; "missingness of a pricing field" is not.

---

## 6. A separate finding: `Credit_Score` is noise

While validating the honest model, `Credit_Score` showed no signal at all:

```
Univariate ROC-AUC = 0.5030

Default rate by decile:
0.166  0.163  0.163  0.158  0.161  0.156  0.164  0.164  0.168  0.170
```

Completely flat. In any real bureau dataset credit score is the strongest single
predictor. Combined with its range (500–900, which is not FICO's 300–850), this
column appears to be **randomly generated**.

The original README reported this as a modelling finding — *"Weak predictors:
Credit_Score, LTV, loan_amount showed little separation"* — rather than as a
data quality problem. It is documented as a limitation in `MODEL_CARD.md`.

---

## 7. What now prevents recurrence

| Control | Location |
|---|---|
| Missingness indicators are forbidden in feature engineering | `test_feature_engineering_creates_no_missingness_indicators` |
| Excluded columns cannot reach the feature set | `test_excluded_columns_absent_from_loaded_features` |
| **Training aborts if test ROC-AUC > 0.95** | `LeakageGuardError` in `models/train.py` |
| No single feature may hold >50% of SHAP importance | `test_no_single_feature_dominates` |
| Predictions must not pile up at 0 and 1 | `test_predictions_are_not_degenerate` |
| The exact leak is documented as a live assertion | `test_leakage_still_present_in_raw_data` |
| Exclusions are configuration, with reasons | `config/model.yaml` |

The guardrail is the important one. **A model that scores too well now fails the
build**, which is the control that was missing when a 1.0000 AUC was accepted as
a result.

---

## 8. Lessons

1. **A perfect score is a bug report, not an achievement.** The instinct to
   distrust 1.0000 matters more than any modelling technique.
2. **Logistic regression is a leak detector.** If a linear model matches a
   gradient-boosted ensemble at a suspiciously high score, look at the data.
3. **Check feature importance before believing a metric.** Two features holding
   97.5% of importance was visible immediately.
4. **Missingness indicators are dangerous on observational data.** They are
   legitimate when missingness is genuinely random. Here, missingness *was* the
   outcome.
5. **Leakage has layers.** The obvious fix left a 0.7155-AUC channel still open.
6. **Encode the finding as a test.** Documentation is forgotten; a failing build
   is not.
