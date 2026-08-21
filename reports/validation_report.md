# Model Validation Report

Generated 2026-08-21 01:23 UTC from model
`v20260821T011832Z` on the held-out test set (24,910 loans,
16.32% default rate).

Regenerate with `python scripts/validation_report.py`.

## 1. Headline performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.8244 |
| Gini | 0.6489 |
| PR-AUC | 0.6104 (baseline 0.1632) |
| KS | 0.4892 |
| Brier score | 0.0933 |
| Mean predicted PD | 0.1654 |
| Observed default rate | 0.1632 |
| Mean calibration error | 0.0060 |
| Max calibration error | 0.0136 |

## 2. Risk grade rank ordering

Observed default rate must increase from A through G, or the grades cannot
support a lending decision.

**Monotonic: yes**

| grade | n | share | mean_pd | observed_default_rate |
|---|---|---|---|---|
| A | 88 | 0.0035 | 0.0144 | 0.0000 |
| B | 5,898 | 0.2368 | 0.0348 | 0.0342 |
| C | 5,961 | 0.2393 | 0.0679 | 0.0589 |
| D | 8,318 | 0.3339 | 0.1321 | 0.1354 |
| E | 1,784 | 0.0716 | 0.2650 | 0.2618 |
| F | 1,077 | 0.0432 | 0.4338 | 0.4234 |
| G | 1,784 | 0.0716 | 0.8243 | 0.8206 |

## 3. Calibration

Predicted versus observed default rate by predicted-PD decile. A well
calibrated model tracks the diagonal; see `reports/reliability_curve.png`.

| n | predicted | observed | gap |
|---|---|---|---|
| 3,524 | 0.0317 | 0.0295 | -0.0022 |
| 2,461 | 0.0385 | 0.0398 | 0.0014 |
| 2,314 | 0.0582 | 0.0501 | -0.0081 |
| 2,115 | 0.0653 | 0.0629 | -0.0024 |
| 3,804 | 0.0987 | 0.0852 | -0.0136 |
| 1,948 | 0.1109 | 0.1196 | 0.0087 |
| 2,341 | 0.1357 | 0.1491 | 0.0134 |
| 1,758 | 0.1830 | 0.1832 | 0.0002 |
| 2,601 | 0.3107 | 0.3045 | -0.0062 |
| 2,044 | 0.7839 | 0.7803 | -0.0036 |

### Calibration method selection

Both candidates were fitted on the held-out calibration slice and measured on
the test set. Selection is on calibration error rather than Brier score, which
mixes discrimination and calibration together and barely moves when only the
latter improves.

| method | brier_score | calibration_error | max_calibration_error | roc_auc |
|---|---|---|---|---|
| uncalibrated | 0.09324495521108878 | 0.008838255425211487 | 0.03090558337763294 | 0.8249228652677001 |
| isotonic | 0.09333241282043375 | 0.005975624526143574 | 0.013552461497164422 | 0.8244480075586446 |
| sigmoid | 0.09382267713989936 | 0.02359722024058183 | 0.03612403404134873 | 0.8249228652677001 |

## 4. Decision outcomes

At the break-even threshold for an unsecured-equivalent LGD of 0.45
(PD <= 0.2373):

| Quantity | Value |
|---|---|
| Approval rate | 82.20% |
| Default rate among approved | 8.45% |
| Recall on defaulters | 57.43% |
| Precision | 52.66% |
| True negatives | 18,745 |
| False positives | 2,099 |
| False negatives | 1,731 |
| True positives | 2,335 |

## 5. Performance by segment

A single headline AUC can hide a segment where the model does not discriminate. Segments below 200 loans are not evaluated, since the estimate would be too noisy to act on.

### Region

| segment | n | default_rate | mean_pd | calibration_gap | roc_auc |
|---|---|---|---|---|---|
| North | 12,636 | 0.1447 | 0.1464 | -0.0018 | 0.8244 |
| south | 10,604 | 0.1797 | 0.1822 | -0.0025 | 0.8186 |
| central | 1,461 | 0.1951 | 0.1986 | -0.0035 | 0.8357 |
| North-East | 209 | 0.2249 | 0.2289 | -0.0040 | 0.8622 |

### loan_purpose

| segment | n | default_rate | mean_pd | calibration_gap | roc_auc |
|---|---|---|---|---|---|
| p3 | 10,273 | 0.1726 | 0.1781 | -0.0055 | 0.8167 |
| p4 | 8,072 | 0.1555 | 0.1558 | -0.0003 | 0.8469 |
| p1 | 5,972 | 0.1495 | 0.1499 | -0.0004 | 0.8102 |
| p2 | 574 | 0.2509 | 0.2267 | 0.0242 | 0.7342 |

### occupancy_type

| segment | n | default_rate | mean_pd | calibration_gap | roc_auc |
|---|---|---|---|---|---|
| pr | 23,010 | 0.1586 | 0.1621 | -0.0035 | 0.8259 |
| ir | 1,362 | 0.2357 | 0.2203 | 0.0154 | 0.7984 |
| sr | 538 | 0.1766 | 0.1683 | 0.0083 | 0.7885 |

### credit_type

| segment | n | default_rate | mean_pd | calibration_gap | roc_auc |
|---|---|---|---|---|---|
| CIB | 9,169 | 0.1604 | 0.1567 | 0.0037 | 0.8177 |
| CRIF | 8,294 | 0.1647 | 0.1706 | -0.0059 | 0.8270 |
| EXP | 7,447 | 0.1650 | 0.1703 | -0.0053 | 0.8303 |

### Review triggers

No segment fell below the ROC-AUC floor of 0.65 or exceeded the calibration tolerance of 0.03.
