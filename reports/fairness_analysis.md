# Fair Lending Analysis

Model `v20260821T011832Z`, scored across 124,547 loans.
Regenerate the underlying numbers with `make report`
(writes `reports/fairness_report.json`).

---

## What was done, and why

Gender and age are excluded from the model, and the API rejects them outright:
send either field and the request fails validation. Excluding a field from the
feature list still leaves it sitting in the request payload where a future change
could pick it up; refusing to accept it removes that possibility.

Exclusion on its own is not sufficient, though, and treating it as sufficient is
the usual mistake. A model with no gender field can still produce uneven outcomes
by leaning on something correlated with gender. So outcomes are measured across
groups even though the model never sees them — the protected attributes are
joined back after scoring, purely for this analysis.

The measure used is the **adverse impact ratio**: each group's approval rate
divided by that of the most-approved group. A ratio below 0.80 is the
conventional threshold for treating a disparity as worth investigating, borrowed
by analogy from employment selection guidance. It is a review trigger, not a
legal test.

Decisions here are taken at the break-even PD for a 45% LGD (approve where
PD ≤ 0.2373), giving an overall approval rate of 82.24%.

---

## What it cost to exclude them

| Model | ROC-AUC |
|---|---|
| With gender and age | 0.8258 |
| Without | 0.8253 |
| **Difference** | **0.00046 (4.6 basis points)** |

Both models were trained identically, differing only in whether the two columns
were present.

This number is worth having for a specific reason. The argument against removing
protected attributes is always that it costs predictive power. Here it costs
four and a half basis points of AUC, which is inside the noise of the 95%
confidence interval on the metric itself (0.8166–0.8320). The trade-off is real
but negligible, and now it is measured rather than asserted.

---

## Outcomes by group

### Gender

| Group | Loans | Approval rate | Mean PD | Observed default rate | AIR |
|---|---|---|---|---|---|
| Joint | 34,867 | 89.15% | 12.04% | 11.82% | 1.000 |
| Male | 35,319 | 81.56% | 17.10% | 17.75% | 0.915 |
| Female | 23,588 | 80.39% | 17.45% | 16.60% | 0.902 |
| Sex Not Available | 30,773 | 76.60% | 19.97% | 19.57% | 0.859 |

No group falls below 0.80. Male and female approval rates sit within 1.2
percentage points of each other. The two extremes are joint applications, which
default less often and are approved more, and applications where gender was not
recorded, which default more often and are approved less.

That last group is worth a note: "Sex Not Available" is not a demographic
category, it is a data-collection artifact, and the model is picking up whatever
correlates with the field being blank. It is the lowest-approved group in the
book. Nothing here is acting on gender, but it is the kind of pattern a reviewer
should see rather than have buried.

### Age

| Group | Loans | Approval rate | Mean PD | Observed default rate | AIR |
|---|---|---|---|---|---|
| 35–44 | 27,566 | 84.94% | 14.62% | 14.10% | 1.000 |
| 25–34 | 15,985 | 84.48% | 14.79% | 13.86% | 0.995 |
| 45–54 | 29,378 | 83.00% | 16.02% | 16.29% | 0.977 |
| 55–64 | 27,709 | 80.72% | 17.64% | 18.01% | 0.950 |
| 65–74 | 17,061 | 79.40% | 18.32% | 18.03% | 0.935 |
| >74 | 5,720 | 76.45% | 20.22% | 20.12% | 0.900 |
| <25 | 1,128 | 73.85% | 20.11% | 19.95% | 0.869 |

Approval declines steadily with age above 45, and the youngest group is approved
least. Both extremes stay above the threshold. The gradient tracks the observed
default rate closely, which suggests the model is responding to genuine risk
characteristics that correlate with age rather than to age itself — but the
correlation is exactly why the field is excluded.

### Region

Region **is** a model input, so it is reported here for a different reason.

| Group | Loans | Approval rate | Mean PD | Observed default rate | AIR |
|---|---|---|---|---|---|
| North | 63,377 | 85.39% | 14.63% | 14.56% | 1.000 |
| south | 53,010 | 79.72% | 17.99% | 17.80% | 0.934 |
| central | 7,155 | 74.17% | 20.50% | 20.24% | 0.869 |
| North-East | 1,005 | 73.53% | 22.13% | 21.89% | 0.861 |

Coarse geography is a recognised proxy risk — this is the shape of variable that
redlining concerns attach to. It is retained here because the levels are very
broad (four regions covering the entire book, not postcodes or census tracts)
and because it carries genuine, well-calibrated risk signal: predicted and
observed default rates agree within 0.3 percentage points in every region.

That is a judgement call, and it is the one item in this analysis I would expect
a model risk function to push back on. The defensible position is that it is
declared, measured and monitored rather than quietly included.

---

## Calibration within groups

Across all three attributes, mean predicted PD sits within roughly one
percentage point of the observed default rate for every group. The model is not
systematically over- or under-predicting risk for any of them, which matters
because a group whose PD is inflated would be denied credit on the strength of a
model error rather than its actual risk.

---

## Limitations

**Only originated loans appear in the data.** Everyone the original lender
declined is absent, so these approval rates describe how the model would treat a
population that has already been filtered by someone else's credit policy — and
that policy's own biases are invisible here. Correcting for it needs
reject-inference data this dataset does not contain.

**The four-fifths rule is a heuristic borrowed by analogy.** It comes from
employment selection guidance, not lending, and passing it is not evidence of
compliance with anything.

**Gender in this dataset includes "Joint" and "Sex Not Available".** Neither is a
protected class in the usual sense, and both behave differently from the
individual categories, which makes the comparison less clean than it looks.

**This is not a compliance assessment.** It is a design decision, measured and
documented. A real fair-lending review would involve legal counsel, a wider set
of protected characteristics than this dataset carries, and analysis of the
policy that generated the data in the first place.
