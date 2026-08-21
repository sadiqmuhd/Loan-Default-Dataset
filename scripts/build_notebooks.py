"""Build and execute the analysis notebooks.

The notebooks are generated from this script rather than edited by hand, so
their outputs cannot drift away from the code they describe: rerunning this
regenerates them against the current model and dataset.

    python scripts/build_notebooks.py

Requires the full dataset in data/. Skips gracefully if it is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

SETUP = """\
import sys, warnings
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 40)

DATA = "../data/Loan_Default.csv"
"""


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


# ---------------------------------------------------------------- 01 EDA


def build_eda() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 1. Exploratory Data Analysis\n\n"
            "First look at the loan book: what is in it, what is missing, and "
            "which variables actually separate defaulters from non-defaulters.\n\n"
            "The conclusions here feed directly into `config/model.yaml`."
        ),
        code(SETUP + "\ndf = pd.read_csv(DATA)\nprint(df.shape)\ndf.head(3)"),
        md(
            "## Target\n\nAbout one loan in four defaults, which is imbalanced but "
            "not severely so. It is mild enough that resampling would do more harm "
            "than good — it would distort the predicted probabilities, which this "
            "project needs to keep meaningful."
        ),
        code(
            'counts = df["Status"].value_counts()\n'
            "print(counts)\n"
            "print(f\"\\ndefault rate: {df['Status'].mean():.4f}\")"
        ),
        md(
            "## Missing data\n\nSeveral columns are missing a substantial share of "
            "their values. The pattern of what is missing turns out to matter far "
            "more than the amount — see notebook 02."
        ),
        code(
            "missing = (df.isna().mean().sort_values(ascending=False) * 100).round(2)\n"
            'missing[missing > 0].to_frame("percent_missing")'
        ),
        md(
            "## Numeric variables\n\nNote the ranges rather than the averages. LTV "
            "reaches 7,831% and income goes to zero, both of which are impossible "
            "and are rejected by the data contract in `config/data_contract.yaml`."
        ),
        code('df.select_dtypes("number").describe().T.round(2)'),
        md(
            "## Is income monthly or annual?\n\nThis matters, because loan-to-income "
            "is a headline affordability metric and getting the units wrong makes "
            "it meaningless."
        ),
        code(
            'd = df.dropna(subset=["income", "loan_amount", "dtir1"])\n'
            "print(f\"median income      {d['income'].median():,.0f}\")\n"
            "print(f\"median loan amount {d['loan_amount'].median():,.0f}\")\n"
            "print()\n"
            "print(f\"LTI if income is annual : {(d['loan_amount']/d['income']).median():.1f}x\")\n"
            "print(f\"LTI if income is monthly: {(d['loan_amount']/(d['income']*12)).median():.2f}x\")\n"
            "print()\n"
            'pm = d["loan_amount"] / d["term"]\n'
            "print(f\"implied principal-only burden: {(100*pm/d['income']).median():.1f}% of monthly income\")\n"
            "print(f\"reported dtir1               : {d['dtir1'].median():.1f}%\")"
        ),
        md(
            "A 52x loan-to-income ratio is not a real mortgage; 4.36x is. And a "
            "principal-only burden of 15.9% sits sensibly inside a reported total "
            "debt-to-income of 39% once interest, taxes and other debt are added.\n\n"
            "**Income is monthly.** The feature engineering annualises it."
        ),
        md("## Which categorical variables separate the classes?"),
        code(
            'cats = ["lump_sum_payment", "Neg_ammortization", "business_or_commercial",\n'
            '        "submission_of_application", "loan_purpose", "occupancy_type",\n'
            '        "credit_type", "Region"]\n'
            "for c in cats:\n"
            '    g = df.groupby(c, observed=True)["Status"].agg(["mean", "size"])\n'
            '    g.columns = ["default_rate", "n"]\n'
            '    print(f"--- {c} ---")\n'
            '    print(g.sort_values("default_rate", ascending=False).round(4).to_string())\n'
            "    print()"
        ),
        md(
            "Loan *structure* dominates. Balloon repayments default at 66% against "
            "a 25% base rate, negative amortisation at 39%, business purpose at 28%. "
            "These are the non-traditional mortgage features that drove losses in "
            "2007-08, so the signal is economically sensible rather than an artifact."
        ),
        md("## Credit score\n\nThe variable you would expect to matter most."),
        code(
            "from sklearn.metrics import roc_auc_score\n"
            'sub = df.dropna(subset=["Credit_Score", "Status"])\n'
            'auc = roc_auc_score(sub["Status"], sub["Credit_Score"])\n'
            'print(f"univariate ROC-AUC: {auc:.4f}")\n'
            "print(f\"range: {sub['Credit_Score'].min():.0f} to {sub['Credit_Score'].max():.0f}\")\n"
            "print()\n"
            "sub = sub.copy()\n"
            'sub["decile"] = pd.qcut(sub["Credit_Score"], 10, labels=False, duplicates="drop")\n'
            'print(sub.groupby("decile")["Status"].mean().round(4).to_string())'
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            'rates = sub.groupby("decile")["Status"].mean()\n'
            'ax.bar(rates.index, rates.values, color="steelblue")\n'
            'ax.axhline(df["Status"].mean(), color="crimson", ls="--", label="overall default rate")\n'
            'ax.set_xlabel("Credit score decile (low to high)")\n'
            'ax.set_ylabel("Default rate")\n'
            'ax.set_title("Default rate by credit score decile")\n'
            "ax.legend()\n"
            "plt.tight_layout()"
        ),
        md(
            "Completely flat, and an AUC of 0.503 is a coin flip. In real bureau "
            "data credit score is the single strongest predictor of default. Its "
            "range here (500-900) is also not FICO's (300-850).\n\n"
            "**This column appears to be randomly generated.** It is documented as "
            "a limitation in `MODEL_CARD.md` and there is a test asserting it stays "
            "true, so that if the data ever changes the documentation gets revisited."
        ),
        md(
            "## Where this leads\n\nThe next notebook deals with the reason an "
            "early version of this model scored a perfect 1.0000."
        ),
    ]
    return nb


# ------------------------------------------------------ 02 leakage


def build_leakage() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 2. Target Leakage Investigation\n\n"
            "An early version of this project reported **ROC-AUC 1.0000** on a "
            "held-out test set. Logistic regression, random forest and XGBoost all "
            "scored exactly 1.0.\n\n"
            "This notebook is how that was tracked down. The original notebook is "
            "kept unmodified at `archive/original_EDA_and_Modeling_LEAKING.ipynb` "
            "as evidence."
        ),
        md(
            "## The symptom\n\nThree different model families, all perfect, on "
            "29,734 held-out rows:\n\n"
            "```\n"
            "                     Accuracy  Precision  Recall  F1 Score  ROC-AUC\n"
            "Logistic Regression       1.0        1.0     1.0       1.0      1.0\n"
            "Random Forest             1.0        1.0     1.0       1.0      1.0\n"
            "XGBoost                   1.0        1.0     1.0       1.0      1.0\n"
            "```\n\n"
            "Logistic regression reaching 1.0 is the tell. A linear model cannot "
            "perfectly separate a real credit population. When a linear model and a "
            "tree ensemble agree exactly, the cause is the data, not the model."
        ),
        code(SETUP + "\ndf = pd.read_csv(DATA)\ny = df['Status']"),
        md(
            "## Hypothesis: a feature is a copy of the target\n\nThe original "
            "feature engineering created `_missing` indicator columns. If a column "
            "is only ever populated for non-defaulted loans, that indicator *is* "
            "the label."
        ),
        code(
            "rows = []\n"
            "for c in df.columns:\n"
            '    if c == "Status" or not df[c].isna().any():\n'
            "        continue\n"
            "    miss = df[c].isna()\n"
            "    rows.append({\n"
            '        "column": c,\n'
            '        "pct_missing": 100 * miss.mean(),\n'
            '        "P(default|missing)": y[miss].mean(),\n'
            '        "P(default|present)": y[~miss].mean(),\n'
            '        "agreement_with_target": (miss.astype(int) == y).mean(),\n'
            "    })\n"
            'pd.DataFrame(rows).sort_values("agreement_with_target", ascending=False).round(4)'
        ),
        md(
            "`Interest_rate_spread` agrees with the target on **100.000%** of rows. "
            "Not 99.9% — every single one of the 148,670."
        ),
        code(
            'agreement = (df["Interest_rate_spread"].isna().astype(int) == y).mean()\n'
            "print(f\"agreement: {agreement:.6%}  ({(df['Interest_rate_spread'].isna().astype(int) == y).sum():,} / {len(y):,})\")\n"
            "print()\n"
            'print(pd.crosstab(df["rate_of_interest"].isna(), df["Status"]))'
        ),
        md(
            "## Why it happens\n\n`rate_of_interest`, `Interest_rate_spread` and "
            "`Upfront_charges` are loan **pricing** fields. They only exist for "
            "loans that were originated and priced. Their presence is a consequence "
            "of the outcome being predicted, not information available when the "
            "decision is made.\n\n"
            "This is temporal leakage: post-decision information leaking backwards."
        ),
        md(
            "## What it did to the model\n\nFeature importances from the original "
            "trained artifact:\n\n"
            "```\n"
            "0.49691  rate_of_interest_missing\n"
            "0.47780  Interest_rate_spread_missing\n"
            "0.01121  age_nan\n"
            "0.00857  credit_type_EQUI\n"
            "0.00552  Upfront_charges_missing\n"
            "0.00000  Credit_Score\n"
            "0.00000  LTV\n"
            "0.00000  dtir1\n"
            "0.00000  income\n"
            "```\n\n"
            "97.5% of importance on two leakage indicators. 82 of 87 features at "
            "exactly zero, including every genuine underwriting variable. It was a "
            "lookup table, not a credit model."
        ),
        md(
            "## The second layer\n\nDropping the three pricing columns is not "
            "enough. `LTV`, `property_value` and `dtir1` leak through missingness "
            "too — more weakly, but enough to matter."
        ),
        code(
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.metrics import roc_auc_score\n"
            "from xgboost import XGBClassifier\n\n"
            "M = pd.DataFrame({c: df[c].isna().astype(int)\n"
            '                  for c in ["property_value", "LTV", "dtir1"]})\n'
            "Xtr, Xte, ytr, yte = train_test_split(M, y, test_size=.2, stratify=y, random_state=42)\n"
            'm = XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss",\n'
            "                  random_state=42).fit(Xtr, ytr)\n"
            'print(f"AUC using ONLY those three missingness flags: {roc_auc_score(yte, m.predict_proba(Xte)[:,1]):.4f}")'
        ),
        md(
            "0.7155 from nothing but three yes/no flags. And median imputation does "
            "not fix it: every imputed row lands on the same exact value, which a "
            "tree isolates as easily as a NaN.\n\n"
            "The fix adopted is **complete-case training** — drop rows missing those "
            "columns rather than impute them. It costs 16.2% of the data and closes "
            "the channel completely."
        ),
        md("## Performance at each stage of the fix"),
        code(
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.preprocessing import OneHotEncoder\n"
            "from sklearn.metrics import average_precision_score\n\n"
            'LEAK = ["rate_of_interest", "Interest_rate_spread", "Upfront_charges"]\n\n'
            "def run(label, drop, complete_case=False, drop_protected=False):\n"
            '    d = df.dropna(subset=["property_value","LTV","dtir1"]) if complete_case else df\n'
            '    yy = d["Status"]\n'
            '    cols = ["Status"] + drop + (["Gender","age"] if drop_protected else [])\n'
            "    X = d.drop(columns=[c for c in cols if c in d.columns])\n"
            "    num = X.select_dtypes(include=np.number).columns.tolist()\n"
            "    cat = [c for c in X.columns if c not in num]\n"
            "    pre = ColumnTransformer([\n"
            '        ("num", SimpleImputer(strategy="median"), num),\n'
            '        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),\n'
            '                          ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)])\n'
            "    Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=.2, stratify=yy, random_state=42)\n"
            '    p = Pipeline([("pre", pre), ("m", XGBClassifier(\n'
            "        n_estimators=250, learning_rate=.08, max_depth=6, subsample=.8,\n"
            '        colsample_bytree=.8, min_child_weight=4, eval_metric="logloss",\n'
            "        random_state=42, n_jobs=-1))]).fit(Xtr, ytr)\n"
            "    pr = p.predict_proba(Xte)[:, 1]\n"
            '    return {"configuration": label, "n": len(d), "ROC-AUC": roc_auc_score(yte, pr),\n'
            '            "PR-AUC": average_precision_score(yte, pr)}\n\n'
            "results = [\n"
            '    run("original (all leakage present)", []),\n'
            '    run("drop ID only", ["ID"]),\n'
            '    run("drop pricing columns + ID", LEAK + ["ID"]),\n'
            '    run("complete cases, no protected attrs", LEAK + ["ID"], True, True),\n'
            "]\n"
            "pd.DataFrame(results).round(4)"
        ),
        md(
            "From a fake 1.0000 to an honest **0.8244**. For a retail PD model that "
            "is a good result — production scorecards commonly land in this range."
        ),
        md(
            "## Where the real signal turned out to be\n\nWith the leakage gone, "
            "the drivers are interpretable: balloon payments, negative amortisation, "
            "business purpose, application channel. Loan structure, not a NaN "
            "pattern."
        ),
        md(
            "## What now prevents this recurring\n\n"
            "| Control | Where |\n"
            "|---|---|\n"
            "| Training aborts if test AUC > 0.95 | `LeakageGuardError` in `models/train.py` |\n"
            "| Missingness indicators forbidden | `test_feature_engineering_creates_no_missingness_indicators` |\n"
            "| Excluded columns cannot reach features | `test_excluded_columns_absent_from_loaded_features` |\n"
            "| No feature may hold >50% of SHAP importance | `test_no_single_feature_dominates` |\n"
            "| Predictions must not pile up at 0 and 1 | `test_predictions_are_not_degenerate` |\n"
            "| Data profiler flags leaky missingness | `data/quality.py` |\n\n"
            "The guardrail is the important one. A model that scores too well now "
            "**fails the build**, which is exactly the control that was missing when "
            "1.0000 was accepted as a result."
        ),
        md(
            "## What I took from this\n\n"
            "1. A perfect score is a bug report, not an achievement.\n"
            "2. Logistic regression makes a good leak detector — if it matches a "
            "gradient-boosted ensemble at a suspiciously high score, look at the data.\n"
            "3. Check feature importance before believing a metric. Two features "
            "holding 97.5% was visible immediately.\n"
            "4. Missingness indicators are dangerous on observational data. They are "
            "fine when missingness is genuinely random; here it *was* the outcome.\n"
            "5. Leakage has layers — the obvious fix left a 0.7155 channel open.\n"
            "6. Encode the finding as a test. Documentation is forgotten; a failing "
            "build is not."
        ),
    ]
    return nb


# -------------------------------------------------- 03 model development


def build_model_dev() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 3. Model Development\n\n"
            "Candidate selection, calibration and the checks that decide whether "
            "the model is usable. Everything here runs against the same code the "
            "API serves, imported from `src/loan_default`."
        ),
        code(
            SETUP + "\nfrom sklearn.model_selection import train_test_split\n"
            "from loan_default.config import get_settings\n"
            "from loan_default.data.loader import load_dataset\n"
            "from loan_default.models.registry import ModelRegistry\n\n"
            "model, meta, metrics = ModelRegistry(get_settings().artifacts_dir).load('latest')\n"
            "print('model version:', meta.model_version)\n"
            "print('trained on   :', meta.n_training_rows, 'rows')\n"
            "print('data sha256  :', meta.data_sha256[:16], '...')"
        ),
        md(
            "## What the model is allowed to see\n\nThe feature contract is "
            "configuration, not code, and every exclusion has a documented reason."
        ),
        code(
            "for reason, cols in meta.excluded_columns.items():\n"
            '    print(f"{reason:14} {cols}")\n'
            "print()\n"
            'print(f"{len(meta.feature_columns)} features used")'
        ),
        md(
            "## Candidate selection\n\nChosen on cross-validated PR-AUC. At a 16% "
            "base rate, PR-AUC is a more informative summary than ROC-AUC, and "
            "accuracy is useless — predicting 'no default' for everyone scores 0.84."
        ),
        code(
            'pd.Series(metrics["candidate_cv_pr_auc"], name="CV PR-AUC").sort_values(ascending=False).round(4).to_frame()'
        ),
        md(
            "The gap between logistic regression (0.39) and the tree ensembles is "
            "large, which is expected: the signal here is concentrated in "
            "categorical interactions that a linear model in the raw feature space "
            "cannot represent."
        ),
        md("## Held-out performance"),
        code(
            'c = metrics["calibrated"]\n'
            "summary = {\n"
            '    "ROC-AUC": c["roc_auc"], "Gini": c["gini"], "PR-AUC": c["pr_auc"],\n'
            '    "KS": c["ks_statistic"], "Brier": c["brier_score"],\n'
            '    "mean predicted PD": c["mean_predicted_pd"],\n'
            '    "observed default rate": c["base_rate"],\n'
            "}\n"
            "for k, v in summary.items():\n"
            '    print(f"{k:24} {v:.4f}")\n'
            "print()\n"
            'print("95% CI on ROC-AUC:", [round(x, 4) for x in c["roc_auc_ci"]])'
        ),
        md(
            "## Calibration\n\nA PD is not just a ranking here — it gets multiplied "
            "by LGD and EAD to produce a loss in currency. A model that ranks "
            "perfectly but says 5% when the truth is 15% understates expected loss "
            "threefold.\n\n"
            "Both candidate calibrators were fitted on a held-out slice and measured:"
        ),
        code(
            'comp = pd.DataFrame(metrics["calibration_comparison"]).T\n'
            'comp[["brier_score", "calibration_error", "max_calibration_error", "roc_auc"]].round(5)'
        ),
        md(
            "Isotonic roughly halves calibration error. Platt scaling makes it "
            "*worse than doing nothing* — its sigmoid assumption does not fit this "
            "score distribution.\n\n"
            "Note also that Brier score barely moves. That is why selection is on "
            "calibration error: Brier mixes discrimination and calibration together "
            "and hides exactly the improvement being measured."
        ),
        code(
            "from loan_default.models.calibration import reliability_table\n"
            "ds = load_dataset()\n"
            "_, Xte, _, yte = train_test_split(ds.X, ds.y, test_size=.2, stratify=ds.y, random_state=42)\n"
            "frame = Xte[meta.feature_columns]\n"
            "p = model.predict_proba(frame)[:, 1]\n"
            "tbl = reliability_table(yte, p)\n"
            "tbl.round(4)"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(6, 6))\n"
            'ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")\n'
            'ax.plot(tbl["predicted"], tbl["observed"], "o-", lw=1.5, label="model")\n'
            'ax.set_xlabel("Predicted PD"); ax.set_ylabel("Observed default rate")\n'
            'ax.set_title("Reliability curve"); ax.legend(); ax.grid(alpha=.3)\n'
            "plt.tight_layout()"
        ),
        md(
            "## Risk grades\n\nThe check that decides whether these grades can "
            "support a lending decision: does the observed default rate increase "
            "monotonically from A to G?"
        ),
        code(
            "from loan_default.risk.grades import grade_summary, is_monotonic\n"
            "g = grade_summary(p, yte)\n"
            'print("monotonic:", is_monotonic(g))\n'
            'g[["grade","n","share","mean_pd","observed_default_rate"]].round(4)'
        ),
        md(
            "Predicted and observed track closely at every grade, which is the "
            "calibration doing its job. Grade A holds only 88 loans with zero "
            "observed defaults — too thin to support a PD estimate, and noted as a "
            "limitation."
        ),
        md(
            "## Performance by segment\n\nA single headline AUC can hide a segment "
            "where the model does not work. Lending decisions are made per "
            "applicant, so this matters."
        ),
        code(
            "from loan_default.models.segments import segment_report, format_report\n"
            "print(format_report(segment_report(frame, yte, p)))"
        ),
        md(
            "Every segment clears the review thresholds. The weakest is "
            "`loan_purpose=p2` at 0.734 on 574 loans — worth watching, not worth "
            "blocking on."
        ),
        md("## What drives a prediction\n\nSHAP, aggregated back to the original features."),
        code(
            "from loan_default.models.explain import PredictionExplainer\n"
            "ex = PredictionExplainer(model)\n"
            "imp = ex.global_importance(frame, sample=2000)\n"
            "imp.head(12).round(4)"
        ),
        code(
            "top = imp.head(12).iloc[::-1]\n"
            "fig, ax = plt.subplots(figsize=(7, 5))\n"
            'ax.barh(top["feature"], top["mean_abs_shap"], color="steelblue")\n'
            'ax.set_xlabel("mean |SHAP|"); ax.set_title("Global feature importance")\n'
            "plt.tight_layout()"
        ),
        md(
            "Importance is spread across many features, which is what a real model "
            "looks like. Compare with the leaking version, where two features held "
            "97.5% and 82 held nothing.\n\n"
            "Note `Credit_Score` contributes very little — consistent with notebook "
            "01, where it had a univariate AUC of 0.503."
        ),
        md("## Limitations recorded with the model"),
        code('for i, item in enumerate(meta.limitations, 1):\n    print(f"{i}. {item}\\n")'),
    ]
    return nb


def main() -> int:
    if not (ROOT / "data" / "Loan_Default.csv").exists():
        print("Dataset not found; skipping notebook build.", file=sys.stderr)
        return 0

    NOTEBOOKS.mkdir(exist_ok=True)
    builders = {
        "01_eda.ipynb": build_eda,
        "02_leakage_investigation.ipynb": build_leakage,
        "03_model_development.ipynb": build_model_dev,
    }

    for name, builder in builders.items():
        nb = builder()
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        path = NOTEBOOKS / name
        print(f"executing {name} ...", flush=True)
        client = NotebookClient(
            nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS)}}
        )
        client.execute()
        nbf.write(nb, path)
        print(f"  wrote {path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
