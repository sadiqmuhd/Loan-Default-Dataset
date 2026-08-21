"""Derive the data contract from the dataset itself.

This exists because the original hand-written Pydantic schema rejected 99.62%
of the very dataset the model was trained on (148,111 of 148,670 rows), while
allowing eleven enum values that appear nowhere in the data. Enumerations must
be generated from the data, never typed by hand.

Numeric bounds are NOT taken from the observed min/max, because the observed
range contains impossible values (LTV reaches 7831%, income goes negative).
Bounds are domain-set and deliberately reject those rows; the script reports
how many observations each bound excludes so the choice stays honest.

Usage:
    python scripts/generate_data_contract.py
Writes:
    config/data_contract.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import CONFIG_DIR, PROJECT_ROOT  # noqa: E402

# Domain-set numeric bounds. Values outside these are data errors, not applicants.
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "loan_amount": (1_000.0, 100_000_000.0),
    "term": (12.0, 480.0),
    "property_value": (1_000.0, 100_000_000.0),
    "income": (1.0, 100_000_000.0),  # rejects the 1,260 rows with income <= 0
    "Credit_Score": (300.0, 900.0),
    "LTV": (0.0, 250.0),  # rejects the LTV=7831% data errors
    "dtir1": (0.0, 100.0),
}

TARGET = "Status"
# Columns dropped from the serving contract entirely (see config/model.yaml).
NOT_IN_CONTRACT = {"ID", "year", "rate_of_interest", "Interest_rate_spread", "Upfront_charges"}


def main() -> int:
    data_path = PROJECT_ROOT / "data" / "Loan_Default.csv"
    if not data_path.exists():
        print(f"ERROR: dataset not found at {data_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {data_path.name}\n")

    contract: dict[str, dict] = {"categorical": {}, "numeric": {}}

    cat_cols = [
        c for c in df.columns if df[c].dtype == object or isinstance(df[c].dtype, pd.StringDtype)
    ]
    for col in sorted(cat_cols):
        if col in NOT_IN_CONTRACT or col == TARGET:
            continue
        values = sorted(str(v) for v in df[col].dropna().unique())
        nullable = bool(df[col].isna().any())
        contract["categorical"][col] = {
            "allowed": values,
            "nullable": nullable,
            "null_count": int(df[col].isna().sum()),
        }
        print(f"  {col:28} {len(values):>2} levels  nullable={nullable!s:<5} {values}")

    print()
    num_cols = [c for c in df.columns if c not in cat_cols]
    for col in sorted(num_cols):
        if col in NOT_IN_CONTRACT or col == TARGET:
            continue
        s = df[col]
        lo, hi = NUMERIC_BOUNDS.get(col, (float(s.min()), float(s.max())))
        rejected = int(((s < lo) | (s > hi)).sum())
        contract["numeric"][col] = {
            "min": float(lo),
            "max": float(hi),
            "nullable": bool(s.isna().any()),
            "observed_min": float(s.min()),
            "observed_max": float(s.max()),
            "rejected_by_bounds": rejected,
        }
        flag = f"  <-- rejects {rejected:,} out-of-domain rows" if rejected else ""
        print(
            f"  {col:28} [{lo:>12,.1f}, {hi:>14,.1f}]  observed [{s.min():>12,.1f}, {s.max():>14,.1f}]{flag}"
        )

    out = CONFIG_DIR / "data_contract.yaml"
    header = (
        "# GENERATED FILE - do not edit by hand.\n"
        "# Regenerate with: python scripts/generate_data_contract.py\n"
        "#\n"
        "# Categorical levels are derived from data/Loan_Default.csv.\n"
        "# Numeric bounds are domain-set (see scripts/generate_data_contract.py)\n"
        "# and deliberately reject impossible observations.\n\n"
    )
    with out.open("w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(contract, fh, sort_keys=True, default_flow_style=False)

    print(f"\nWrote {out.relative_to(PROJECT_ROOT)}")
    print(f"  categorical fields: {len(contract['categorical'])}")
    print(f"  numeric fields:     {len(contract['numeric'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
