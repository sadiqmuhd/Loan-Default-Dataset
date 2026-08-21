"""Dataset loading, hashing and the leakage-safe row/column filters.

The hash matters for governance: a model artifact records the SHA-256 of the
exact file it was trained on, so a prediction can always be traced back to its
training data.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from loan_default.config import excluded_columns, load_model_config

logger = logging.getLogger(__name__)


@dataclass
class LoadedDataset:
    """A prepared dataset plus the provenance needed to reproduce it."""

    X: pd.DataFrame
    y: pd.Series
    data_hash: str
    source_path: str
    n_raw_rows: int
    n_rows: int
    dropped_columns: dict[str, list[str]]
    complete_case_columns: list[str] = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        return self.n_raw_rows - self.n_rows

    def provenance(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "data_sha256": self.data_hash,
            "n_raw_rows": self.n_raw_rows,
            "n_rows_used": self.n_rows,
            "rows_dropped": self.rows_dropped,
            "dropped_columns": self.dropped_columns,
            "complete_case_columns": self.complete_case_columns,
            "default_rate": float(self.y.mean()),
        }


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a SHA-256 of the file, so a 28MB CSV never lands in memory twice."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(
    path: Path | str | None = None,
    *,
    apply_complete_case: bool = True,
) -> LoadedDataset:
    """Load the loan dataset with all leakage controls applied.

    Three filters are applied, each for a documented reason (config/model.yaml):

    1. Column exclusions - target leakage, protected characteristics, identifiers.
    2. Complete-case filter - removes the second-order leakage channel where
       missingness of ``property_value`` / ``LTV`` / ``dtir1`` predicts default
       (measured ROC-AUC 0.7155 using missingness alone).
    3. Rows with a null target.
    """
    cfg = load_model_config()
    source = Path(path) if path is not None else Path(cfg["data_path"])
    if not source.is_absolute():
        from loan_default.config import PROJECT_ROOT

        source = PROJECT_ROOT / source
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found at {source}. See README.md for how to obtain it."
        )

    data_hash = file_sha256(source)
    df = pd.read_csv(source)
    n_raw = len(df)
    target = cfg["target"]

    df = df[df[target].notna()]

    cc_cols: list[str] = []
    if apply_complete_case:
        cc_cols = list(cfg["complete_case_columns"])
        before = len(df)
        df = df.dropna(subset=cc_cols)
        logger.info(
            "complete-case filter on %s dropped %d rows (%.1f%%)",
            cc_cols,
            before - len(df),
            100 * (before - len(df)) / max(before, 1),
        )

    drop = [c for c in excluded_columns(cfg) if c in df.columns]
    y = df[target].astype(int)
    X = df.drop(columns=[target, *drop])

    logger.info("loaded %d rows, %d columns, default rate %.4f", len(X), X.shape[1], y.mean())

    return LoadedDataset(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        data_hash=data_hash,
        source_path=str(source),
        n_raw_rows=n_raw,
        n_rows=len(X),
        dropped_columns=dict(cfg["exclusions"]),
        complete_case_columns=cc_cols,
    )
