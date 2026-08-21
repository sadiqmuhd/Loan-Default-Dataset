"""Pipeline construction.

Everything - feature engineering, imputation, scaling, encoding, the estimator -
lives inside a single fitted sklearn ``Pipeline``. That is what removes the
train/serve skew from the original code, where the imputer was fitted outside
the pipeline during training and then never applied at inference, so training
saw median-imputed values and serving saw raw NaN.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from loan_default.features.engineering import FeatureEngineer


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Impute, scale and encode.

    ``handle_unknown="ignore"`` on the encoder means an unseen category degrades
    to all-zeros rather than raising at inference. The API still rejects unknown
    categories up front via the Pandera contract; this is defence in depth.
    """
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_estimator(name: str, params: dict[str, Any], seed: int):
    """Instantiate a candidate estimator by name."""
    params = dict(params)
    if name == "logistic_regression":
        # class_weight="balanced" is acceptable here because logistic regression
        # is the interpretable benchmark, not the served model, and it is scored
        # on ranking metrics. The served model is calibrated post-hoc instead.
        return LogisticRegression(random_state=seed, class_weight="balanced", **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    if name == "xgboost":
        return XGBClassifier(random_state=seed, **params)
    raise ValueError(f"Unknown estimator: {name!r}")


def build_pipeline(
    estimator_name: str,
    estimator_params: dict[str, Any],
    numeric_features: list[str],
    categorical_features: list[str],
    seed: int,
    feature_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Full pipeline: raw record in, probability out.

    The pipeline accepts *raw* columns. Engineered features are produced inside
    it, so the API only ever has to supply what an underwriter would actually
    have on an application form.
    """
    return Pipeline(
        [
            ("features", FeatureEngineer(params=feature_params)),
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("estimator", build_estimator(estimator_name, estimator_params, seed)),
        ]
    )


def pipeline_feature_names(pipeline: Pipeline) -> list[str]:
    """Post-encoding feature names, for importance and SHAP reporting."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())
