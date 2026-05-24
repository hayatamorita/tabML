from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCALED_MODELS = {"ridge", "lasso", "knn", "svr", "mlp"}
TREE_MODELS = {"random_forest", "extra_trees", "xgboost", "lightgbm", "catboost", "tabpfn"}


def resolve_preprocess_type(
    model_name: str,
    global_config: dict[str, Any],
    model_config: dict[str, Any] | None = None,
) -> str:
    if model_config and model_config.get("preprocess_type"):
        return str(model_config["preprocess_type"])
    feature_config = global_config.get("features", {})
    if feature_config.get("preprocess_type"):
        return str(feature_config["preprocess_type"])
    if model_name in SCALED_MODELS:
        return "sklearn_scaled"
    if model_name in TREE_MODELS:
        return "sklearn_tree"
    if model_name == "autogluon":
        return "none"
    raise ValueError(f"Unknown model for preprocess default: {model_name}")


def infer_columns(X: pd.DataFrame, config: dict[str, Any]) -> tuple[list[str], list[str]]:
    feature_config = config.get("features", {})
    numerical_cols = feature_config.get("numerical_cols", "auto")
    categorical_cols = feature_config.get("categorical_cols", "auto")

    if numerical_cols == "auto":
        numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    if categorical_cols == "auto":
        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return list(numerical_cols), list(categorical_cols)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame, config: dict[str, Any], preprocess_type: str | None = None):
    preprocess_type = preprocess_type or config.get("features", {}).get("preprocess_type", "sklearn_tree")
    if preprocess_type == "none":
        return None

    numerical_cols, categorical_cols = infer_columns(X, config)
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if preprocess_type == "sklearn_scaled":
        num_steps.append(("scaler", StandardScaler()))
    elif preprocess_type != "sklearn_tree":
        raise ValueError(f"Unknown preprocess_type: {preprocess_type}")

    transformers = []
    if numerical_cols:
        transformers.append(("num", Pipeline(num_steps), numerical_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_test_features(test_df: pd.DataFrame | None, config: dict[str, Any]) -> pd.DataFrame | None:
    if test_df is None:
        return None
    data_config = config.get("data", {})
    feature_config = config.get("features", {})
    drop_cols = list(feature_config.get("drop_cols", []))
    target_col = data_config.get("target_col")
    cols_to_drop = [c for c in drop_cols + [target_col] if c and c in test_df.columns]
    return test_df.drop(columns=cols_to_drop)
