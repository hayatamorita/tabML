from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None


def evaluate(
    y_true,
    y_pred,
    config: dict[str, Any],
    y_proba=None,
) -> dict[str, float]:
    task_type = config.get("task", {}).get("type", "regression")
    metrics = config.get("evaluation", {}).get("metrics", [])
    if task_type == "regression":
        return calculate_regression_metrics(y_true, y_pred, metrics)
    return calculate_classification_metrics(y_true, y_pred, metrics, y_proba=y_proba)


def calculate_regression_metrics(y_true, y_pred, metrics: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in metrics:
        if metric == "rmse":
            if root_mean_squared_error is not None:
                result[metric] = float(root_mean_squared_error(y_true, y_pred))
            else:
                result[metric] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif metric == "mae":
            result[metric] = float(mean_absolute_error(y_true, y_pred))
        elif metric == "r2":
            result[metric] = float(r2_score(y_true, y_pred))
        else:
            raise ValueError(f"Unsupported regression metric: {metric}")
    return result


def calculate_classification_metrics(y_true, y_pred, metrics: list[str], y_proba=None) -> dict[str, float]:
    result: dict[str, float] = {}
    labels = np.unique(y_true)
    average = "binary" if len(labels) == 2 else "macro"
    for metric in metrics:
        if metric == "accuracy":
            result[metric] = float(accuracy_score(y_true, y_pred))
        elif metric == "f1":
            result[metric] = float(f1_score(y_true, y_pred, average=average))
        elif metric == "auc":
            if y_proba is None:
                raise ValueError("auc requires y_proba.")
            auc_input = y_proba[:, 1] if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] == 2 else y_proba
            result[metric] = float(roc_auc_score(y_true, auc_input, multi_class="ovr" if len(labels) > 2 else "raise"))
        elif metric == "logloss":
            if y_proba is None:
                raise ValueError("logloss requires y_proba.")
            result[metric] = float(log_loss(y_true, y_proba))
        else:
            raise ValueError(f"Unsupported classification metric: {metric}")
    return result


def summarize_cv_metrics(fold_metrics: list[dict[str, float]] | pd.DataFrame, main_metric: str) -> dict[str, float]:
    df = pd.DataFrame(fold_metrics)
    summary = {col: float(df[col].mean()) for col in df.columns if pd.api.types.is_numeric_dtype(df[col])}
    if main_metric in summary:
        summary["main_metric"] = summary[main_metric]
    return summary
