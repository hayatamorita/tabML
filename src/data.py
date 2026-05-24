from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    return config.get(name, {})


def load_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if _section(config, "io").get("input_format", "csv") != "csv":
        raise ValueError("Only csv input_format is supported.")

    data_config = _section(config, "data")
    train_path = data_config.get("train_path")
    if train_path is None:
        raise ValueError("data.train_path is required.")

    train_df = pd.read_csv(Path(train_path))
    test_path = data_config.get("test_path")
    if test_path is None or not Path(test_path).exists():
        return train_df, None
    return train_df, pd.read_csv(Path(test_path))


def prepare_xy(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    data_config = _section(config, "data")
    target_col = data_config.get("target_col")
    if not target_col:
        raise ValueError("data.target_col is required.")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' is not in train data.")

    y = df[target_col]
    X = df.drop(columns=[target_col])
    drop_cols = _section(config, "features").get("drop_cols", [])
    return X.drop(columns=[c for c in drop_cols if c in X.columns]), y


def make_regression_bins(y: pd.Series | np.ndarray, n_bins: int = 10) -> np.ndarray:
    y_series = pd.Series(y).reset_index(drop=True)
    unique_count = y_series.nunique(dropna=False)
    if unique_count < 2:
        return np.zeros(len(y_series), dtype=int)
    bins = min(n_bins, unique_count, len(y_series))
    try:
        return pd.qcut(y_series, q=bins, labels=False, duplicates="drop").fillna(0).to_numpy()
    except ValueError:
        return np.zeros(len(y_series), dtype=int)


def make_cv_splitter(config: dict[str, Any], y: pd.Series | np.ndarray):
    split_config = _section(config, "split")
    task_type = _section(config, "task").get("type", "regression")
    n_splits = int(split_config.get("n_splits", 5))
    shuffle = bool(split_config.get("shuffle", True))
    random_state = split_config.get("random_state", split_config.get("seed", 42))
    stratified = bool(split_config.get("stratified", True))

    if stratified:
        split_y = make_regression_bins(y) if task_type == "regression" else y
        if not _can_stratify(split_y, n_splits):
            return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state), y
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state), split_y
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state), y


def make_holdout_split(X: pd.DataFrame, y: pd.Series, config: dict[str, Any]):
    split_config = _section(config, "split")
    task_type = _section(config, "task").get("type", "regression")
    stratify = None
    if split_config.get("stratified", True):
        stratify = make_regression_bins(y) if task_type == "regression" else y
        if not _can_holdout_stratify(stratify, split_config.get("test_size", 0.2), len(y)):
            stratify = None

    return train_test_split(
        X,
        y,
        test_size=split_config.get("test_size", 0.2),
        random_state=split_config.get("random_state", 42),
        shuffle=split_config.get("shuffle", True),
        stratify=stratify,
    )


def _can_stratify(labels, min_count: int) -> bool:
    counts = pd.Series(labels).value_counts(dropna=False)
    return len(counts) > 1 and bool((counts >= min_count).all())


def _can_holdout_stratify(labels, test_size: float | int, n_samples: int) -> bool:
    counts = pd.Series(labels).value_counts(dropna=False)
    if len(counts) <= 1 or not bool((counts >= 2).all()):
        return False
    if isinstance(test_size, float):
        test_count = int(np.ceil(n_samples * test_size))
    else:
        test_count = int(test_size)
    train_count = n_samples - test_count
    return test_count >= len(counts) and train_count >= len(counts)
