from __future__ import annotations

from pathlib import Path
from typing import Any


def _missing_optional(package: str, install_hint: str) -> ImportError:
    return ImportError(f"{package} is not installed. Install it with: {install_hint}")


def create_model(config: dict[str, Any], model_config: dict[str, Any] | None = None):
    model_config = model_config or config
    model_name = model_config["name"]
    params = dict(model_config.get("params", {}))
    task_type = config.get("task", {}).get("type", "regression")
    if task_type == "regression":
        return create_regression_model(model_name, params)
    return create_classification_model(model_name, params)


def create_regression_model(model_name: str, params: dict[str, Any]):
    if model_name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(**params)
    if model_name == "lasso":
        from sklearn.linear_model import Lasso

        return Lasso(**params)
    if model_name == "knn":
        from sklearn.neighbors import KNeighborsRegressor

        return KNeighborsRegressor(**params)
    if model_name == "svr":
        from sklearn.svm import SVR

        return SVR(**params)
    if model_name == "mlp":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(**params)
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**params)
    if model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(**params)
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise _missing_optional("xgboost", "pip install xgboost") from exc
        return XGBRegressor(**params)
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise _missing_optional("lightgbm", "pip install lightgbm") from exc
        return LGBMRegressor(**params)
    if model_name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise _missing_optional("catboost", "pip install catboost") from exc
        return CatBoostRegressor(verbose=False, **params)
    if model_name == "tabpfn":
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as exc:
            raise _missing_optional("tabpfn", "pip install tabpfn") from exc
        return TabPFNRegressor(**params)
    raise ValueError(f"Unsupported regression model: {model_name}")


def create_classification_model(model_name: str, params: dict[str, Any]):
    if model_name == "ridge":
        from sklearn.linear_model import RidgeClassifier

        return RidgeClassifier(**params)
    if model_name == "lasso":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(penalty="l1", solver="liblinear", **params)
    if model_name == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(**params)
    if model_name == "svr":
        from sklearn.svm import SVC

        return SVC(probability=True, **params)
    if model_name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(**params)
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**params)
    if model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(**params)
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise _missing_optional("xgboost", "pip install xgboost") from exc
        return XGBClassifier(**params)
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise _missing_optional("lightgbm", "pip install lightgbm") from exc
        return LGBMClassifier(**params)
    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise _missing_optional("catboost", "pip install catboost") from exc
        return CatBoostClassifier(verbose=False, **params)
    if model_name == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise _missing_optional("tabpfn", "pip install tabpfn") from exc
        return TabPFNClassifier(**params)
    raise ValueError(f"Unsupported classification model: {model_name}")


def train_autogluon(train_df, config: dict[str, Any], output_dir: str | Path):
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise _missing_optional("autogluon.tabular", "pip install autogluon.tabular") from exc

    label = config["data"]["target_col"]
    problem_type = None if config.get("task", {}).get("type") == "classification" else config.get("task", {}).get("type")
    predictor = TabularPredictor(label=label, problem_type=problem_type, path=str(output_dir))
    return predictor.fit(train_df)
