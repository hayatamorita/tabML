from __future__ import annotations

import json
import pickle
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_dir(output_root: str | Path) -> Path:
    output_root = ensure_dir(output_root)
    date_part = datetime.now().strftime("%Y%m%d")
    alphabet = string.ascii_lowercase + string.digits
    for _ in range(100):
        suffix = "".join(random.choices(alphabet, k=4))
        run_dir = output_root / f"{date_part}-{suffix}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
            ensure_dir(run_dir / "summary")
            return run_dir
    raise RuntimeError("Could not create a unique run directory.")


def make_model_dir(run_dir: str | Path, model_alias: str) -> Path:
    model_dir = ensure_dir(Path(run_dir) / model_alias)
    ensure_dir(model_dir / "models")
    ensure_dir(model_dir / "predictions")
    ensure_dir(model_dir / "reports")
    return model_dir


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def make_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_jsonable(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(obj), f, ensure_ascii=False, indent=2)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def get_model_alias(model_config: dict[str, Any]) -> str:
    return str(model_config.get("alias") or model_config["name"])


def save_config(config: dict[str, Any], path: str | Path) -> None:
    save_json(config, path)


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items
