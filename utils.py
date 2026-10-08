"""Shared helpers: config loading, seeding, run dirs, metrics, logging."""
from __future__ import annotations

import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_run_dir(base_dir: str | Path) -> Path:
    """Creates <base_dir>/<YYYYmmdd_HHMMSS>/ and returns it. Each run gets
    its own timestamped folder so artifacts from different runs never
    overwrite each other.
    """
    run_dir = Path(base_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: str | Path | None = None, level: int = logging.INFO) -> None:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if run_dir is not None:
        file_handler = logging.FileHandler(Path(run_dir) / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def set_seed(seed: int) -> None:
    """Fix every RNG we touch so runs are reproducible (XGBoost's own
    randomness -- row/column subsampling, tree structure ties -- is seeded
    separately via its `random_state` param, set from this same seed in
    model.py).
    """
    random.seed(seed)
    np.random.seed(seed)


def mae(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - targets)))


def rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - targets) ** 2)))
