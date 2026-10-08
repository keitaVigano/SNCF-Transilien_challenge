"""Training / evaluation loop for the XGBoost model.

XGBoost's sklearn API already implements the boosting loop and early
stopping internally (`model.fit(..., eval_set=...)`); this wrapper just
runs that, mirrors its per-round metrics into the same MetricsLogger
format the SELM/DELM pipeline uses (so TensorBoard/metrics.csv tooling
stays consistent across models), and exposes the same evaluate()/predict()
surface the rest of the project expects.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from metrics_logger import MetricsLogger
from utils import mae, rmse

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model: xgb.XGBRegressor, train_cfg: dict[str, Any], checkpoint_dir: str | Path):
        self.model = model
        self.cfg = train_cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metrics = MetricsLogger(self.checkpoint_dir, use_tensorboard=train_cfg.get("use_tensorboard", True))

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        eval_metric = self.cfg["eval_metric"]
        results = self.model.evals_result()
        train_curve = results["validation_0"][eval_metric]
        val_curve = results["validation_1"][eval_metric]
        for round_idx, (t, v) in enumerate(zip(train_curve, val_curve), start=1):
            self.metrics.log_group(eval_metric, {"train": t, "val": v}, step=round_idx, epoch=round_idx)

        best_iteration = getattr(self.model, "best_iteration", None)
        n_rounds = (best_iteration + 1) if best_iteration is not None else len(val_curve)
        logger.info(
            "Boosting done: %d rounds trained, best_iteration=%s, best val %s=%.4f",
            len(val_curve), best_iteration, eval_metric, min(val_curve),
        )
        self.model.save_model(str(self.checkpoint_dir / "xgb_model.json"))
        self.metrics.close()

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
        preds = self.model.predict(X)
        targets = y.to_numpy()
        return mae(preds, targets), rmse(preds, targets)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
