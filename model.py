"""XGBoost gradient boosting model builder for the SNCF Transilien delay
prediction pipeline.

Unlike SELM/DELM (feature/selm branch), there's no hand-rolled forward
pass or closed-form solver here -- XGBoost's sklearn API already provides
the boosting loop, early stopping, and eval-metric tracking (see
trainer.py). This module just assembles the hyperparameters from config.

`objective` mirrors the mse/huber/mae choice explored for SELM's SGD loss:
  * reg:squarederror    -- chases the conditional mean, sensitive to the
                            rare, enormous one-off delays (see
                            analysis/error_analysis.py on feature/selm).
  * reg:absoluteerror   -- chases the conditional median; aligned with the
                            challenge's MAE scoring, robust to outliers.
  * reg:pseudohubererror -- a smooth compromise between the two, with
                            `huber_slope` as XGBoost's analogue of
                            SELM's `huber_beta`.
"""
from __future__ import annotations

import xgboost as xgb

_OBJECTIVES = {"reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"}


def build_model(cfg: dict) -> xgb.XGBRegressor:
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    objective = model_cfg.get("objective", "reg:squarederror")
    if objective not in _OBJECTIVES:
        raise ValueError(f"Unknown objective '{objective}', choose from {sorted(_OBJECTIVES)}")

    params = dict(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        learning_rate=model_cfg["learning_rate"],
        subsample=model_cfg["subsample"],
        colsample_bytree=model_cfg["colsample_bytree"],
        min_child_weight=model_cfg["min_child_weight"],
        reg_alpha=model_cfg["reg_alpha"],
        reg_lambda=model_cfg["reg_lambda"],
        gamma=model_cfg["gamma"],
        objective=objective,
        tree_method=model_cfg.get("tree_method", "hist"),
        enable_categorical=model_cfg.get("enable_categorical", True),
        n_jobs=model_cfg.get("n_jobs", -1),
        random_state=cfg["seed"],
        early_stopping_rounds=train_cfg["early_stopping_rounds"],
        eval_metric=train_cfg["eval_metric"],
    )
    if objective == "reg:pseudohubererror":
        params["huber_slope"] = model_cfg.get("huber_slope", 1.0)

    return xgb.XGBRegressor(**params)
