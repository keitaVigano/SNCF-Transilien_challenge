"""Hyperparameter search for the XGBoost pipeline using Optuna.

Mirrors the SELM/DELM tune.py on feature/selm: TPE search over n_trials,
one run dir per trial under models/<study>/trial_NNNN/ with the same
artifacts a normal run produces (config.yaml, train.log, metrics.csv,
TensorBoard event files), best combination written to
models/<study>/best_config.yaml.

Unlike that tune.py, there's no "short trial budget vs. full final budget"
split here: `n_estimators` is just a generous upper bound and
`early_stopping_rounds` finds the right number of boosting rounds on its
own, every time, trial or final run alike -- so a trial's result is
already the number you'd get retraining standalone with the same params
(no separate "copy the full budget back in" step needed, and no
cross-trial RNG coupling to worry about either: each trial builds its own
fresh XGBRegressor seeded independently from cfg["seed"], not a shared
in-process generator like SELM's torch RNG).

Usage:
    python tune.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import optuna
import yaml

from main import load_datasets
from model import build_model
from trainer import Trainer
from utils import load_config, make_run_dir, set_seed, setup_logging

logger = logging.getLogger(__name__)


def suggest_hyperparameters(trial: optuna.Trial, base_cfg: dict) -> dict:
    """Samples one hyperparameter configuration for this trial, starting
    from the base config and overriding only the entries being searched.
    """
    trial_cfg = copy.deepcopy(base_cfg)
    model_cfg = trial_cfg["model"]

    model_cfg["max_depth"] = trial.suggest_int("max_depth", 3, 12)
    model_cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
    model_cfg["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    model_cfg["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    model_cfg["min_child_weight"] = trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True)
    model_cfg["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
    model_cfg["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
    model_cfg["gamma"] = trial.suggest_float("gamma", 1e-8, 5.0, log=True)

    # Set tune.fixed_objective (one of the three reg:* strings) to pin the
    # objective for every trial instead of searching it -- same idea as
    # SELM's tune.fixed_loss, to compare e.g. "best squarederror config" vs
    # "best absoluteerror config" on an equal search budget.
    tune_cfg = base_cfg["tune"]
    fixed_objective = tune_cfg.get("fixed_objective")
    model_cfg["objective"] = fixed_objective if fixed_objective else trial.suggest_categorical(
        "objective", ["reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"]
    )
    if model_cfg["objective"] == "reg:pseudohubererror":
        model_cfg["huber_slope"] = trial.suggest_float("huber_slope", 0.1, 10.0, log=True)

    trial_cfg["train"]["use_tensorboard"] = tune_cfg.get("use_tensorboard", False)
    return trial_cfg


def objective(
    trial: optuna.Trial,
    base_cfg: dict,
    study_dir: Path,
    X_train,
    y_train,
    X_val,
    y_val,
) -> float:
    trial_cfg = suggest_hyperparameters(trial, base_cfg)
    trial_dir = study_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(trial_dir)
    with open(trial_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(trial_cfg, f, sort_keys=False)
    logger.info("Trial %d params: %s", trial.number, trial.params)

    model = build_model(trial_cfg)
    trainer = Trainer(model, trial_cfg["train"], trial_dir)
    trainer.fit(X_train, y_train, X_val, y_val)
    val_mae, val_rmse = trainer.evaluate(X_val, y_val)
    logger.info("Trial %d done: val MAE=%.4f RMSE=%.4f", trial.number, val_mae, val_rmse)
    return val_mae


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    study_dir = make_run_dir(Path(cfg["output"]["models_dir"]) / "optuna")
    setup_logging(study_dir)
    logger.info("Study directory: %s", study_dir)

    _, X_train, y_train, X_val, y_val = load_datasets(cfg)

    tune_cfg = cfg["tune"]
    study = optuna.create_study(
        study_name=tune_cfg.get("study_name", "xgb_tuning"),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
    )
    study.optimize(
        lambda trial: objective(trial, cfg, study_dir, X_train, y_train, X_val, y_val),
        n_trials=tune_cfg["n_trials"],
    )

    # Every trial's objective() call re-points the root logger at its own
    # directory via setup_logging(); point it back at the study dir now that
    # all trials are done, so the summary below lands in the right log file.
    setup_logging(study_dir)
    logger.info("Best trial: #%d, val MAE=%.4f", study.best_trial.number, study.best_value)
    logger.info("Best params: %s", study.best_params)

    best_cfg = copy.deepcopy(cfg)
    best_cfg["model"].update({k: v for k, v in study.best_params.items() if k in best_cfg["model"]})
    best_config_path = study_dir / "best_config.yaml"
    with open(best_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)
    logger.info(
        "Wrote %s -- copy its model section into config.yaml to retrain with these settings.",
        best_config_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters with Optuna.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)
