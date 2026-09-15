"""Hyperparameter search for SELM (solver: sgd) using Optuna.

The paper (Section 3.3) tunes its hyperparameters via random search over
n_MC=300 combinations rather than grid search, citing Bergstra & Bengio's
result that random search beats grid search when the hyperparameter space
is intractable to enumerate exhaustively. Optuna's default TPE sampler is a
smarter descendant of that same idea: it biases future trials using past
results instead of sampling uniformly at random, so it converges faster
while following the same "don't grid-search, search smartly instead" spirit.

Each trial gets its own run directory under models/<study>/trial_NNNN/,
with the exact same artifacts a normal training run produces (config.yaml,
train.log, metrics.csv, TensorBoard event files) -- so all trials can be
compared side by side in TensorBoard exactly like any other set of runs:
    tensorboard --logdir models/<study>

Usage:
    python tune.py --config config.yaml
"""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import optuna
import yaml

from dataset import TransilienDelayDataset
from main import build_loaders, build_model, load_datasets
from model import SELM
from trainer import Trainer
from utils import get_device, load_config, make_run_dir, set_seed, setup_logging

logger = logging.getLogger(__name__)


def suggest_hyperparameters(trial: optuna.Trial, base_cfg: dict) -> dict:
    """Samples one hyperparameter configuration for this trial, starting
    from the base config and overriding only the entries being searched.
    """
    trial_cfg = copy.deepcopy(base_cfg)

    trial_cfg["model"]["hidden_dim"] = trial.suggest_categorical("hidden_dim", [256, 512, 1024, 2048, 4096])
    trial_cfg["model"]["embedding_dim"] = trial.suggest_categorical("embedding_dim", [4, 8, 16, 32])
    trial_cfg["model"]["activation"] = trial.suggest_categorical("activation", ["tanh", "sigmoid", "relu"])
    trial_cfg["train"]["lr"] = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
    trial_cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

    tune_cfg = base_cfg["tune"]
    trial_cfg["train"]["epochs"] = tune_cfg["epochs_per_trial"]
    trial_cfg["train"]["early_stopping_patience"] = tune_cfg["early_stopping_patience_per_trial"]
    # Trials are short by design (few epochs) -- TensorBoard is only useful
    # for comparing the final, long real run, so skip it here to save disk.
    trial_cfg["train"]["use_tensorboard"] = tune_cfg.get("use_tensorboard", False)
    return trial_cfg


def objective(
    trial: optuna.Trial,
    base_cfg: dict,
    study_dir: Path,
    device,
    preprocessor,
    train_ds: TransilienDelayDataset,
    val_ds: TransilienDelayDataset,
) -> float:
    trial_cfg = suggest_hyperparameters(trial, base_cfg)
    trial_dir = study_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(trial_dir)
    with open(trial_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(trial_cfg, f, sort_keys=False)
    logger.info("Trial %d params: %s", trial.number, trial.params)

    train_loader, val_loader = build_loaders(trial_cfg, train_ds, val_ds)
    model: SELM = build_model(trial_cfg, preprocessor)
    trainer = Trainer(model, device, trial_cfg["train"], trial_dir)
    trainer.fit_sgd(train_loader, val_loader)
    val_mae, val_rmse = trainer.evaluate(val_loader)
    logger.info("Trial %d done: val MAE=%.4f RMSE=%.4f", trial.number, val_mae, val_rmse)
    return val_mae


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])

    study_dir = make_run_dir(Path(cfg["output"]["models_dir"]) / "optuna")
    setup_logging(study_dir)
    logger.info("Study directory: %s", study_dir)
    logger.info("Using device: %s", device)

    preprocessor, train_ds, val_ds = load_datasets(cfg)

    tune_cfg = cfg["tune"]
    study = optuna.create_study(
        study_name=tune_cfg.get("study_name", "selm_tuning"),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
    )
    study.optimize(
        lambda trial: objective(trial, cfg, study_dir, device, preprocessor, train_ds, val_ds),
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
    best_cfg["train"].update({k: v for k, v in study.best_params.items() if k in best_cfg["train"]})
    best_config_path = study_dir / "best_config.yaml"
    with open(best_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)
    logger.info(
        "Wrote %s -- copy its model/train sections into config.yaml (keeping the full "
        "epochs/patience budget) to retrain with these settings.",
        best_config_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune SELM hyperparameters with Optuna.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)
