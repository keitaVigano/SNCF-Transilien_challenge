"""Orchestrates the XGBoost train-delay-prediction pipeline: load data,
split, build features, train with early stopping, evaluate, and predict
on the held-out test set.

Usage:
    python main.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import shutil

import pandas as pd

from dataset import FeatureBuilder, chronological_split
from model import build_model
from trainer import Trainer
from utils import load_config, make_run_dir, set_seed, setup_logging

logger = logging.getLogger(__name__)


def _read_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")


def load_datasets(
    cfg: dict,
) -> tuple[FeatureBuilder, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Reads x_train/y_train, splits chronologically, optionally drops
    extreme-outlier TRAIN rows, and fits/applies the FeatureBuilder. Shared
    by main.py (one real run) and tune.py (many trials against the same
    data).
    """
    data_cfg = cfg["data"]

    x_train_full = _read_features(data_cfg["x_train_path"])
    y_train_full = pd.read_csv(data_cfg["y_train_path"], index_col=0)
    df = x_train_full.join(y_train_full)

    train_df, val_df = chronological_split(df, data_cfg["date_col"], data_cfg["val_fraction"])
    logger.info("Train rows: %d, val rows: %d", len(train_df), len(val_df))

    # Optional: drop TRAIN rows whose true delay is an extreme, one-off
    # disruption (e.g. the 2023-11-10 network-wide event found by
    # analysis/error_analysis.py on feature/selm -- |p0q0| up to 93 min,
    # dwarfing the typical <2 min error, with nothing in the lag features
    # signaling it in advance). Applied after the split, so val is
    # untouched: this measures whether a model trained without those
    # unlearnable shocks still generalizes to the real
    # (disruption-included) validation distribution.
    max_abs_target = data_cfg.get("max_abs_target")
    if max_abs_target is not None:
        target_col = data_cfg["target_col"]
        before = len(train_df)
        train_df = train_df[train_df[target_col].abs() <= max_abs_target]
        logger.info(
            "Dropped %d/%d train rows with |%s| > %s (max_abs_target filter, val untouched)",
            before - len(train_df), before, target_col, max_abs_target,
        )

    target_col = data_cfg["target_col"]
    feature_builder = FeatureBuilder(
        categorical_cols=data_cfg["categorical_cols"],
        numeric_cols=data_cfg["numeric_cols"],
        date_col=data_cfg["date_col"],
        use_date_features=data_cfg["use_date_features"],
        target_col=target_col,
        use_target_encoding=data_cfg.get("use_target_encoding", False),
        te_smoothing=data_cfg.get("target_encoding_smoothing", 20.0),
        te_n_folds=data_cfg.get("target_encoding_folds", 5),
        seed=cfg["seed"],
    )
    # fit_transform_train K-fold encodes the target-encoded columns for
    # TRAIN rows (anti-leakage, see dataset.py:FeatureBuilder); val/test
    # go through the plain transform(), using the full-train smoothed
    # stats fit() stored as a side effect.
    X_train = feature_builder.fit_transform_train(train_df)
    X_val = feature_builder.transform(val_df)
    y_train = train_df[target_col]
    y_val = val_df[target_col]
    return feature_builder, X_train, y_train, X_val, y_val


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    run_dir = make_run_dir(cfg["output"]["models_dir"])
    setup_logging(run_dir)
    shutil.copy(config_path, run_dir / "config.yaml")
    logger.info("Run directory: %s", run_dir)

    data_cfg = cfg["data"]
    feature_builder, X_train, y_train, X_val, y_val = load_datasets(cfg)

    model = build_model(cfg)
    trainer = Trainer(model, cfg["train"], run_dir)
    trainer.fit(X_train, y_train, X_val, y_val)

    val_mae, val_rmse = trainer.evaluate(X_val, y_val)
    logger.info("Final validation MAE=%.4f RMSE=%.4f", val_mae, val_rmse)

    target_col = data_cfg["target_col"]
    x_test = _read_features(data_cfg["x_test_path"])
    X_test = feature_builder.transform(x_test)

    predictions = trainer.predict(X_test)
    submission = pd.DataFrame({target_col: predictions}, index=x_test.index)
    predictions_path = run_dir / cfg["output"]["predictions_filename"]
    submission.to_csv(predictions_path)
    logger.info("Wrote predictions to %s", predictions_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the XGBoost train-delay model.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)
