"""Orchestrates the SELM train-delay-prediction pipeline: load data, split,
preprocess, train (closed-form ridge regression or SGD), evaluate, and
predict on the held-out test set.

Usage:
    python main.py --config config.yaml
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
from torch.utils.data import DataLoader

from dataset import Preprocessor, TransilienDelayDataset, chronological_split
from model import SELM
from trainer import Trainer
from utils import get_device, load_config, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _read_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    return df


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    logger.info("Using device: %s", device)

    data_cfg = cfg["data"]

    x_train_full = _read_features(data_cfg["x_train_path"])
    y_train_full = pd.read_csv(data_cfg["y_train_path"], index_col=0)
    df = x_train_full.join(y_train_full)

    train_df, val_df = chronological_split(df, data_cfg["date_col"], data_cfg["val_fraction"])
    logger.info("Train rows: %d, val rows: %d", len(train_df), len(val_df))

    preprocessor = Preprocessor(
        categorical_cols=data_cfg["categorical_cols"],
        numeric_cols=data_cfg["numeric_cols"],
        date_col=data_cfg["date_col"],
        use_date_features=data_cfg["use_date_features"],
    ).fit(train_df)

    target_col = data_cfg["target_col"]
    train_ds = TransilienDelayDataset.from_dataframe(train_df, preprocessor, target_col)
    val_ds = TransilienDelayDataset.from_dataframe(val_df, preprocessor, target_col)

    batch_size = cfg["train"]["batch_size"]
    num_workers = data_cfg["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model_cfg = cfg["model"]
    model = SELM(
        numeric_dim=len(preprocessor.all_numeric_cols),
        cardinalities=preprocessor.cardinalities_,
        embedding_dim=model_cfg["embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        activation=model_cfg["activation"],
        hidden_init=model_cfg["hidden_init"],
        hidden_init_range=tuple(model_cfg["hidden_init_range"]),
    )

    trainer = Trainer(model, device, cfg["train"], cfg["output"]["checkpoint_dir"])

    solver = cfg["train"]["solver"]
    if solver == "closed_form":
        trainer.fit_closed_form(train_loader)
    elif solver == "sgd":
        trainer.fit_sgd(train_loader, val_loader)
    else:
        raise ValueError(f"Unknown solver '{solver}', choose 'closed_form' or 'sgd'")

    val_mae, val_rmse = trainer.evaluate(val_loader)
    logger.info("Final validation MAE=%.4f RMSE=%.4f", val_mae, val_rmse)

    x_test = _read_features(data_cfg["x_test_path"])
    test_ds = TransilienDelayDataset.from_dataframe(x_test, preprocessor, target_col=None)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    predictions = trainer.predict(test_loader)
    submission = pd.DataFrame({target_col: predictions}, index=x_test.index)
    submission.to_csv(cfg["output"]["predictions_path"])
    logger.info("Wrote predictions to %s", cfg["output"]["predictions_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the SELM train-delay model.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)
