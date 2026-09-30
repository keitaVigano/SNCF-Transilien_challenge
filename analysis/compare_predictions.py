"""Load a trained SELM checkpoint, rebuild its exact val split/preprocessing
from its own config.yaml, and plot predicted vs. real delay.

Usage:
    python compare_predictions.py --run models/20260918_182634
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\keita\git\SNCF-Transilien_challenge")
sys.path.insert(0, str(PROJECT_DIR))  # dataset.py, model.py must match the checkpoint's shape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

import dataset as ds
from model import SELM


def read_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")


def load_run(run_dir: Path) -> pd.DataFrame:
    """Returns the validation dataframe with two extra columns: pred, residual.

    NOTE: if this raises a state_dict size mismatch, the checkpoint was
    trained with a different dataset.py than the one currently on disk
    (check `git log -p -- dataset.py` / `git stash list` for the version
    that matches -- the tensor shape in the error is ground truth, not the
    file's mtime).
    """
    cfg = yaml.safe_load(open(run_dir / "config.yaml", "r", encoding="utf-8"))
    data_cfg, model_cfg = cfg["data"], cfg["model"]

    x = read_features(PROJECT_DIR / data_cfg["x_train_path"])
    y = pd.read_csv(PROJECT_DIR / data_cfg["y_train_path"], index_col=0)
    df = x.join(y)
    train_df, val_df = ds.chronological_split(df, data_cfg["date_col"], data_cfg["val_fraction"])

    preprocessor = ds.Preprocessor(
        categorical_cols=data_cfg["categorical_cols"],
        numeric_cols=data_cfg["numeric_cols"],
        date_col=data_cfg["date_col"],
        use_date_features=data_cfg["use_date_features"],
    ).fit(train_df)
    val_ds = ds.TransilienDelayDataset.from_dataframe(val_df, preprocessor, data_cfg["target_col"])

    model = SELM(
        numeric_dim=len(preprocessor.all_numeric_cols),
        cardinalities=preprocessor.cardinalities_,
        embedding_dim=model_cfg["embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        activation=model_cfg["activation"],
        hidden_init=model_cfg["hidden_init"],
        hidden_init_range=tuple(model_cfg["hidden_init_range"]),
    )
    model.load_state_dict(torch.load(run_dir / "selm_sgd.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds = model(val_ds.x_num, val_ds.x_cat).squeeze(1).numpy()

    out = val_df.reset_index(drop=True).copy()
    out["pred"] = preds
    out["residual"] = out[data_cfg["target_col"]] - preds
    return out


def plot_pred_vs_true(df: pd.DataFrame, target_col: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    sample = df.sample(min(8000, len(df)), random_state=0)
    ax.scatter(sample[target_col], sample["pred"], s=5, alpha=0.25, color="#4C72B0")
    lims = [df[target_col].min(), df[target_col].max()]
    ax.plot(lims, lims, color="red", linewidth=1, label="y = x (ideale)")
    ax.set_xlabel(f"{target_col} reale (min)")
    ax.set_ylabel(f"{target_col} predetto (min)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="models/20260918_182634", help="Path to a run dir (relative to the project).")
    parser.add_argument("--out", default=None, help="Where to save the PNG (default: <run>/pred_vs_true.png).")
    args = parser.parse_args()

    run_dir = PROJECT_DIR / args.run
    cfg = yaml.safe_load(open(run_dir / "config.yaml", "r", encoding="utf-8"))
    target_col = cfg["data"]["target_col"]

    analysis_df = load_run(run_dir)
    mae = analysis_df["residual"].abs().mean()
    rmse = np.sqrt((analysis_df["residual"] ** 2).mean())
    print(f"val MAE={mae:.4f} RMSE={rmse:.4f} n={len(analysis_df)}")

    out_path = Path(args.out) if args.out else run_dir / "pred_vs_true.png"
    plot_pred_vs_true(analysis_df, target_col, f"{run_dir.name} -- val MAE={mae:.3f}", out_path)
