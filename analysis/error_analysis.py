"""Error analysis for a completed SELM/DELM run: rebuilds that run's exact
train/val split and preprocessing, reloads its saved checkpoint via
main.build_model (so it works for either architecture), and looks at where
the model is wrong on the validation set (which has real p0q0 labels --
the real test set doesn't).

Usage:
    python analysis/error_analysis.py --run models/20260922_183608
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\keita\git\SNCF-Transilien_challenge")
SCRATCH_DIR = Path(__file__).parent

sys.path.insert(0, str(PROJECT_DIR))  # dataset.py, model.py, main.py (current working tree)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

import dataset as ds
from main import build_model

sns.set_theme(style="whitegrid")


def read_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")


def load_run(run_dir: Path):
    with open(run_dir / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]

    x_train_full = read_features(PROJECT_DIR / data_cfg["x_train_path"])
    y_train_full = pd.read_csv(PROJECT_DIR / data_cfg["y_train_path"], index_col=0)
    df = x_train_full.join(y_train_full)

    train_df, val_df = ds.chronological_split(df, data_cfg["date_col"], data_cfg["val_fraction"])
    print(f"train rows: {len(train_df)}, val rows: {len(val_df)}")
    print(f"val date range: {val_df[data_cfg['date_col']].min()} .. {val_df[data_cfg['date_col']].max()}")

    preprocessor = ds.Preprocessor(
        categorical_cols=data_cfg["categorical_cols"],
        numeric_cols=data_cfg["numeric_cols"],
        date_col=data_cfg["date_col"],
        use_date_features=data_cfg["use_date_features"],
    ).fit(train_df)
    val_ds = ds.TransilienDelayDataset.from_dataframe(val_df, preprocessor, data_cfg["target_col"])

    model = build_model(cfg, preprocessor)
    # The saved checkpoint's state_dict already has the DELM encoders fit by
    # the original run (and DELM.load_state_dict marks the model as fitted);
    # for SELM this is just the usual embeddings + output weights.
    model.load_state_dict(torch.load(run_dir / "selm_sgd.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds = model(val_ds.x_num, val_ds.x_cat).squeeze(1).numpy()

    out = val_df.reset_index(drop=True).copy()
    out["pred"] = preds
    out["residual"] = out[data_cfg["target_col"]] - preds  # positive => model under-predicted the delay
    out["abs_error"] = out["residual"].abs()
    dow = pd.to_datetime(out["date"]).dt.dayofweek
    out["dow"] = dow
    out["is_weekend"] = dow >= 5
    return out, cfg


def main(run_dir: Path) -> None:
    analysis_df, cfg = load_run(run_dir)
    n = len(analysis_df)
    mae = analysis_df["abs_error"].mean()
    rmse = np.sqrt((analysis_df["residual"] ** 2).mean())
    print(f"Recomputed val MAE={mae:.4f} RMSE={rmse:.4f}")

    worst = analysis_df.sort_values("abs_error", ascending=False).head(25)
    print("\n=== 25 peggiori osservazioni (per errore assoluto) ===")
    cols = ["train", "gare", "date", "arret", "p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4", "p0q0", "pred", "residual"]
    print(worst[cols].to_string(index=False))

    print("\n=== Quantili errore assoluto ===")
    print(analysis_df["abs_error"].quantile([0.5, 0.75, 0.9, 0.95, 0.99, 1.0]))

    print("\n=== Quota errore per fascia ===")
    big = (analysis_df["abs_error"] > 5).sum()
    print(f"osservazioni con |errore| > 5 min: {big} ({100*big/n:.2f}% del validation set)")
    print(
        f"di queste, contributo alla somma degli errori assoluti: "
        f"{100*analysis_df.loc[analysis_df['abs_error']>5,'abs_error'].sum()/analysis_df['abs_error'].sum():.1f}%"
    )

    station_err = (
        analysis_df.groupby("gare")
        .agg(n=("abs_error", "size"), mean_abs_error=("abs_error", "mean"), mean_delay=("p0q0", "mean"))
        .sort_values("mean_abs_error", ascending=False)
    )
    print("\n=== Top 10 stazioni per errore medio (>=20 osservazioni) ===")
    print(station_err[station_err["n"] >= 20].head(10))

    print("\n=== Errore medio assoluto per giorno della settimana ===")
    print(analysis_df.groupby("dow")["abs_error"].mean())

    print("\n=== Correlazione |errore| con il ritardo reale (p0q0) ===")
    print(np.corrcoef(analysis_df["abs_error"], analysis_df["p0q0"].abs())[0, 1])

    by_date = (
        analysis_df.groupby("date")
        .agg(n=("abs_error", "size"), mean_ae=("abs_error", "mean"), max_ae=("abs_error", "max"))
        .sort_values("mean_ae", ascending=False)
    )
    print("\n=== Top 5 date per errore medio ===")
    print(by_date.head(5))

    # --------------------------------------------------------------- plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    sns.histplot(analysis_df["residual"], bins=80, ax=ax, color="#4C72B0")
    ax.set_title("Distribuzione dei residui (y_true - y_pred)")
    ax.set_xlabel("residuo (minuti)")
    ax.axvline(0, color="black", linewidth=1)

    ax = axes[0, 1]
    sample = analysis_df.sample(min(8000, n), random_state=0)
    ax.scatter(sample["p0q0"], sample["pred"], s=4, alpha=0.25, color="#4C72B0")
    lims = [analysis_df["p0q0"].min(), analysis_df["p0q0"].max()]
    ax.plot(lims, lims, color="red", linewidth=1, label="y=x (ideale)")
    ax.set_title("Predetto vs reale (val set)")
    ax.set_xlabel("ritardo reale p0q0 (min)")
    ax.set_ylabel("ritardo predetto (min)")
    ax.legend()

    ax = axes[0, 2]
    ax.scatter(sample["p0q0"], sample["residual"], s=4, alpha=0.25, color="#55A868")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Residuo vs ritardo reale")
    ax.set_xlabel("ritardo reale p0q0 (min)")
    ax.set_ylabel("residuo (min)")

    ax = axes[1, 0]
    top_stations = station_err[station_err["n"] >= 20].head(15)
    sns.barplot(x=top_stations["mean_abs_error"], y=top_stations.index, ax=ax, color="#C44E52")
    ax.set_title("Top 15 stazioni per MAE (>=20 oss.)")
    ax.set_xlabel("MAE (min)")
    ax.set_ylabel("gare")

    ax = axes[1, 1]
    sns.boxplot(x="dow", y="abs_error", data=analysis_df, ax=ax, showfliers=False, color="#8172B2")
    ax.set_title("Errore assoluto per giorno settimana (0=lun)")
    ax.set_xlabel("day of week")
    ax.set_ylabel("|errore| (min)")

    ax = axes[1, 2]
    bins = [0, 1, 2, 5, 10, 20, np.inf]
    analysis_df["delay_bucket"] = pd.cut(analysis_df["p0q0"].abs(), bins=bins)
    bucket_err = analysis_df.groupby("delay_bucket")["abs_error"].mean()
    bucket_err.plot(kind="bar", ax=ax, color="#937860")
    ax.set_title("MAE per fascia di ritardo reale |p0q0|")
    ax.set_xlabel("|ritardo reale| (min)")
    ax.set_ylabel("MAE (min)")
    ax.tick_params(axis="x", rotation=30)

    architecture = cfg["model"].get("architecture", "selm")
    fig.suptitle(f"Error analysis -- {run_dir.name} ({architecture}, val MAE={mae:.3f}, RMSE={rmse:.3f})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = run_dir / "error_analysis.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="models/20260930_215048", help="Path to a run dir (relative to the project).")
    args = parser.parse_args()
    main(PROJECT_DIR / args.run)
