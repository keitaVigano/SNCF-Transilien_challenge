"""Preprocessing and torch Dataset for the SNCF Transilien delay data.

Input layout (see analysis.ipynb / README of the challenge):
    train, gare, date, arret, p2q0, p3q0, p4q0, p0q2, p0q3, p0q4
Target: p0q0 (the delay to predict at the current checkpoint).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

UNKNOWN_TOKEN = "__unknown__"


class Preprocessor:
    """Fits label encoders / a scaler on the training split and applies them
    consistently to validation/test data (unseen categories fall back to a
    reserved "unknown" index instead of raising).
    """

    def __init__(
        self,
        categorical_cols: list[str],
        numeric_cols: list[str],
        date_col: str | None = None,
        use_date_features: bool = True,
    ) -> None:
        self.categorical_cols = list(categorical_cols)
        self.numeric_cols = list(numeric_cols)
        self.date_col = date_col
        self.use_date_features = use_date_features and date_col is not None

        self.vocab_: dict[str, dict[Any, int]] = {}
        self.cardinalities_: dict[str, int] = {}
        self.num_mean_: np.ndarray | None = None
        self.num_std_: np.ndarray | None = None

    @property
    def all_categorical_cols(self) -> list[str]:
        cols = list(self.categorical_cols)
        if self.use_date_features:
            cols.append("dow")
        return cols

    @property
    def all_numeric_cols(self) -> list[str]:
        cols = list(self.numeric_cols)
        if self.use_date_features:
            cols.append("is_weekend")
        return cols

    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dow = pd.to_datetime(df[self.date_col]).dt.dayofweek
        df["dow"] = dow
        df["is_weekend"] = (dow >= 5).astype(np.float32)
        return df

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        if self.use_date_features:
            df = self._add_date_features(df)

        for col in self.all_categorical_cols:
            categories = pd.unique(df[col])
            vocab = {cat: i for i, cat in enumerate(categories)}
            vocab[UNKNOWN_TOKEN] = len(vocab)
            self.vocab_[col] = vocab
            self.cardinalities_[col] = len(vocab)

        numeric = df[self.all_numeric_cols].to_numpy(dtype=np.float32)
        self.num_mean_ = numeric.mean(axis=0)
        self.num_std_ = numeric.std(axis=0)
        self.num_std_[self.num_std_ == 0] = 1.0
        return self

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.num_mean_ is None:
            raise RuntimeError("Preprocessor.fit() must be called before transform().")
        if self.use_date_features:
            df = self._add_date_features(df)

        x_cat = {
            col: np.array(
                df[col].map(lambda v, c=col: self.vocab_[c].get(v, self.vocab_[c][UNKNOWN_TOKEN])),
                dtype=np.int64,
                copy=True,
            )
            for col in self.all_categorical_cols
        }
        numeric = df[self.all_numeric_cols].to_numpy(dtype=np.float32)
        x_num = (numeric - self.num_mean_) / self.num_std_
        return {"x_num": x_num.astype(np.float32), "x_cat": x_cat}


class TransilienDelayDataset(Dataset):
    """Wraps preprocessed arrays into a torch Dataset. `y` is optional so the
    same class serves the label-less test set.
    """

    def __init__(self, x_num: np.ndarray, x_cat: dict[str, np.ndarray], y: np.ndarray | None = None):
        self.x_num = torch.from_numpy(x_num).float()
        self.x_cat = {k: torch.from_numpy(v).long() for k, v in x_cat.items()}
        self.y = torch.from_numpy(y).float().unsqueeze(1) if y is not None else None

    def __len__(self) -> int:
        return self.x_num.shape[0]

    def __getitem__(self, idx: int):
        cat = {k: v[idx] for k, v in self.x_cat.items()}
        if self.y is not None:
            return self.x_num[idx], cat, self.y[idx]
        return self.x_num[idx], cat

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, preprocessor: Preprocessor, target_col: str | None = None
    ) -> "TransilienDelayDataset":
        transformed = preprocessor.transform(df)
        y = df[target_col].to_numpy(dtype=np.float32) if target_col is not None else None
        return cls(transformed["x_num"], transformed["x_cat"], y)


def chronological_split(df: pd.DataFrame, date_col: str, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits on the most recent dates so validation mimics forecasting the
    future, rather than a random (leaky) split of a time-ordered dataset.
    """
    dates = pd.to_datetime(df[date_col])
    unique_dates = np.sort(dates.unique())
    n_val_dates = max(1, int(round(len(unique_dates) * val_fraction)))
    cutoff = unique_dates[-n_val_dates]
    train_df = df[dates < cutoff]
    val_df = df[dates >= cutoff]
    return train_df, val_df
