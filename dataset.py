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
from workalendar.europe import France

UNKNOWN_TOKEN = "__unknown__"

# The six pre-aggregated delay lag/lead columns the challenge provides for
# every row (train or test) -- never the target, so always safe to reuse in
# groupby-based features.
LAG_COLS = ["p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]

_FRANCE_CALENDAR = France()


class Preprocessor:
    """Fits label encoders / a scaler on the training split and applies them
    consistently to validation/test data (unseen categories fall back to a
    reserved "unknown" index instead of raising).

    Beyond the raw columns, it derives three families of engineered
    features, all safe to compute on train, val, *and* the label-less test
    set (none of them ever touch the target column p0q0 except where noted):

      * date features (`_add_date_features`): day-of-week, weekend flag,
        and a French public-holiday flag (via `workalendar`) -- stateless,
        computed fresh from `date_col` on whichever dataframe is passed in.
      * same-day traffic features (`_add_traffic_features`): how many
        trains passed through this station on this date, and the average
        of the *other* trains' own lag features (p2q0, p3q0, ...) at the
        same station/date -- also stateless, computed independently per
        dataframe via a groupby on columns that are always present (gare,
        date, and the lag columns themselves are never the target).
      * station history features (`_fit_station_stats` / `_apply_station_stats`):
        the historical average delay (and of each lag column) per station,
        fit once on the training split and looked up by station name for
        any dataframe -- same fit-then-lookup pattern as the categorical
        vocabulary below, since this one does use the target.
    """

    def __init__(
        self,
        categorical_cols: list[str],
        numeric_cols: list[str],
        date_col: str | None = None,
        use_date_features: bool = True,
        target_col: str | None = None,
    ) -> None:
        self.categorical_cols = list(categorical_cols)
        self.numeric_cols = list(numeric_cols)
        self.date_col = date_col
        self.use_date_features = use_date_features and date_col is not None
        self.target_col = target_col

        self.vocab_: dict[str, dict[Any, int]] = {}
        self.cardinalities_: dict[str, int] = {}
        self.num_mean_: np.ndarray | None = None
        self.num_std_: np.ndarray | None = None

        self.station_target_mean_: dict[Any, float] = {}
        self.station_lag_means_: dict[str, dict[Any, float]] = {}
        self.global_target_mean_: float = 0.0
        self.global_lag_means_: dict[str, float] = {}

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
            cols += ["is_weekend", "is_holiday"]
        cols.append("station_traffic")
        cols += [f"other_trains_avg_{c}" for c in LAG_COLS]
        if self.target_col is not None:
            cols.append("station_target_mean")
            cols += [f"station_hist_mean_{c}" for c in LAG_COLS]
        return cols

    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dates = pd.to_datetime(df[self.date_col])
        dow = dates.dt.dayofweek
        df["dow"] = dow
        df["is_weekend"] = (dow >= 5).astype(np.float32)
        df["is_holiday"] = dates.dt.date.map(_FRANCE_CALENDAR.is_holiday).astype(np.float32)
        return df

    def _add_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Same-day, same-station aggregates computed fresh from `df` alone
        (no train/test lookup needed -- see the class docstring).
        """
        df = df.copy()
        group = df.groupby(["gare", self.date_col])
        df["station_traffic"] = group["gare"].transform("size")
        for col in LAG_COLS:
            col_group = group[col]
            group_sum = col_group.transform("sum")
            group_count = col_group.transform("count")
            denom = (group_count - 1).clip(lower=1)  # avoid div-by-zero when a train is alone in its group
            df[f"other_trains_avg_{col}"] = (group_sum - df[col]) / denom
        return df

    def _fit_station_stats(self, df: pd.DataFrame) -> None:
        self.global_target_mean_ = float(df[self.target_col].mean())
        self.station_target_mean_ = df.groupby("gare")[self.target_col].mean().to_dict()
        for col in LAG_COLS:
            self.global_lag_means_[col] = float(df[col].mean())
            self.station_lag_means_[col] = df.groupby("gare")[col].mean().to_dict()

    def _apply_station_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["station_target_mean"] = (
            df["gare"].map(self.station_target_mean_).fillna(self.global_target_mean_).astype(np.float32)
        )
        for col in LAG_COLS:
            df[f"station_hist_mean_{col}"] = (
                df["gare"].map(self.station_lag_means_[col]).fillna(self.global_lag_means_[col]).astype(np.float32)
            )
        return df

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        if self.use_date_features:
            df = self._add_date_features(df)
        df = self._add_traffic_features(df)
        if self.target_col is not None:
            self._fit_station_stats(df)
            df = self._apply_station_stats(df)

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
        df = self._add_traffic_features(df)
        if self.target_col is not None:
            df = self._apply_station_stats(df)

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
