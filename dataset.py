"""Preprocessing for the SNCF Transilien delay data, tailored to XGBoost.

Unlike the SELM/DELM pipeline (feature/selm branch), there's no numeric
standardization (tree splits are invariant to monotonic rescaling of a
numeric column) and no embedding vocabulary: categorical columns (station,
day-of-week) use XGBoost's own native categorical split support
(`enable_categorical=True`, see model.py) instead.

Input layout (same as the SELM pipeline):
    train, gare, date, arret, p2q0, p3q0, p4q0, p0q2, p0q3, p0q4
Target: p0q0 (the delay to predict at the current checkpoint).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class FeatureBuilder:
    """Fits each categorical column's vocabulary on the training split and
    applies it consistently to val/test.

    Unseen categories become NaN rather than raising: XGBoost's histogram
    splitter already routes missing values down whichever branch minimizes
    loss (learned at training time), so NaN is a reasonable fallback for a
    station never seen in training -- no separate "unknown" bucket needed.

    `.where(isin(...))` before `.astype(CategoricalDtype(...))` is
    deliberate: constructing a Categorical whose values aren't all in its
    declared categories is deprecated in pandas (as of pandas 3.x) and
    raises an error in XGBoost's predict() if those out-of-vocabulary
    values aren't already NaN first.

    Optionally also adds *target-encoded* numeric versions of the
    categorical columns (`<col>_te`) alongside the native categorical ones
    -- a smoothed per-category mean of the target, which lets the tree use
    a single numeric split instead of several categorical ones, and is a
    common complement to (not strictly a replacement for) native
    categorical splits. Smoothing pulls rare categories' encoding toward
    the global mean (the "m-estimate": (n*mean + m*global_mean)/(n+m), m =
    `te_smoothing`) so a station seen twice in training doesn't just
    memorize its own two delays.

    Leakage: a row's target-encoded value must never be computed from its
    OWN target, or the model partially memorizes per-row labels (worse for
    rare categories, where one row can dominate its own category's mean).
    Val/test are safe either way (their target is never used to fit
    anything), so `transform()` uses the plain full-train smoothed stats.
    TRAIN rows use `fit_transform_train()` instead, which K-fold encodes
    them (each fold's rows get the smoothed means of the *other* folds),
    matching the standard anti-leakage recipe for in-sample target
    encoding.
    """

    def __init__(
        self,
        categorical_cols: list[str],
        numeric_cols: list[str],
        date_col: str | None = None,
        use_date_features: bool = True,
        target_col: str | None = None,
        use_target_encoding: bool = False,
        te_smoothing: float = 20.0,
        te_n_folds: int = 5,
        seed: int = 42,
    ) -> None:
        self.categorical_cols = list(categorical_cols)
        self.numeric_cols = list(numeric_cols)
        self.date_col = date_col
        self.use_date_features = use_date_features and date_col is not None
        self.target_col = target_col
        self.use_target_encoding = use_target_encoding and target_col is not None
        self.te_smoothing = te_smoothing
        self.te_n_folds = te_n_folds
        self.seed = seed

        self.categories_: dict[str, list[Any]] = {}
        self.te_stats_: dict[str, dict[str, Any]] = {}

    @property
    def all_categorical_cols(self) -> list[str]:
        cols = list(self.categorical_cols)
        if self.use_date_features:
            cols.append("dow")
        return cols

    @property
    def all_feature_cols(self) -> list[str]:
        cols = list(self.numeric_cols) + self.all_categorical_cols
        if self.use_date_features:
            cols.append("is_weekend")
        if self.use_target_encoding:
            cols += [f"{col}_te" for col in self.all_categorical_cols]
        return cols

    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dow = pd.to_datetime(df[self.date_col]).dt.dayofweek
        # Stored as strings (not ints): day-of-week has no meaningful order
        # for delay behavior, so it's treated as a plain category like
        # `gare`, not as an ordinal numeric feature.
        df["dow"] = dow.astype(str)
        df["is_weekend"] = (dow >= 5).astype(np.float32)
        return df

    def _apply_categorical_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.all_categorical_cols:
            cats = self.categories_[col]
            df[col] = df[col].where(df[col].isin(cats)).astype(pd.CategoricalDtype(categories=cats))
        return df

    def _smoothed_means(self, df: pd.DataFrame, col: str) -> tuple[dict[Any, float], float]:
        global_mean = float(df[self.target_col].mean())
        grouped = df.groupby(col)[self.target_col].agg(["mean", "count"])
        smoothed = (grouped["count"] * grouped["mean"] + self.te_smoothing * global_mean) / (
            grouped["count"] + self.te_smoothing
        )
        return smoothed.to_dict(), global_mean

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        if self.use_date_features:
            df = self._add_date_features(df)
        for col in self.all_categorical_cols:
            self.categories_[col] = sorted(df[col].dropna().unique().tolist(), key=str)
        if self.use_target_encoding:
            for col in self.all_categorical_cols:
                cat_means, global_mean = self._smoothed_means(df, col)
                self.te_stats_[col] = {"map": cat_means, "global_mean": global_mean}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.categories_:
            raise RuntimeError("FeatureBuilder.fit() must be called before transform().")
        if self.use_date_features:
            df = self._add_date_features(df)
        df = self._apply_categorical_dtypes(df)
        if self.use_target_encoding:
            for col in self.all_categorical_cols:
                stats = self.te_stats_[col]
                df[f"{col}_te"] = df[col].map(stats["map"]).fillna(stats["global_mean"]).astype(np.float32)
        return df[self.all_feature_cols]

    def fit_transform_train(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """fit() + transform(), except the target-encoded columns use
        K-fold (not plain in-sample) smoothed means -- see class
        docstring. Only meaningful difference from `fit(train_df)` followed
        by `transform(train_df)`; everything else (vocab, dtypes) is
        identical.
        """
        self.fit(train_df)
        df = train_df.copy()
        if self.use_date_features:
            df = self._add_date_features(df)
        df = self._apply_categorical_dtypes(df)

        if self.use_target_encoding:
            rng = np.random.RandomState(self.seed)
            fold_ids = rng.permutation(len(df)) % self.te_n_folds
            for col in self.all_categorical_cols:
                encoded = np.empty(len(df), dtype=np.float32)
                for k in range(self.te_n_folds):
                    in_fold = fold_ids == k
                    cat_means, global_mean = self._smoothed_means(df.loc[~in_fold], col)
                    encoded[in_fold] = (
                        df.loc[in_fold, col].map(cat_means).fillna(global_mean).astype(np.float32)
                    )
                df[f"{col}_te"] = encoded
        return df[self.all_feature_cols]


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
