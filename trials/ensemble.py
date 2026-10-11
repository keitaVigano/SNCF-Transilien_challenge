"""
SNCF Transilien Waiting Time Forecast (ENS Challenge 166)
Self-contained pipeline: data loading, feature engineering, chronological
CV, 3-model ensemble (LightGBM / XGBoost / CatBoost), threshold-tuned
blending, and submission export.

Only external dependencies: pandas, numpy, scikit-learn, scipy,
lightgbm, xgboost, catboost. No other project files required.
"""

import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import xgboost_train as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_DIR = "data"
OUTPUT_PATH = "data/submission_ensemble.csv"
N_SPLITS = 5


# ==========================================================================
# 1. Data loading
# ==========================================================================
class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_data(self):
        x_train = pd.read_csv(os.path.join(self.data_dir, "x_train_final.csv"), index_col=0)
        x_train.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
        y_train = pd.read_csv(os.path.join(self.data_dir, "y_train_final.csv"), index_col=0)
        x_test = pd.read_csv(os.path.join(self.data_dir, "x_test_final.csv"), index_col=0)
        x_test.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

        df_train = x_train.copy()
        df_train["date"] = pd.to_datetime(df_train["date"])
        df_train["p0q0"] = y_train.iloc[:, 0].values

        df_test = x_test.copy()
        df_test["date"] = pd.to_datetime(df_test["date"])

        return df_train, df_test, y_train


# ==========================================================================
# 2. Feature engineering
# ==========================================================================
class FeatureEngineer:
    def __init__(self):
        self.cat_cols = ["gare", "mission_prefix1", "mission_prefix2", "mission_prefix3"]

    @staticmethod
    def _add_mission_features(df):
        df = df.copy()
        train_str = df["train"].astype(str)
        df["mission_prefix1"] = train_str.str[:1]
        df["mission_prefix2"] = train_str.str[:2]
        df["mission_prefix3"] = train_str.str[:3]
        return df

    @staticmethod
    def _add_calendar_features(df):
        df = df.copy()
        df["dow"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["dayofyear"] = df["date"].dt.dayofyear
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
        df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
        return df

    @staticmethod
    def _add_kinematics_features(df):
        df = df.copy()

        p0q2, p0q3, p0q4 = df["p0q2"].values, df["p0q3"].values, df["p0q4"].values
        p0q_matrix = np.column_stack([p0q2, p0q3, p0q4])
        df["p0q_spatial_mean"] = p0q_matrix.mean(axis=1)
        df["p0q_spatial_std"] = p0q_matrix.std(axis=1)
        df["p0q_spatial_min"] = p0q_matrix.min(axis=1)
        df["p0q_spatial_max"] = p0q_matrix.max(axis=1)
        df["delta_p0q_2_3"] = p0q2 - p0q3
        df["delta_p0q_3_4"] = p0q3 - p0q4
        df["accel_p0q"] = df["delta_p0q_2_3"] - df["delta_p0q_3_4"]

        p2q0, p3q0, p4q0 = df["p2q0"].values, df["p3q0"].values, df["p4q0"].values
        pq0_matrix = np.column_stack([p2q0, p3q0, p4q0])
        df["pq0_temporal_mean"] = pq0_matrix.mean(axis=1)
        df["pq0_temporal_std"] = pq0_matrix.std(axis=1)
        df["pq0_temporal_min"] = pq0_matrix.min(axis=1)
        df["pq0_temporal_max"] = pq0_matrix.max(axis=1)
        df["delta_pq0_2_3"] = p2q0 - p3q0
        df["delta_pq0_3_4"] = p3q0 - p4q0
        df["accel_pq0"] = df["delta_pq0_2_3"] - df["delta_pq0_3_4"]

        df["spatial_vs_temporal_diff2"] = p0q2 - p2q0
        df["spatial_vs_temporal_prod2"] = p0q2 * p2q0
        df["total_delay_magnitude"] = np.abs(p0q2) + np.abs(p2q0)
        df["is_severe_train_delay"] = (p0q2 < -5).astype(np.float32)
        df["is_severe_station_congestion"] = (p2q0 < -5).astype(np.float32)
        df["is_zero_delay_train"] = ((p0q2 == 0) & (p0q3 == 0) & (p0q4 == 0)).astype(np.float32)
        df["is_zero_congestion_station"] = ((p2q0 == 0) & (p3q0 == 0) & (p4q0 == 0)).astype(np.float32)
        return df

    @staticmethod
    def _add_summary_stats(df):
        df = df.copy()
        df["gare_day_mean_p2q0"] = df.groupby(["gare", "date"])["p2q0"].transform("mean")
        df = df.sort_values(["train", "arret"])
        df["train_cum_mean_p0q2"] = (
            df.groupby("train")["p0q2"].expanding().mean().reset_index(level=0, drop=True)
        )
        return df

    def _engineer(self, df):
        df = self._add_mission_features(df)
        df = self._add_calendar_features(df)
        df = self._add_kinematics_features(df)
        df = self._add_summary_stats(df)
        return df

    def transform_all(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        df_train = self._engineer(df_train)
        df_test = self._engineer(df_test)

        # identical categorical levels on train and test
        for col in self.cat_cols:
            levels = sorted(
                set(df_train[col].astype(str).unique()) | set(df_test[col].astype(str).unique())
            )
            df_train[col] = pd.Categorical(df_train[col].astype(str), categories=levels)
            df_test[col] = pd.Categorical(df_test[col].astype(str), categories=levels)

        return df_train, df_test


# ==========================================================================
# 3. Chronological (out-of-time) CV splitter
# ==========================================================================
class TimeBasedCVSplitter:
    """Expanding-window CV: fold i trains on everything before a cutoff
    date and validates on the following chronological slice. No shuffling,
    no leakage from the future into training."""

    def __init__(self, n_splits: int = 5, min_train_ratio: float = 0.5):
        self.n_splits = n_splits
        self.min_train_ratio = min_train_ratio

    def split(self, df: pd.DataFrame, date_col: str = "date"):
        n = len(df)
        order = np.argsort(df[date_col].values, kind="mergesort")  # positions, chronological
        start = int(n * self.min_train_ratio)
        fold_bounds = np.linspace(start, n, self.n_splits + 1, dtype=int)

        for i in range(self.n_splits):
            train_end = fold_bounds[i]
            val_end = fold_bounds[i + 1]
            train_pos = order[:train_end]
            val_pos = order[train_end:val_end]
            if len(val_pos) == 0:
                continue
            yield train_pos, val_pos


# ==========================================================================
# 4. Model wrappers
# ==========================================================================
class LGBMForecaster:
    """LightGBM wrapper for MAE (L1) loss."""

    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            "objective": "regression_l1",
            "metric": "mae",
            "n_estimators": 2000,
            "learning_rate": 0.04,
            "num_leaves": 95,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 0.5,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = LGBMRegressor(**self.params)

    def fit(self, X_train, y_train, X_val, y_val, categorical_features=None, early_stopping_rounds=100):
        callbacks = [log_evaluation(period=0)]
        if early_stopping_rounds is not None:
            callbacks.append(early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=categorical_features if categorical_features else "auto",
            callbacks=callbacks,
        )
        return self

    def predict(self, X):
        return self.model.predict(X)


class XGBForecaster:
    """XGBoost wrapper. Relies on pandas 'category' dtype columns for
    native categorical support (tree_method='hist', enable_categorical=True)."""

    def __init__(self, params: Dict[str, Any] = None):
        default_params = dict(
            objective="reg:absoluteerror",
            n_estimators=1500,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.5,
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
            n_jobs=-1,
        )
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, X_train, y_train, X_val, y_val, categorical_features=None, early_stopping_rounds=100):
        self.model.set_params(early_stopping_rounds=early_stopping_rounds)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return self

    def predict(self, X):
        return self.model.predict(X)


class CatBoostForecaster:
    """CatBoost wrapper. Categorical columns are cast to string, as
    CatBoost's native handling expects."""

    def __init__(self, params: Dict[str, Any] = None):
        default_params = dict(
            loss_function="MAE",
            iterations=2000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3.0,
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
        )
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = CatBoostRegressor(**self.params)
        self.cat_features: List[str] = []

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.cat_features:
            X[c] = X[c].astype(str)
        return X

    def fit(self, X_train, y_train, X_val, y_val, categorical_features=None, early_stopping_rounds=100):
        self.cat_features = list(categorical_features or [])
        X_train = self._prep(X_train)
        X_val = self._prep(X_val)

        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
        val_pool = Pool(X_val, y_val, cat_features=self.cat_features)

        self.model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True,
            verbose=False,
        )
        return self

    def predict(self, X):
        return self.model.predict(self._prep(X))


# ==========================================================================
# 5. Ensemble weighting
# ==========================================================================
class EnsembleOptimizer:
    """Finds non-negative weights (summing to 1) across model predictions
    that minimize MAE on out-of-fold predictions."""

    def __init__(self):
        self.weights = None

    def fit(self, oof_preds_list: List[np.ndarray], y_true: np.ndarray):
        preds = np.column_stack(oof_preds_list)
        n_models = preds.shape[1]

        def loss(w):
            w = np.clip(w, 0, None)
            w = w / w.sum() if w.sum() > 0 else np.ones(n_models) / n_models
            return mean_absolute_error(y_true, preds @ w)

        x0 = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0)] * n_models
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        result = minimize(loss, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        w = np.clip(result.x, 0, None)
        self.weights = w / w.sum() if w.sum() > 0 else x0
        return self

    def predict(self, preds_list: List[np.ndarray]) -> np.ndarray:
        preds = np.column_stack(preds_list)
        return preds @ self.weights


# ==========================================================================
# 6. Threshold (nearest-class) optimizer
# ==========================================================================
class ThresholdOptimizer:
    """Snaps continuous blended predictions to the nearest discrete target
    value observed in training. If there are few distinct target values
    (<=20), it also tunes the cut-points between them (instead of using
    plain midpoints) to minimize MAE. With many/continuous targets it
    falls back to plain nearest-value snapping, since tuning one cut-point
    per pair of values does not scale."""

    MAX_CLASSES_FOR_TUNING = 20

    def __init__(self):
        self.classes = None
        self.thresholds = None

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.classes = np.sort(np.unique(y_true))
        if len(self.classes) < 2:
            self.thresholds = np.array([])
            return self

        midpoints = (self.classes[:-1] + self.classes[1:]) / 2.0

        if len(self.classes) > self.MAX_CLASSES_FOR_TUNING:
            self.thresholds = midpoints
            return self

        def loss(offsets):
            thresholds = np.sort(midpoints + offsets)
            mapped = self.map_predictions(y_pred, thresholds)
            return mean_absolute_error(y_true, mapped)

        x0 = np.zeros(len(midpoints))
        result = minimize(
            loss, x0, method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 3000},
        )
        self.thresholds = np.sort(midpoints + result.x)
        return self

    def map_predictions(self, y_pred: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        bucket = np.searchsorted(thresholds, y_pred)
        return self.classes[bucket]

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        if self.thresholds is None or len(self.thresholds) == 0:
            return y_pred
        return self.map_predictions(y_pred, self.thresholds)


# ==========================================================================
# 7. Metrics / IO helpers
# ==========================================================================
def evaluate_mae(y_true, y_pred) -> float:
    return mean_absolute_error(y_true, y_pred)


def save_submission(preds: np.ndarray, output_path: str, index: pd.Index = None):
    submission = pd.DataFrame({"p0q0": preds})
    if index is not None:
        submission.index = index
    submission.index.name = "index"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    submission.to_csv(output_path)


def validate_submission(output_path: str) -> bool:
    df = pd.read_csv(output_path)
    checks = {
        "file exists": os.path.exists(output_path),
        "no missing values": not df.isnull().values.any(),
        "has expected column": "p0q0" in df.columns,
        "non-empty": len(df) > 0,
    }
    print("[Validate] Submission checks:")
    for name, ok in checks.items():
        print(f"  - {name}: {'OK' if ok else 'FAILED'}")
    return all(checks.values())


def submit_file(path: str):
    # Platform-specific upload is intentionally not implemented here
    # (no API/credentials available). Submit manually instead.
    print(f"[Submit] Automated upload not implemented — submit manually: {path}", flush=True)


# ==========================================================================
# 8. Pipeline
# ==========================================================================
def main():
    start_time = time.time()
    print("=" * 70, flush=True)
    print("SNCF Transilien Waiting Time Forecast (ENS Challenge 166)", flush=True)
    print("Self-contained ensemble + chronological CV pipeline", flush=True)
    print("=" * 70, flush=True)

    # 1. Load data
    data_loader = DataLoader(data_dir=DATA_DIR)
    df_train, df_test, y_sample = data_loader.load_data()

    # 2. Feature engineering
    fe = FeatureEngineer()
    df_train, df_test = fe.transform_all(df_train, df_test)

    ignore_cols = {"train", "date", "p0q0"}
    feature_cols = [c for c in df_train.columns if c not in ignore_cols]
    cat_cols = fe.cat_cols

    print(f"\n[Pipeline] Training on {len(feature_cols)} engineered features:", flush=True)
    print(feature_cols, flush=True)

    # 3. Chronological CV
    cv = TimeBasedCVSplitter(n_splits=N_SPLITS, min_train_ratio=0.5)

    oof_preds_lgb = np.zeros(len(df_train))
    oof_preds_xgb = np.zeros(len(df_train))
    oof_preds_cat = np.zeros(len(df_train))
    evaluated_mask = np.zeros(len(df_train), dtype=bool)

    test_preds_lgb = np.zeros(len(df_test))
    test_preds_xgb = np.zeros(len(df_test))
    test_preds_cat = np.zeros(len(df_test))

    y_train_full = df_train["p0q0"].values

    print("\n" + "=" * 50, flush=True)
    print(f"Starting {N_SPLITS}-Fold Chronological Out-of-Time Cross Validation", flush=True)
    print("=" * 50, flush=True)

    actual_folds = 0
    for fold, (train_idx, val_idx) in enumerate(cv.split(df_train, date_col="date")):
        print(f"\n--- FOLD {fold + 1}/{N_SPLITS} ---", flush=True)
        actual_folds += 1

        X_tr = df_train.iloc[train_idx][feature_cols].copy()
        y_tr = df_train.iloc[train_idx]["p0q0"].values
        X_va = df_train.iloc[val_idx][feature_cols].copy()
        y_va = df_train.iloc[val_idx]["p0q0"].values
        X_te = df_test[feature_cols].copy()

        lgb_model = LGBMForecaster()
        lgb_model.fit(X_tr.copy(), y_tr, X_va.copy(), y_va, categorical_features=cat_cols)
        val_p_lgb = lgb_model.predict(X_va.copy())
        oof_preds_lgb[val_idx] = val_p_lgb
        test_preds_lgb += lgb_model.predict(X_te.copy()) / N_SPLITS
        mae_lgb = evaluate_mae(y_va, val_p_lgb)

        xgb_model = XGBForecaster()
        xgb_model.fit(X_tr.copy(), y_tr, X_va.copy(), y_va, categorical_features=cat_cols)
        val_p_xgb = xgb_model.predict(X_va.copy())
        oof_preds_xgb[val_idx] = val_p_xgb
        test_preds_xgb += xgb_model.predict(X_te.copy()) / N_SPLITS
        mae_xgb = evaluate_mae(y_va, val_p_xgb)

        cat_model = CatBoostForecaster()
        cat_model.fit(X_tr.copy(), y_tr, X_va.copy(), y_va, categorical_features=cat_cols)
        val_p_cat = cat_model.predict(X_va.copy())
        oof_preds_cat[val_idx] = val_p_cat
        test_preds_cat += cat_model.predict(X_te.copy()) / N_SPLITS
        mae_cat = evaluate_mae(y_va, val_p_cat)

        evaluated_mask[val_idx] = True
        print(f"Fold {fold + 1} MAE -> LightGBM: {mae_lgb:.5f} | XGBoost: {mae_xgb:.5f} | CatBoost: {mae_cat:.5f}", flush=True)

    if actual_folds == 0:
        raise RuntimeError("No validation fold produced any rows — check min_train_ratio / N_SPLITS.")
    if actual_folds != N_SPLITS:
        print(f"\n[Pipeline] Note: only {actual_folds}/{N_SPLITS} folds produced a validation slice.", flush=True)
        scale = N_SPLITS / actual_folds
        test_preds_lgb *= scale
        test_preds_xgb *= scale
        test_preds_cat *= scale

    y_eval = y_train_full[evaluated_mask]
    oof_mae_lgb = evaluate_mae(y_eval, oof_preds_lgb[evaluated_mask])
    oof_mae_xgb = evaluate_mae(y_eval, oof_preds_xgb[evaluated_mask])
    oof_mae_cat = evaluate_mae(y_eval, oof_preds_cat[evaluated_mask])

    print("\n" + "=" * 50, flush=True)
    print("OVERALL OUT-OF-FOLD (OOF) RESULTS", flush=True)
    print("=" * 50, flush=True)
    print(f"LightGBM OOF MAE : {oof_mae_lgb:.5f}", flush=True)
    print(f"XGBoost  OOF MAE : {oof_mae_xgb:.5f}", flush=True)
    print(f"CatBoost OOF MAE : {oof_mae_cat:.5f}", flush=True)

    # 4. Ensemble weighting
    ensemble = EnsembleOptimizer()
    ensemble.fit(
        [oof_preds_lgb[evaluated_mask], oof_preds_xgb[evaluated_mask], oof_preds_cat[evaluated_mask]],
        y_eval,
    )
    print(f"[Ensemble] weights (lgb, xgb, cat): {np.round(ensemble.weights, 4)}", flush=True)

    oof_blend = ensemble.predict(
        [oof_preds_lgb[evaluated_mask], oof_preds_xgb[evaluated_mask], oof_preds_cat[evaluated_mask]]
    )
    final_test_preds_raw = ensemble.predict([test_preds_lgb, test_preds_xgb, test_preds_cat])

    # 5. Threshold optimization
    thresh_opt = ThresholdOptimizer()
    thresh_opt.fit(oof_blend, y_eval)
    final_test_preds_optimized = thresh_opt.transform(final_test_preds_raw)

    # 6. Save and validate submission
    save_submission(final_test_preds_optimized, output_path=OUTPUT_PATH, index=df_test.index)
    is_valid = validate_submission(output_path=OUTPUT_PATH)

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70, flush=True)
    print(f"Pipeline finished in {elapsed_time / 60:.2f} minutes.", flush=True)
    print(f"  - Ensemble (raw)        : {evaluate_mae(y_eval, oof_blend):.5f} MAE", flush=True)
    print(f"  - Ensemble (threshold)  : {evaluate_mae(y_eval, thresh_opt.transform(oof_blend)):.5f} MAE", flush=True)
    print("=" * 70, flush=True)

    if is_valid:
        submit_file(OUTPUT_PATH)
    else:
        print("[Submit] Skipped: submission failed validation checks.", flush=True)


if __name__ == "__main__":
    main()