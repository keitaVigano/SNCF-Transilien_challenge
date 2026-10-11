import pandas as pd
import numpy as np
from typing import Dict, Any
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error
import optuna
from optuna.samplers import TPESampler


class LGBMForecaster:
    """Optimized LightGBM model wrapper for MAE L1 loss."""
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'n_estimators': 300,
            'learning_rate': 0.04,
            'num_leaves': 95,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.85,
            'subsample_freq': 1,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
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


# --- Caricamento dati ---
x_train = pd.read_csv('data/x_train_final.csv', index_col=0)
x_train.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
y_train = pd.read_csv('data/y_train_final.csv', index_col=0)

df = x_train.copy()
df["date"] = pd.to_datetime(df["date"])
df["target"] = y_train.iloc[:, 0].values


def add_mission_features(df):
    """Estrae i prefissi del codice treno (linea/tipo di servizio)."""
    df = df.copy()
    train_str = df["train"].astype(str)
    df["mission_prefix1"] = train_str.str[:1]
    df["mission_prefix2"] = train_str.str[:2]
    df["mission_prefix3"] = train_str.str[:3]
    return df


def add_calendar_features(df):
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


def add_kinematics_features(df):
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


def add_summary_stats(df):
    df = df.copy()
    df["gare_day_mean_p2q0"] = df.groupby(["gare", "date"])["p2q0"].transform("mean")

    df = df.sort_values(["train", "arret"])
    df["train_cum_mean_p0q2"] = (
        df.groupby("train")["p0q2"].expanding().mean().reset_index(level=0, drop=True)
    )
    return df


# --- Pipeline feature engineering ---
df = add_mission_features(df)
df = add_calendar_features(df)
df = add_kinematics_features(df)
df = add_summary_stats(df)

cat_cols = ["gare", "mission_prefix1", "mission_prefix2", "mission_prefix3"]

feature_cols = [
    "arret", "p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4",
    "dow", "month", "day", "dayofyear", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "gare_day_mean_p2q0", "train_cum_mean_p0q2",
    "p0q_spatial_mean", "p0q_spatial_std", "p0q_spatial_min", "p0q_spatial_max",
    "delta_p0q_2_3", "delta_p0q_3_4", "accel_p0q",
    "pq0_temporal_mean", "pq0_temporal_std", "pq0_temporal_min", "pq0_temporal_max",
    "delta_pq0_2_3", "delta_pq0_3_4", "accel_pq0",
    "spatial_vs_temporal_diff2", "spatial_vs_temporal_prod2", "total_delay_magnitude",
    "is_severe_train_delay", "is_severe_station_congestion",
    "is_zero_delay_train", "is_zero_congestion_station",
] + cat_cols

# --- Split temporale (senza leakage) ---
date_split = df["date"].quantile(0.8)
train_mask = df["date"] <= date_split

X_tr = df.loc[train_mask, feature_cols].copy()
X_val = df.loc[~train_mask, feature_cols].copy()
y_tr = df.loc[train_mask, "target"]
y_val = df.loc[~train_mask, "target"]

# --- Conversione a dtype 'category', con lo STESSO set di categorie tra train e val ---
# Fondamentale: se una categoria (es. una stazione) compare solo in val e non in train,
# LightGBM la tratta comunque correttamente, ma è bene avere l'unione dei livelli
# per coerenza tra i due DataFrame.
for col in cat_cols:
    all_categories = sorted(set(X_tr[col].astype(str).unique()) | set(X_val[col].astype(str).unique()))
    X_tr[col] = pd.Categorical(X_tr[col].astype(str), categories=all_categories)
    X_val[col] = pd.Categorical(X_val[col].astype(str), categories=all_categories)

# --- Modello ---
'''forecaster = LGBMForecaster()
forecaster.fit(X_tr, y_tr, X_val, y_val, categorical_features=cat_cols, early_stopping_rounds=100)

pred_val = forecaster.predict(X_val)
mae = mean_absolute_error(y_val, pred_val)
print(f"MAE validation (categoriche native): {mae:.4f}")
print(f"best_iteration: {forecaster.model.best_iteration_}")

importances = pd.Series(forecaster.model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances)'''

def objective(trial: optuna.Trial) -> float:
    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 1000,  # alto, tanto usiamo early stopping
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    forecaster = LGBMForecaster(params)
    forecaster.fit(
        X_tr, y_tr, X_val, y_val,
        categorical_features=cat_cols,
        early_stopping_rounds=100,
    )
    pred_val = forecaster.predict(X_val)
    return mean_absolute_error(y_val, pred_val)


study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
)
study.optimize(objective, n_trials=30, timeout=3600, show_progress_bar=True)

print("Best MAE:", study.best_value)
print("Best params:", study.best_params)

# --- Riaddestra il modello finale con i migliori parametri ---
best_params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': 1000,
    **study.best_params,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}
forecaster = LGBMForecaster(best_params)
forecaster.fit(X_tr, y_tr, X_val, y_val, categorical_features=cat_cols, early_stopping_rounds=100)

pred_val = forecaster.predict(X_val)
mae = mean_absolute_error(y_val, pred_val)
print(f"MAE validation (best params, Optuna): {mae:.4f}")
print(f"best_iteration: {forecaster.model.best_iteration_}")

importances = pd.Series(forecaster.model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances)

# --- Submission ---
x_test = pd.read_csv('data/x_test_final.csv', index_col=0)
x_test.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
df_test = x_test.copy()
df_test["date"] = pd.to_datetime(df_test["date"])

df_test = add_mission_features(df_test)
df_test = add_calendar_features(df_test)
df_test = add_kinematics_features(df_test)
df_test = add_summary_stats(df_test)

X_test = df_test[feature_cols].copy()

# stesse categorie usate in training (unione train+val), applicate anche al test
for col in cat_cols:
    train_categories = X_tr[col].cat.categories
    X_test[col] = pd.Categorical(X_test[col].astype(str), categories=train_categories)

pred_test = forecaster.predict(X_test)

submission = pd.DataFrame({"p0q0": pred_test}, index=df_test.index)
submission.index.name = x_test.index.name if x_test.index.name else "index"
submission.to_csv("data/submission_lgbm_optuna.csv")
print(submission.head())
print(f"Submission salvata: {submission.shape[0]} righe")