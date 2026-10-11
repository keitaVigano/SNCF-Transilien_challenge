import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# leaderboard 81 vs 79
x_train = pd.read_csv('data/x_train_final.csv', index_col=0)
x_train.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
y_train = pd.read_csv('data/y_train_final.csv', index_col=0)

df = x_train.copy()
df["date"] = pd.to_datetime(df["date"])
df["target"] = y_train.iloc[:, 0].values

df["dow"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# encoding stazione: frequency encoding (semplice, no leakage)
freq_gare = df["gare"].value_counts(normalize=True)
df["gare_freq"] = df["gare"].map(freq_gare)


def add_summary_stats(df):
    df = df.copy()

    # statistica riassuntiva per stazione, per giorno
    df["gare_day_mean_p2q0"] = df.groupby(["gare", "date"])["p2q0"].transform("mean")

    # statistica cumulativa per treno, dentro la corsa (solo fermate già attraversate)
    df = df.sort_values(["train", "arret"])
    df["train_cum_mean_p0q2"] = (
        df.groupby("train")["p0q2"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


df = add_summary_stats(df)

# data split
date_split = df["date"].quantile(0.8)
train_mask = df["date"] <= date_split

feature_cols = [
    "arret", "p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4",
    "dow", "month", "gare_freq",
    "gare_day_mean_p2q0", "train_cum_mean_p0q2",
]

X_tr, X_val = df.loc[train_mask, feature_cols].fillna(-999), df.loc[~train_mask, feature_cols].fillna(-999)
y_tr, y_val = df.loc[train_mask, "target"], df.loc[~train_mask, "target"]

# Modello baseline (per confronto)
rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
    verbose=2,
)
rf.fit(X_tr, y_tr)

pred_val = rf.predict(X_val)
mae = mean_absolute_error(y_val, pred_val)
print(f"MAE validation (baseline): {mae:.4f}")

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances)


# Ottimizzazione con Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "n_jobs": -1,
        "random_state": 42,
    }
    model = RandomForestRegressor(**params)
    model.fit(X_tr, y_tr)
    pred_val = model.predict(X_val)
    return mean_absolute_error(y_val, pred_val)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20, show_progress_bar=True)

print("Migliori parametri:", study.best_params)
print(f"Miglior MAE: {study.best_value:.4f}")

best_rf = RandomForestRegressor(**study.best_params, n_jobs=-1, random_state=42)
best_rf.fit(X_tr, y_tr)
pred_val_best = best_rf.predict(X_val)
mae_best = mean_absolute_error(y_val, pred_val_best)
print(f"MAE validation (modello ottimizzato): {mae_best:.4f}")
# Riallenamento finale su tutto il train
X_full = df[feature_cols].fillna(-999)
y_full = df["target"]

best_rf_final = RandomForestRegressor(**study.best_params, n_jobs=-1, random_state=42)
best_rf_final.fit(X_full, y_full)

# --- Submission ---
x_test = pd.read_csv('data/x_test_final.csv', index_col=0)
x_test.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
df_test = x_test.copy()
df_test["date"] = pd.to_datetime(df_test["date"])
df_test["dow"] = df_test["date"].dt.dayofweek
df_test["month"] = df_test["date"].dt.month
df_test["gare_freq"] = df_test["gare"].map(freq_gare).fillna(0)

df_test = add_summary_stats(df_test)

X_test = df_test[feature_cols].fillna(-999)
pred_test = best_rf_final.predict(X_test)

submission = pd.DataFrame({"p0q0": pred_test}, index=df_test.index)
submission.index.name = x_test.index.name if x_test.index.name else "index"
submission.to_csv("data/submission_opt_fe.csv")
print(submission.head())
print(f"Submission salvata: {submission.shape[0]} righe")