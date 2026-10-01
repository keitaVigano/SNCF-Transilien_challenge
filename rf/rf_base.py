import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# result 0.8849.88 vs MAE validation: 0.8407
x_train = pd.read_csv('data/x_train_final.csv', index_col=0)
x_train.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
y_train = pd.read_csv('data/y_train_final.csv', index_col=0)

df = x_train.copy()
df["date"] = pd.to_datetime(df["date"])
df["target"] = y_train.iloc[:, 0].values


df["dow"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
#df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

# encoding stazione: frequency encoding (semplice, no leakage)
#freq_gare = df["gare"].value_counts(normalize=True)
#df["gare_freq"] = df["gare"].map(freq_gare)


date_split = df["date"].quantile(0.8)
train_mask = df["date"] <= date_split

feature_cols = [
    "arret", "p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4",
    "dow", "month"
]

X_tr, X_val = df.loc[train_mask, feature_cols], df.loc[~train_mask, feature_cols]
y_tr, y_val = df.loc[train_mask, "target"], df.loc[~train_mask, "target"]


# Modello 
rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
    verbose=2
)
rf.fit(X_tr, y_tr)

pred_val = rf.predict(X_val)
mae = mean_absolute_error(y_val, pred_val)
print(f"MAE validation: {mae:.4f}")
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances)


#submission
x_test = pd.read_csv('data/x_test_final.csv', index_col=0)
x_test.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
df_test = x_test.copy()
df_test["date"] = pd.to_datetime(df_test["date"])
df_test["dow"] = df_test["date"].dt.dayofweek
df_test["month"] = df_test["date"].dt.month
#df_test["is_weekend"] = df_test["dow"].isin([5, 6]).astype(int)
#df_test["gare_freq"] = df_test["gare"].map(freq_gare)
#df_test["gare_freq"] = df_test["gare_freq"].fillna(0)  # stazioni mai viste in train
X_test = df_test[feature_cols].fillna(-999)
pred_test = rf.predict(X_test)
submission = pd.DataFrame(
    {"p0q0": pred_test},
    index=x_test.index
)
submission.index.name = x_test.index.name if x_test.index.name else "index"

submission.to_csv("data/submission.csv")
print(submission.head())
print(f"Submission salvata: {submission.shape[0]} righe")