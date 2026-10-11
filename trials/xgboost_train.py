import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ------------------------------------------------------------------
# 1. Caricamento dati
# ------------------------------------------------------------------
X_train_raw = pd.read_csv('data/x_train_final.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
X_test = pd.read_csv('data/x_test_final.csv').drop(columns=['Unnamed: 0'])
Y = pd.read_csv('data/y_train_final.csv').drop(columns=['Unnamed: 0'])

assert len(X_train_raw) == len(Y), "X_train e y_train non hanno la stessa dimensione"

df = pd.concat([X_train_raw, Y], axis=1)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
X_test['date'] = pd.to_datetime(X_test['date'], format='%Y-%m-%d')

# ------------------------------------------------------------------
# 2. Pulizia: rimozione outlier estremi (p0q3, p0q4 < -1400)
# ------------------------------------------------------------------
df = df[(df['p0q3'] > -1400) & (df['p0q4'] > -1400)]

# ------------------------------------------------------------------
# 3. Feature engineering
# ------------------------------------------------------------------
def add_features(data, gare_freq):
    data = data.copy()
    data['holidays'] = data['date'].between('2023-06-01', '2023-08-31').astype(int)
    data['quarter'] = data['date'].dt.quarter
    data['gare_freq'] = data['gare'].map(gare_freq)
    data['propagation_index'] = (
        data['p2q0'] * 0.5 + data['p3q0'] * 0.3 + data['p4q0'] * 0.2
    )
    data['gare_arret'] = data['gare'].astype(str) + '_' + data['arret'].astype(str)
    return data

gare_freq = df['gare'].value_counts()   # calcolata solo sul train
df = add_features(df, gare_freq)
X_test = add_features(X_test, gare_freq)

features = [
    'arret', 'gare_freq', 'quarter', 'gare_arret', 'holidays',
    'p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4',
    'gare', 'propagation_index'
]
target = 'p0q0'

# ------------------------------------------------------------------
# 4. Split temporale train / validation (80% / 20% sulle date)
# ------------------------------------------------------------------
df = df.sort_values('date').reset_index(drop=True)
cutoff = df['date'].quantile(0.80)

train_df = df[df['date'] < cutoff]
val_df = df[df['date'] >= cutoff]

X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]
X_full, y_full = df[features], df[target]
X_test_prep = X_test[features]

# ------------------------------------------------------------------
# 5. Modello finale: XGBoost + Target Encoding (parametri Optuna)
# ------------------------------------------------------------------
best_params_xgb = {
    'n_estimators': 4200,
    'max_depth': 12,
    'learning_rate': 0.018935255486594967,
    'subsample': 0.9032513682124419,
    'colsample_bytree': 0.7301385721132697,
    'min_child_weight': 48,
    'gamma': 1.31348393511213,
    'reg_alpha': 0.32617971711310934,
    'reg_lambda': 1.6701248209971402,
}

def build_pipeline():
    return Pipeline([
        ('target_enc', ce.TargetEncoder(
            cols=['gare', 'arret', 'quarter', 'gare_arret'],
            smoothing=10
        )),
        ('model', XGBRegressor(
            objective='reg:absoluteerror',
            tree_method='hist',
            random_state=42,
            n_jobs=8,
            **best_params_xgb
        )),
    ])

# ------------------------------------------------------------------
# 6. Validazione
# ------------------------------------------------------------------
pipe_val = build_pipeline()
pipe_val.fit(X_train, y_train)

mae_val = mean_absolute_error(y_val, np.round(pipe_val.predict(X_val)))
mae_train = mean_absolute_error(y_train, np.round(pipe_val.predict(X_train)))
print(f'MAE validation: {mae_val:.4f} | MAE train: {mae_train:.4f}')

# ------------------------------------------------------------------
# 7. Riaddestramento su tutti i dati e submission
# ------------------------------------------------------------------
pipe_final = build_pipeline()
pipe_final.fit(X_full, y_full)

y_pred_test = pipe_final.predict(X_test_prep)
pd.DataFrame(y_pred_test).to_csv('final_submission_xboost.csv')