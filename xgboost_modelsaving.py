import os
import joblib
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# 0. Configurazione cache modelli
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)
VAL_PATH = os.path.join(MODEL_DIR, 'pipe_val.joblib')
FINAL_PATH = os.path.join(MODEL_DIR, 'pipe_final.joblib')
META_PATH = os.path.join(MODEL_DIR, 'metadata.joblib')

RETRAIN = True  # True la prima volta (il clip cambia i dati), poi mettilo a False

CLIP_LOWER = -50
CLIP_COLS = ['p0q0', 'p0q2', 'p0q3', 'p0q4', 'p2q0', 'p3q0', 'p4q0']

# 1. Caricamento dati
X_train_raw = pd.read_csv('data/x_train_final.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
X_test = pd.read_csv('data/x_test_final.csv').drop(columns=['Unnamed: 0'])
Y = pd.read_csv('data/y_train_final.csv').drop(columns=['Unnamed: 0'])

assert len(X_train_raw) == len(Y), "X_train e y_train non hanno la stessa dimensione"

df = pd.concat([X_train_raw, Y], axis=1)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
X_test['date'] = pd.to_datetime(X_test['date'], format='%Y-%m-%d')

# 2. Pulizia: rimozione errori di misura (-1440) e clip a -70
df = df[(df['p0q3'] > -1400) & (df['p0q4'] > -1400)]

df[CLIP_COLS] = df[CLIP_COLS].clip(lower=CLIP_LOWER)
test_cols = [c for c in CLIP_COLS if c in X_test.columns]
X_test[test_cols] = X_test[test_cols].clip(lower=CLIP_LOWER)

def add_features(data, gare_freq):
    data = data.copy()
    data['holidays'] = data['date'].between('2023-06-01', '2023-08-31').astype(int)
    data['quarter'] = data['date'].dt.quarter
    data['gare_freq'] = data['gare'].map(gare_freq)
    data['propagation_index'] = (
        data['p2q0'] * 0.5 + data['p3q0'] * 0.3 + data['p4q0'] * 0.2
    )
    data['propagation_index_2'] = (
        data['p0q2'] * 0.5 + data['p0q3'] * 0.3 + data['p0q4'] * 0.2
    )
    data['gare_arret'] = data['gare'].astype(str) + '_' + data['arret'].astype(str)
    return data

gare_freq = df['gare'].value_counts()   
df = add_features(df, gare_freq)
X_test = add_features(X_test, gare_freq)
features = [
    'arret', 'gare_freq', 'quarter', 'gare_arret', 'holidays',
    'p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4',
    'gare', 'propagation_index', 'propagation_index_2'
]
target = 'p0q0'

# 4. Split temporale train / validation (80% / 20% sulle date)
df = df.sort_values('date').reset_index(drop=True)
cutoff = df['date'].quantile(0.80)

train_df = df[df['date'] < cutoff]
val_df = df[df['date'] >= cutoff]

X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]
X_full, y_full = df[features], df[target]
X_test_prep = X_test[features]

# 5. Modello: XGBoost + Target Encoding (parametri Optuna)
best_params_xgb = {
    'n_estimators': 4100,
    'max_depth': 11,
    'learning_rate': 0.018935258486594967,
    'subsample': 0.9032513482124419,
    'colsample_bytree': 0.7301335721132697,
    'min_child_weight': 49,
    'gamma': 1.31348393511219,
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

# 6. Validazione (con cache su disco)
if os.path.exists(VAL_PATH) and not RETRAIN:
    pipe_val = joblib.load(VAL_PATH)
    print('Modello di validazione caricato da disco')
else:
    pipe_val = build_pipeline()
    pipe_val.fit(X_train, y_train)
    joblib.dump(pipe_val, VAL_PATH)
    print(f'Modello di validazione salvato in {VAL_PATH}')

mae_val = mean_absolute_error(y_val, np.round(pipe_val.predict(X_val)))
mae_train = mean_absolute_error(y_train, np.round(pipe_val.predict(X_train)))
print(f'MAE validation: {mae_val:.4f} | MAE train: {mae_train:.4f}')


# 7. Modello finale su tutti i dati (con cache) e submission
if os.path.exists(FINAL_PATH) and not RETRAIN:
    pipe_final = joblib.load(FINAL_PATH)
    print('Modello finale caricato da disco')
else:
    pipe_final = build_pipeline()
    pipe_final.fit(X_full, y_full)
    joblib.dump(pipe_final, FINAL_PATH)
    joblib.dump(
        {'gare_freq': gare_freq, 'features': features,
         'params': best_params_xgb, 'clip_lower': CLIP_LOWER},
        META_PATH
    )
    print(f'Modello finale salvato in {FINAL_PATH}')

y_pred_test = np.round(pipe_final.predict(X_test_prep))
pd.DataFrame(y_pred_test).to_csv('final_submission_xboost_round41.csv')