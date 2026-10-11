import numpy as np
import pandas as pd
import optuna
import category_encoders as ce
from sklearn.model_selection import TimeSeriesSplit
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
cat_cols = ['gare', 'arret', 'quarter', 'gare_arret']

# ------------------------------------------------------------------
# 4. Split temporale train / holdout (80% / 20% sulle date)
#    L'holdout NON viene usato da Optuna: serve solo per la stima finale.
# ------------------------------------------------------------------
df = df.sort_values('date').reset_index(drop=True)
cutoff = df['date'].quantile(0.80)

train_df = df[df['date'] < cutoff].reset_index(drop=True)
val_df = df[df['date'] >= cutoff].reset_index(drop=True)

X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]
X_full, y_full = df[features], df[target]
X_test_prep = X_test[features]

# ------------------------------------------------------------------
# 5. Ottimizzazione iperparametri con Optuna
#    - CV temporale expanding-window (TimeSeriesSplit) solo sul train
#    - Target encoding rifittato dentro ogni fold (niente leakage)
#    - Early stopping: n_estimators viene deciso dai dati
# ------------------------------------------------------------------
N_TRIALS = 100
N_SPLITS = 3
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

def make_model(params, n_estimators, early_stopping_rounds=None):
    return XGBRegressor(
        objective='reg:absoluteerror',
        tree_method='hist',
        random_state=42,
        n_jobs=8,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        **{k: v for k, v in params.items() if k != 'smoothing'},
    )

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 14),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'smoothing': trial.suggest_float('smoothing', 1.0, 50.0),
    }

    maes, best_iters = [], []
    for step, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
        Xtr, ytr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        Xva, yva = X_train.iloc[va_idx], y_train.iloc[va_idx]

        enc = ce.TargetEncoder(cols=cat_cols, smoothing=params['smoothing'])
        Xtr_enc = enc.fit_transform(Xtr, ytr)
        Xva_enc = enc.transform(Xva)

        model = make_model(params, n_estimators=5000, early_stopping_rounds=100)
        model.fit(Xtr_enc, ytr, eval_set=[(Xva_enc, yva)], verbose=False)

        pred = np.round(model.predict(Xva_enc))
        maes.append(mean_absolute_error(yva, pred))
        best_iters.append(model.best_iteration + 1)

        # pruning: scarta i trial chiaramente peggiori dopo ogni fold
        trial.report(float(np.mean(maes)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr('best_n_estimators', int(np.mean(best_iters)))
    return float(np.mean(maes))

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    study_name='xgb_transilien',
    storage='sqlite:///optuna_xgb.db',   # permette di riprendere lo studio
    load_if_exists=True,
)

# Trial 0: parametri attuali come punto di partenza (warm start)
study.enqueue_trial({
    'max_depth': 12,
    'learning_rate': 0.018935255486594967,
    'subsample': 0.9032513682124419,
    'colsample_bytree': 0.7301385721132697,
    'min_child_weight': 48,
    'gamma': 1.31348393511213,
    'reg_alpha': 0.32617971711310934,
    'reg_lambda': 1.6701248209971402,
    'smoothing': 10.0,
})

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f'\nMiglior MAE (CV temporale): {study.best_value:.4f}')
print('Migliori parametri:')
for k, v in study.best_params.items():
    print(f'  {k}: {v}')

best_params = dict(study.best_params)
best_n_estimators = study.best_trial.user_attrs['best_n_estimators']
print(f'  n_estimators (early stopping): {best_n_estimators}')

# ------------------------------------------------------------------
# 6. Validazione sull'holdout con i parametri migliori
# ------------------------------------------------------------------
enc_val = ce.TargetEncoder(cols=cat_cols, smoothing=best_params['smoothing'])
Xtr_enc = enc_val.fit_transform(X_train, y_train)
Xva_enc = enc_val.transform(X_val)

model_val = make_model(best_params, n_estimators=best_n_estimators)
model_val.fit(Xtr_enc, y_train)

mae_val = mean_absolute_error(y_val, np.round(model_val.predict(Xva_enc)))
mae_train = mean_absolute_error(y_train, np.round(model_val.predict(Xtr_enc)))
print(f'MAE holdout: {mae_val:.4f} | MAE train: {mae_train:.4f}')

# ------------------------------------------------------------------
# 7. Riaddestramento su tutti i dati e submission
#    (+10% di alberi perché si usano più dati)
# ------------------------------------------------------------------
enc_final = ce.TargetEncoder(cols=cat_cols, smoothing=best_params['smoothing'])
Xfull_enc = enc_final.fit_transform(X_full, y_full)
Xtest_enc = enc_final.transform(X_test_prep)

model_final = make_model(best_params, n_estimators=int(best_n_estimators * 1.1))
model_final.fit(Xfull_enc, y_full)

y_pred_test = model_final.predict(Xtest_enc)
pd.DataFrame(y_pred_test).to_csv('final_submission_xboost.csv')