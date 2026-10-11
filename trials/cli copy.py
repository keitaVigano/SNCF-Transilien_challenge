import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from delays.cache import fit_or_load, cache_key
from delays.config import (
    CLIP_LOWER, FEATURES, MODEL_DIR, OUTPUT_DIR, TARGET, XGB_PARAMS,
)
from delays.data import prepare
from delays.experiments import log_result, show_best
from delays.registry import MODELS, get_model


def mae(pipe, X, y):
    return mean_absolute_error(y, np.round(pipe.predict(X)))


def run(models, final, retrain):
    df, train_df, val_df, X_test, gare_freq = prepare()
    Xtr, ytr = train_df[FEATURES], train_df[TARGET]
    Xva, yva = val_df[FEATURES], val_df[TARGET]

    # 1. Confronto su validation
    for name in models:
        t0 = time.time()
        pipe = fit_or_load(MODEL_DIR, 'val', name, get_model, Xtr, ytr, retrain)
        m_val, m_tr = mae(pipe, Xva, yva), mae(pipe, Xtr, ytr)
        log_result(name, m_val, m_tr, time.time() - t0)
        print(f'[{name}] MAE val {m_val:.4f} | train {m_tr:.4f}')

    show_best()

    # 2. Modello finale su tutti i dati + submission
    if final:
        pipe = fit_or_load(MODEL_DIR, 'final', final, get_model,
                           df[FEATURES], df[TARGET], retrain)
        joblib.dump(
            {'gare_freq': gare_freq, 'features': FEATURES,
             'params': XGB_PARAMS, 'clip_lower': CLIP_LOWER},
            MODEL_DIR / 'metadata.joblib',
        )
        OUTPUT_DIR.mkdir(exist_ok=True)
        pred = np.round(pipe.predict(X_test[FEATURES]))
        key = cache_key(final, pipe)
        out = OUTPUT_DIR / (
            f'submission_{final}_clip{abs(CLIP_LOWER)}_{key}.csv'
        )
        pd.DataFrame(pred).to_csv(out)
        print(f'Submission salvata in {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['xgb'], choices=list(MODELS))
    p.add_argument('--final', choices=list(MODELS), default=None)
    p.add_argument('--retrain', action='store_true')
    a = p.parse_args()
    run(a.models, a.final, a.retrain)


