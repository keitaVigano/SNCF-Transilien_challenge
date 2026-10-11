import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from delays.config import CLIP_LOWER, FEATURES

RUNS_DIR = Path('runs')
RESULTS_CSV = RUNS_DIR / 'validation_results.csv'


def mae(pipe, X, y):
    return float(mean_absolute_error(y, np.round(pipe.predict(X))))


def _new_run_dir(kind, name):
    """Create runs/<kind>/<date_time>_<name>/ and return it."""
    now = datetime.now()
    run_dir = RUNS_DIR / kind / f'{now:%Y-%m-%d_%H-%M-%S}_{name}'
    run_dir.mkdir(parents=True)
    return run_dir, now


def _append_result(row):
    """Append one row to the shared CSV (creates it with a header if missing)."""
    is_new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def validate(name, pipe, X_train, y_train, X_val, y_val):
    pipe.fit(X_train, y_train)

    mae_val = mae(pipe, X_val, y_val)
    mae_train = mae(pipe, X_train, y_train)

    run_dir, now = _new_run_dir('validation', name)
    joblib.dump(pipe, run_dir / 'model.joblib')
    info = {
        'name': name,
        'timestamp': f'{now:%Y-%m-%d %H:%M:%S}',
        'mae_val': mae_val,
        'mae_train': mae_train,
        'params': pipe.named_steps['model'].get_params(),
    }
    (run_dir / 'info.json').write_text(json.dumps(info, indent=2, default=str))

    _append_result({
        'timestamp': info['timestamp'],
        'model': name,
        'mae_val': round(mae_val, 4),
        'mae_train': round(mae_train, 4),
        'folder': run_dir.name,
    })

    print(f'[{name}] MAE val {mae_val:.4f} | train {mae_train:.4f} -> {run_dir}')
    return pipe, run_dir


def train_final(name, pipe, X_full, y_full, X_test, gare_freq):
    pipe.fit(X_full, y_full)

    run_dir, now = _new_run_dir('final', name)
    joblib.dump(pipe, run_dir / 'model.joblib')
    joblib.dump(
        {'gare_freq': gare_freq, 'features': FEATURES, 'clip_lower': CLIP_LOWER},
        run_dir / 'metadata.joblib',
    )
    info = {
        'name': name,
        'timestamp': f'{now:%Y-%m-%d %H:%M:%S}',
        'n_train': len(X_full),
        'params': pipe.named_steps['model'].get_params(),
    }
    (run_dir / 'info.json').write_text(json.dumps(info, indent=2, default=str))

    pred = np.round(pipe.predict(X_test))
    pd.DataFrame(pred).to_csv(run_dir / 'submission.csv')

    print(f'[{name}] final model and submission -> {run_dir}')
    return pipe, run_dir