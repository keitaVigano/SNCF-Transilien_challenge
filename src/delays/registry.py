import category_encoders as ce
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import json
from delays.config import CAT_COLS, ROOT, SMOOTHING, XGB_PARAMS

def _wrap(model):
    return Pipeline([
        ('target_enc', ce.TargetEncoder(cols=CAT_COLS, smoothing=SMOOTHING)),
        ('model', model),
    ])

MODELS = {
    'xgb': lambda: _wrap(XGBRegressor(
        objective='reg:absoluteerror', tree_method='hist',
        random_state=42, n_jobs=8, **XGB_PARAMS))
}


def get_model(name, smoothing=SMOOTHING):
    if name not in MODELS:
        raise ValueError(f"Modello '{name}' sconosciuto. Disponibili: {list(MODELS)}")
    pipe = MODELS[name]()
    if name != 'xgb_tuned':
        pipe.set_params(target_enc__smoothing=float(smoothing))
    return pipe




