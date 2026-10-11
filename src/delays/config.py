from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'model'
OUTPUT_DIR = ROOT / 'outputs'

ERROR_THRESHOLD = -1400
CLIP_LOWER = -20
CLIP_COLS = ['p0q0', 'p0q2', 'p0q3', 'p0q4', 'p2q0', 'p3q0', 'p4q0']
SMOOTHING = 3


FEATURES = [
    'arret', 'gare_freq', 'quarter', 'gare_arret', 'holidays',
    'p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4',
    'gare', 'propagation_index','propagation_index_2'
]
TARGET = 'p0q0'
CAT_COLS = ['gare', 'arret', 'quarter', 'gare_arret','p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4']
VAL_QUANTILE = 0.80

XGB_PARAMS = {
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

MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)
VAL_PATH = os.path.join(MODEL_DIR, 'pipe_val.joblib')
FINAL_PATH = os.path.join(MODEL_DIR, 'pipe_final.joblib')
META_PATH = os.path.join(MODEL_DIR, 'metadata.joblib')

CLIP_COLS = ['p0q0', 'p0q2', 'p0q3', 'p0q4', 'p2q0', 'p3q0', 'p4q0']