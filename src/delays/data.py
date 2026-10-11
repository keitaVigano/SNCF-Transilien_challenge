import pandas as pd

from delays.config import (
    CLIP_COLS, CLIP_LOWER, DATA_DIR, ERROR_THRESHOLD, VAL_QUANTILE,
)
from delays.features import add_features


def load_raw():
    X = pd.read_csv(DATA_DIR / 'x_train_final.csv').drop(
        columns=['Unnamed: 0', 'Unnamed: 0.1'])
    X_test = pd.read_csv(DATA_DIR / 'x_test_final.csv').drop(
        columns=['Unnamed: 0'])
    Y = pd.read_csv(DATA_DIR / 'y_train_final.csv').drop(
        columns=['Unnamed: 0'])

    assert len(X) == len(Y), "X_train e y_train non hanno la stessa dimensione"

    df = pd.concat([X, Y], axis=1)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    X_test['date'] = pd.to_datetime(X_test['date'], format='%Y-%m-%d')
    return df, X_test


def clean(df, X_test):
    """Rimuove errori di misura (-1440) e applica il clip."""
    df = df[(df['p0q3'] > ERROR_THRESHOLD) & (df['p0q4'] > ERROR_THRESHOLD)].copy()
    df[CLIP_COLS] = df[CLIP_COLS].clip(lower=CLIP_LOWER)

    X_test = X_test.copy()
    test_cols = [c for c in CLIP_COLS if c in X_test.columns]
    X_test[test_cols] = X_test[test_cols].clip(lower=CLIP_LOWER)
    return df, X_test


def build_features(df, X_test):
    gare_freq = df['gare'].value_counts()
    return add_features(df, gare_freq), add_features(X_test, gare_freq), gare_freq


def temporal_split(df):
    df = df.sort_values('date').reset_index(drop=True)
    cutoff = df['date'].quantile(VAL_QUANTILE)
    return df, df[df['date'] < cutoff], df[df['date'] >= cutoff]


def prepare():
    """Ritorna df completo, train, validation, test e gare_freq."""
    df, X_test = load_raw()
    df, X_test = clean(df, X_test)
    df, X_test, gare_freq = build_features(df, X_test)
    df, train_df, val_df = temporal_split(df)
    return df, train_df, val_df, X_test, gare_freq