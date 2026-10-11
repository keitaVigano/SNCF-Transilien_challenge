import argparse

from delays.config import FEATURES, TARGET
from delays.data import prepare
from delays.registry import MODELS, get_model
from delays.validation import validate, train_final


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=list(MODELS), default='xgb')
    p.add_argument('--validation', action='store_true', help='solo validazione')
    p.add_argument('--final', action='store_true', help='solo modello finale')
    a = p.parse_args()

    fai_val = a.validation or not a.final
    fai_final = a.final or not a.validation

    df, train_df, val_df, X_test, gare_freq = prepare()
    Xtr, ytr = train_df[FEATURES], train_df[TARGET]
    Xva, yva = val_df[FEATURES], val_df[TARGET]

    if fai_val:
        validate(a.model, get_model(a.model), Xtr, ytr, Xva, yva)

    if fai_final:
        train_final(a.model, get_model(a.model),
                    df[FEATURES], df[TARGET], X_test[FEATURES], gare_freq)


if __name__ == '__main__':
    main()


