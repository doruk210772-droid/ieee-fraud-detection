import os
import sys
import optuna
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# Ensure src module access
sys.path.append(os.path.abspath(".."))
from src.data import load_raw_dataset
from src.features import build_feature_pipeline

# Suppress Optuna verbose logging (keep stdout clean)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X, y, drop_cols):
    """
    Optuna objective function for evaluating LightGBM hyperparameter trials.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.5, 0.95),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    tss = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, val_idx in tss.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=[],
        )

        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

    return np.mean(scores)


def run_optuna_search(n_trials=30, nrows=100000):
    """
    Loads data, builds features, and executes Optuna hyperparameter optimization.
    """
    print("--- Preparing Data for Hyperparameter Search ---")
    df_raw = load_raw_dataset(data_dir="../data/raw", split="train", nrows=nrows)
    df_fe = build_feature_pipeline(df_raw)

    drop_cols = ["TransactionID", "TransactionDT", "isFraud", "uid1", "uid2"]
    feature_cols = [c for c in df_fe.columns if c not in drop_cols]

    X = df_fe[feature_cols].copy()
    y = df_fe["isFraud"].values

    # Format categorical features
    cat_cols = X.select_dtypes(include=["object", "string"]).columns
    for c in cat_cols:
        X[c] = X[c].astype("category")

    print(f"\nStarting Optuna Study across {n_trials} trials...")
    study = optuna.create_study(direction="maximize")

    # Pass data matrix into objective via lambda call
    study.optimize(lambda trial: objective(trial, X, y, drop_cols), n_trials=n_trials)

    print("\n" + "=" * 45)
    print("         OPTUNA SEARCH COMPLETE              ")
    print("=" * 45)
    print(f"Best Trial ROC-AUC Score : {study.best_value:.5f}")
    print("\nOptimal Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key:20s}: {value}")
    print("=" * 45)

    return study.best_params


if __name__ == "__main__":
    best_hyperparams = run_optuna_search(n_trials=30, nrows=100000)