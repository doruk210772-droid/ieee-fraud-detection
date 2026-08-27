import sys
from pathlib import Path
import warnings

# Suppress warnings to keep terminal output clean
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import optuna
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold

from src.data import load_raw_dataset
from src.features import build_feature_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X, y):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 6, 14),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    kf = KFold(n_splits=5, shuffle=False)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        scores.append(model.best_score_["valid_0"]["auc"])

    return np.mean(scores)


def run_optuna_search(n_trials=30, nrows=100000):
    print(f"--- Running Full Optuna Search ({n_trials} trials on {nrows} rows) ---")
    data_dir = PROJECT_ROOT / "data" / "raw"

    df_raw = load_raw_dataset(data_dir=data_dir, split="train", nrows=nrows)
    df_feat = build_feature_pipeline(df_raw)

    drop_cols = ["TransactionID", "isFraud", "TransactionDT"]
    features = [c for c in df_feat.columns if c not in drop_cols]

    X = df_feat[features].copy()
    y = df_feat["isFraud"]

    # Convert object and string columns to category for LightGBM compatibility
    cat_cols = X.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        X[col] = X[col].astype("category")

    study = optuna.create_study(direction="maximize")

    for i in range(n_trials):
        print(f"Running Trial {i+1}/{n_trials}...")
        study.optimize(lambda trial: objective(trial, X, y), n_trials=1)
        print(f"  -> Trial {i+1} Score (ROC-AUC): {study.trials[-1].value:.4f}")

    print("\n--- Best Parameters Found ---")
    print(study.best_params)
    print(f"Best Score: {study.best_value:.5f}")
    return study.best_params


if __name__ == "__main__":
    run_optuna_search(n_trials=30, nrows=100000)