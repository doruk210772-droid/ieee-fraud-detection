import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


def train_lgb_time_series(df, target_col="isFraud", drop_cols=None, n_splits=5, custom_params=None):
    """
    Trains a LightGBM model using TimeSeriesSplit cross-validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Engineered feature matrix.
    target_col : str
        Name of the binary target column.
    drop_cols : list
        Columns to exclude from training features.
    n_splits : int
        Number of time-series splits for validation.
    custom_params : dict
        Optional hyperparameter dictionary to override defaults.
    """
    if drop_cols is None:
        drop_cols = ["TransactionID", "TransactionDT", target_col]

    # Select feature columns
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].values

    # Convert object/string columns to category for LightGBM native support
    cat_cols = X.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        X[col] = X[col].astype("category")

    # Optuna-optimized default hyperparameters
    default_params = {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,  # Suppresses positive split gain warning logs
    }

    # Override defaults if custom hyperparameter dictionary is provided
    if custom_params:
        default_params.update(custom_params)

    tss = TimeSeriesSplit(n_splits=n_splits)
    models = []
    oof_preds = np.zeros(len(df))
    cv_scores = []

    print(f"Starting Time-Series CV ({n_splits} Folds) on {len(feature_cols)} features...")

    for fold, (train_idx, val_idx) in enumerate(tss.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = LGBMClassifier(**default_params)
        
        model.fit(
            X_train,
            y_train,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=[],
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds

        fold_auc = roc_auc_score(y_val, val_preds)
        cv_scores.append(fold_auc)
        models.append(model)

        print(f"Fold {fold} ROC-AUC: {fold_auc:.5f}")

    mean_auc = np.mean(cv_scores)
    print("-" * 35)
    print(f"Mean CV ROC-AUC: {mean_auc:.5f}")
    print("-" * 35)

    return models, oof_preds, mean_auc