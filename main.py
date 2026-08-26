import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

# Add src package to system path
sys.path.append(os.path.abspath("."))

from src.data import load_raw_dataset
from src.features import build_feature_pipeline
from src.models import train_lgb_time_series


def main():
    start_time = time.time()
    print("=" * 60)
    print("      IEEE-CIS FRAUD DETECTION: END-TO-END PIPELINE       ")
    print("=" * 60)

    # Directories
    data_dir = "data/raw"
    processed_dir = "data/processed"
    models_dir = "models"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD & ENGINEER TRAIN DATA
    # ---------------------------------------------------------
    print("\n[Step 1/4] Loading and engineering training data...")
    # Set nrows=None for full training set (~590k rows)
    df_train_raw = load_raw_dataset(data_dir=data_dir, split="train", nrows=None)
    df_train_fe = build_feature_pipeline(df_train_raw)

    drop_cols = ["TransactionID", "TransactionDT", "isFraud", "uid1", "uid2"]

    # ---------------------------------------------------------
    # 2. TRAIN TIME-SERIES CV MODELS
    # ---------------------------------------------------------
    print("\n[Step 2/4] Training 5-Fold Time-Series LightGBM models...")
    models, oof_preds, mean_auc = train_lgb_time_series(
        df=df_train_fe,
        target_col="isFraud",
        drop_cols=drop_cols,
        n_splits=5
    )

    # Save trained fold models to disk
    print("\n[Artifact Caching] Saving fold models to disk...")
    for fold_idx, model in enumerate(models, start=1):
        model_path = os.path.join(models_dir, f"lgb_fold_{fold_idx}.joblib")
        joblib.dump(model, model_path)
    print(f"Saved {len(models)} fold models to '{models_dir}/'")

    # ---------------------------------------------------------
    # 3. LOAD & ENGINEER TEST DATA
    # ---------------------------------------------------------
    print("\n[Step 3/4] Loading and engineering test dataset...")
    df_test_raw = load_raw_dataset(data_dir=data_dir, split="test", nrows=None)
    df_test_fe = build_feature_pipeline(df_test_raw)

    # Align test feature columns exactly with train
    feature_cols = [c for c in df_train_fe.columns if c not in drop_cols]
    X_test = df_test_fe[feature_cols].copy()

    # Format categorical dtypes
    cat_cols = X_test.select_dtypes(include=["object", "string"]).columns
    for c in cat_cols:
        X_test[c] = X_test[c].astype("category")

    # ---------------------------------------------------------
    # 4. ENSEMBLE INFERENCE & SUBMISSION GENERATION
    # ---------------------------------------------------------
    print("\n[Step 4/4] Generating out-of-fold test predictions...")
    test_preds = np.zeros(len(X_test))

    for fold_idx, model in enumerate(models, start=1):
        test_preds += model.predict_proba(X_test)[:, 1] / len(models)

    submission_path = os.path.join(processed_dir, "submission.csv")
    sub = pd.DataFrame({
        "TransactionID": df_test_raw["TransactionID"],
        "isFraud": test_preds
    })
    sub.to_csv(submission_path, index=False)

    elapsed_time = (time.time() - start_time) / 60
    print("\n" + "=" * 60)
    print("              PIPELINE EXECUTION COMPLETE             ")
    print("=" * 60)
    print(f"Mean CV ROC-AUC    : {mean_auc:.5f}")
    print(f"Submission Path    : {submission_path}")
    print(f"Total Output Rows  : {len(sub):,}")
    print(f"Total Run Time     : {elapsed_time:.2f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()