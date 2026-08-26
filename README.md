# IEEE-CIS Fraud Detection Pipeline

An end-to-end, modular machine learning pipeline built to detect fraudulent online transactions using the IEEE-CIS Fraud Detection dataset. The system features custom time-based feature engineering, time-series cross-validation, LightGBM modeling, Optuna hyperparameter optimization, and automated artifact saving.

## Project Architecture

```text
ieee-fraud-detection/
├── data/
│   ├── raw/               # Downloaded train & test CSV files
│   └── processed/         # Pipeline outputs (submission.csv)
├── models/                # Trained LightGBM model artifacts (.joblib)
├── notebooks/             # Exploratory work & validation notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_models.ipynb
│   ├── 04_metrics.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   └── 06_submission.ipynb
├── src/                   # Production Python package modules
│   ├── __init__.py
│   ├── data.py            # Data loading & memory optimization
│   ├── features.py        # Feature engineering pipeline
│   ├── models.py          # Time-Series CV LightGBM trainer
│   └── tune.py            # Optuna hyperparameter study
├── main.py                # End-to-end training & prediction script
├── requirements.txt       # Project dependencies
└── README.md
```

---

## Key Features & Methodology

1. **Feature Engineering (`src/features.py`)**:
   - **Cyclical Time Features:** Extracted `day_of_week` and `hour_of_day` from `TransactionDT`.
   - **Email Domain Parsing:** Parsed `P_emaildomain` and `R_emaildomain` into vendor categories and top-level domains (TLDs).
   - **Composite User IDs (UIDs):** Grouped transaction cards across time (`uid1`, `uid2`) to capture behavioral patterns.
   - **Group Aggregations & Ratios:** Computed group means, standard deviations, differences, and ratio metrics (`TransactionAmt_ratio_uid1_mean`).
   - **Anomaly Missingness Signals:** Row-wise `null_count` tracking automated bot behavior.

2. **Cross-Validation (`src/models.py`)**:
   - **5-Fold TimeSeriesSplit:** Prevents future data leakage by ensuring validation folds chronologically succeed training folds.

3. **Hyperparameter Optimization (`src/tune.py`)**:
   - Automated Tree-structured Parzen Estimator (TPE) sampling using **Optuna** to maximize Out-Of-Fold (OOF) ROC-AUC.

---

## Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install required dependencies:

```bash
git clone https://github.com/doruk210772-droid/ieee-fraud-detection.git
cd ieee-fraud-detection
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) dataset from Kaggle and place the raw `.csv` files inside `data/raw/`:
- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`

### 3. Execution

#### Run Hyperparameter Search (Optuna)
To run automated tuning over 30 trials:
```bash
python src/tune.py
```

#### Execute Full Pipeline
To run the full end-to-end sequence (data ingestion, feature engineering, 5-fold CV training, model artifact export to `models/`, and generating `data/processed/submission.csv`):
```bash
python main.py
```

---

## Performance Summary

| Metric | Baseline LightGBM | Upgraded Features + Optuna |
| :--- | :---: | :---: |
| **Validation Mean ROC-AUC** | `0.88819` | **`0.9250+`** |
| **Cross-Validation Strategy** | 5-Fold Time-Series | 5-Fold Time-Series |
| **Evaluation Loss** | Binary Log-Loss | Binary Log-Loss |
