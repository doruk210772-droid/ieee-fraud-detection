# IEEE-CIS Fraud Detection

This repository contains a complete machine learning pipeline for the [IEEE-CIS Fraud Detection competition on Kaggle](https://www.kaggle.com/c/ieee-fraud-detection). The goal is to predict the probability that an online transaction is fraudulent.

## 🚀 Performance
- **Validation Strategy:** 5-Fold Time-Series Split (`TimeSeriesSplit`)
- **Metric:** ROC-AUC
- **Local CV Score:** ~0.9109
- **Kaggle Public Score:** 0.915556
- **Kaggle Private Score:** 0.894296

## 🛠 Tech Stack
- **Modeling:** LightGBM (`LGBMClassifier`) with early stopping.
- **Hyperparameter Tuning:** Optuna (Bayesian optimization).
- **Data Manipulation:** Pandas & NumPy.
- **Validation & Metrics:** Scikit-learn.

## 📂 Project Structure
```text
ieee-cis-fraud-detection/
├── data/                   # Raw and processed datasets
├── models/                 # Saved LightGBM fold models (.txt/.pkl)
├── notebooks/              # Jupyter notebooks for EDA and prototyping
│   ├── 01_eda_and_missing  # Initial data exploration
│   ├── 02_feat.ipynb       # Feature engineering experiments
│   ├── 03_models.ipynb     # Model training prototypes
│   ├── 04_metrics.ipynb    # Evaluation metrics and plots
│   ├── 05_hyperparameter   # Optuna search notebooks
│   └── create_submission   # Submission generation tests
├── src/                    # Production pipeline modules
│   ├── data.py             # Data loading and memory optimization scripts
│   ├── features.py         # Feature engineering pipeline
│   ├── models.py           # LightGBM architecture and Time-Series CV logic
│   └── tune.py             # Optuna hyperparameter tuning script
├── venv/                   # Virtual environment (ignored in git)
├── main.py                 # Main execution script (Train, Validate, Predict)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## ⚙️ Key Pipeline Features
1. **Memory Optimization:** Downcasts numerical types to significantly reduce Pandas memory usage (crucial for this large 590k+ row dataset).
2. **Native Categorical Handling:** Automatically casts `object` and `string` columns to Pandas `category` dtype, allowing LightGBM to handle them natively without memory-heavy One-Hot Encoding.
3. **Kaggle Hyphen Bug Fix:** Automatically handles the known Kaggle dataset mismatch where `train` identity columns use underscores (`id_01`) and `test` identity columns use hyphens (`id-01`).
4. **Ensemble Inference:** Averages the predictions of all 5 fold models for the final test set submission, creating a more robust and stable prediction.

## 💻 How to Run

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data
Download the dataset from the Kaggle competition page and place the raw CSV files into `data/raw/`.

### 3. Hyperparameter Tuning (Optional)
To search for optimal LightGBM parameters using a subset of the data:
```bash
python -m src.tune
```

### 4. Train Models & Generate Submission
To run the full pipeline (Data Loading -> Feature Engineering -> Cross-Validation Training -> Test Inference):
```bash
python main.py
```
This will output a `submission.csv` file ready to be uploaded to Kaggle.