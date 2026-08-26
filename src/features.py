import numpy as np
import pandas as pd


def add_time_features(df):
    """
    Extracts cyclical time features from TransactionDT.
    """
    df = df.copy()
    # TransactionDT is delta in seconds from reference date
    df["day_of_week"] = (df["TransactionDT"] // (3600 * 24)) % 7
    df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
    return df


def add_email_features(df):
    """
    Splits email domains into provider vendor and top-level domain (TLD).
    """
    df = df.copy()
    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in df.columns:
            # Fill missing emails prior to string split
            df[col] = df[col].astype(str).fillna("missing")
            
            # Split domain into provider (e.g. gmail) and TLD (e.g. com)
            df[f"{col}_vendor"] = df[col].apply(lambda x: x.split(".")[0] if x != "missing" else "missing")
            df[f"{col}_tld"] = df[col].apply(lambda x: x.split(".")[-1] if "." in x else "missing")
            
    return df


def add_frequency_encoding(df, cat_cols):
    """
    Computes frequency encoding (value counts) for categorical columns.
    """
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            fq = df[col].value_counts(dropna=False).to_dict()
            df[f"{col}_fq_enc"] = df[col].map(fq)
    return df


def create_uids(df):
    """
    Creates composite User IDs for grouping user behavioral statistics.
    """
    df = df.copy()
    
    # Standard pseudo-UIDs from Kaggle IEEE-CIS winning solutions
    df["uid1"] = df["card1"].astype(str) + "_" + df["card2"].astype(str)
    df["uid2"] = (
        df["card1"].astype(str)
        + "_"
        + df["card2"].astype(str)
        + "_"
        + df["card3"].astype(str)
        + "_"
        + df["card5"].astype(str)
    )
    return df


def add_aggregation_features(df):
    """
    Calculates mean, std, differences, and RATIOS grouped by composite UIDs.
    """
    df = df.copy()
    eps = 1e-5  # Prevents division by zero in ratio calculations

    for uid in ["uid1"]:
        if uid in df.columns and "TransactionAmt" in df.columns:
            # Group means & stds
            group_mean = df.groupby(uid)["TransactionAmt"].transform("mean")
            group_std = df.groupby(uid)["TransactionAmt"].transform("std")

            # Differences
            df[f"TransactionAmt_mean_by_{uid}"] = group_mean
            df[f"TransactionAmt_std_by_{uid}"] = group_std
            df[f"TransactionAmt_diff_{uid}_mean"] = df["TransactionAmt"] - group_mean

            # Ratios (Amt / Group_Mean)
            df[f"TransactionAmt_ratio_{uid}_mean"] = df["TransactionAmt"] / (group_mean + eps)

    return df


def add_null_counts(df):
    """
    Counts total missing values per row as an anomaly signal.
    """
    df = df.copy()
    df["null_count"] = df.isnull().sum(axis=1)
    return df


def build_feature_pipeline(df):
    """
    Main feature engineering pipeline wrapper.
    """
    print("Starting feature engineering pipeline...")
    
    # 1. Row-wise missing value counts
    df = add_null_counts(df)
    
    # 2. Time features
    df = add_time_features(df)
    
    # 3. Email domain parsing
    df = add_email_features(df)
    
    # 4. User identity construction
    df = create_uids(df)
    
    # 5. Group-level aggregations & ratios
    df = add_aggregation_features(df)
    
    # 6. Frequency encodings
    freq_cols = ["card1", "card2", "card3", "card5", "P_emaildomain", "R_emaildomain"]
    df = add_frequency_encoding(df, freq_cols)
    
    print(f"Feature engineering complete. Total columns: {df.shape[1]}")
    return df