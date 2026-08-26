"""
src/data.py
Data loading, merging, RAM optimization (downcasting), and dataset splitting modules
for the IEEE-CIS Fraud Detection pipeline.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterates through numeric columns of a dataframe and downcasts datatypes
    to reduce the memory footprint without precision loss for model training.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and not pd.api.types.is_datetime64_any_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"Memory usage decreased to {end_mem:.2f} MB ({reduction:.1f}% reduction)")

    return df


def load_raw_dataset(
    data_dir: str = "data/raw",
    split: str = "train",
    nrows: Optional[int] = None,
    optimize_memory: bool = True,
) -> pd.DataFrame:
    """
    Loads transaction and identity CSVs and executes a left join on TransactionID.

    Args:
        data_dir: Relative path to raw data directory.
        split: 'train' or 'test'.
        nrows: Number of rows to read (useful for quick exploratory runs).
        optimize_memory: Whether to downcast numerical data types.

    Returns:
        Merged pandas DataFrame.
    """
    base_path = Path(data_dir)
    trans_path = base_path / f"{split}_transaction.csv"
    id_path = base_path / f"{split}_identity.csv"

    if not trans_path.exists():
        raise FileNotFoundError(f"Missing expected file: {trans_path}")

    print(f"Loading {split}_transaction.csv...")
    df_trans = pd.read_csv(trans_path, nrows=nrows)

    if id_path.exists():
        print(f"Loading {split}_identity.csv...")
        df_id = pd.read_csv(id_path, nrows=nrows)
        
        print("Performing left join on TransactionID...")
        df = df_trans.merge(df_id, on="TransactionID", how="left")
        del df_trans, df_id
    else:
        print(f"Warning: {id_path.name} not found. Proceeding with transaction data only.")
        df = df_trans

    if optimize_memory:
        print("Optimizing memory footprint...")
        df = reduce_mem_usage(df)

    return df


def time_based_train_val_split(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    val_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset chronologically based on TransactionDT to prevent data leakage.

    Args:
        df: Input DataFrame containing TransactionDT.
        time_col: Column name representing time delta.
        val_ratio: Fraction of latest time data reserved for validation.

    Returns:
        (train_df, val_df) tuple.
    """
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - val_ratio))

    train_df = df_sorted.iloc[:split_idx].copy()
    val_df = df_sorted.iloc[split_idx:].copy()

    print(
        f"Time split complete: Train shape {train_df.shape}, Val shape {val_df.shape}"
    )
    return train_df, val_df