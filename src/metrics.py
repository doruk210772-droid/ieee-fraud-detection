"""
src/metrics.py
Evaluation metrics and visualization module for IEEE-CIS Fraud Detection.
Handles ROC-AUC, Precision-Recall AUC, optimal thresholding, and diagnostic plots.
"""

from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculates summary evaluation metrics for binary classification predictions.
    """
    # Filter out unpredicted indices (e.g., initial time-series fold split gap)
    mask = ~np.isnan(y_pred_proba) & (y_pred_proba > 0)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred_proba[mask]

    roc_auc = roc_auc_score(y_true_clean, y_pred_clean)
    precision, recall, _ = precision_recall_curve(y_true_clean, y_pred_clean)
    pr_auc = auc(recall, precision)

    y_pred_binary = (y_pred_clean >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_clean, y_pred_binary).ravel()

    metrics = {
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "Precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "Recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "F1_Score": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
    }

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray, y_pred_proba: np.ndarray, metric: str = "f1"
) -> Tuple[float, float]:
    """
    Finds the probability threshold that maximizes a specific metric (e.g., F1-score or Youden's J).
    """
    mask = ~np.isnan(y_pred_proba) & (y_pred_proba > 0)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred_proba[mask]

    precision, recall, thresholds = precision_recall_curve(y_true_clean, y_pred_clean)

    if metric.lower() == "f1":
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        best_score = float(f1_scores[best_idx])
    else:
        # Youden's J statistic (Sensitivity + Specificity - 1)
        fpr, tpr, roc_thresholds = roc_curve(y_true_clean, y_pred_clean)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = float(roc_thresholds[best_idx])
        best_score = float(j_scores[best_idx])

    return best_threshold, best_score


def plot_evaluation_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Plots ROC Curve and Precision-Recall Curve side-by-side.
    """
    mask = ~np.isnan(y_pred_proba) & (y_pred_proba > 0)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred_proba[mask]

    fpr, tpr, _ = roc_curve(y_true_clean, y_pred_clean)
    roc_auc = roc_auc_score(y_true_clean, y_pred_clean)

    precision, recall, _ = precision_recall_curve(y_true_clean, y_pred_clean)
    pr_auc = auc(recall, precision)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ROC Curve
    axes[0].plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    # PR Curve
    axes[1].plot(recall, precision, color="#d95f02", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (6, 5),
) -> None:
    """
    Plots an annotated confusion matrix for a specified probability threshold.
    """
    mask = ~np.isnan(y_pred_proba) & (y_pred_proba > 0)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred_proba[mask]

    y_pred_binary = (y_pred_clean >= threshold).astype(int)
    cm = confusion_matrix(y_true_clean, y_pred_binary)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate (0)", "Fraud (1)"],
        yticklabels=["Legitimate (0)", "Fraud (1)"],
    )
    plt.title(f"Confusion Matrix (Threshold = {threshold:.2f})", fontweight="bold")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()