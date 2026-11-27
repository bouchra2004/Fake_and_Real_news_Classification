"""
Model Evaluation Module

Provides functions to evaluate trained models, compute classification metrics,
generate confusion matrices, compare models, and perform error analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script/server use
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.utils import get_figures_dir, get_results_dir


LABEL_NAMES = {0: "Fake", 1: "Real"}


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    model_name: str = "Model"
) -> dict:
    """
    Evaluate a single model on test data.

    Computes accuracy, precision, recall, F1-score, and confusion matrix.

    Args:
        pipeline: Trained sklearn Pipeline.
        X_test: Test text data.
        y_test: True test labels.
        model_name: Name for display/logging.

    Returns:
        Dictionary with all metrics.
    """
    y_pred = pipeline.predict(X_test)

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Fake", "Real"], zero_division=0
        ),
    }

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {metrics['confusion_matrix']}")
    print(f"\n{metrics['classification_report']}")

    return metrics


def evaluate_all_models(
    model_results: Dict[str, dict],
    X_test: pd.Series,
    y_test: pd.Series
) -> Dict[str, dict]:
    """
    Evaluate all trained models and collect metrics.

    Args:
        model_results: Output from train_all_models(), dict of {name: {pipeline, train_time}}.
        X_test: Test text data.
        y_test: True test labels.

    Returns:
        Dictionary mapping model names to their evaluation metrics.
    """
    all_metrics = {}

    for name, info in model_results.items():
        metrics = evaluate_model(info["pipeline"], X_test, y_test, name)
        metrics["train_time"] = info.get("train_time", None)
        all_metrics[name] = metrics

    return all_metrics


def create_comparison_table(all_metrics: Dict[str, dict]) -> pd.DataFrame:
    """
    Create a comparison DataFrame of all model metrics.

    Returns:
        DataFrame with one row per model and columns for each metric.
    """
    rows = []
    for name, metrics in all_metrics.items():
        rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1_score"],
            "Train Time (s)": metrics.get("train_time", None),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("F1-Score", ascending=False).reset_index(drop=True)
    return df


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    save: bool = True
) -> str:
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        model_name: Name for the title and filename.
        save: Whether to save the figure.

    Returns:
        Path to saved figure (empty string if not saved).
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.tight_layout()

    path = ""
    if save:
        figures_dir = get_figures_dir()
        safe_name = model_name.replace(" ", "_").lower()
        path = os.path.join(figures_dir, f"confusion_matrix_{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {path}")

    plt.close(fig)
    return path


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save: bool = True
) -> str:
    """
    Create a grouped bar chart comparing model metrics.

    Args:
        comparison_df: DataFrame from create_comparison_table().
        save: Whether to save the figure.

    Returns:
        Path to saved figure.
    """
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    df_plot = comparison_df.set_index("Model")[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot.plot(kind="bar", ax=ax, width=0.8, edgecolor="black", linewidth=0.5)

    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)

    plt.tight_layout()

    path = ""
    if save:
        figures_dir = get_figures_dir()
        path = os.path.join(figures_dir, "model_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Model comparison chart saved to {path}")

    plt.close(fig)
    return path


def analyze_errors(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    model_name: str,
    n_examples: int = 5
) -> dict:
    """
    Analyze false positives and false negatives.

    Args:
        pipeline: Trained pipeline.
        X_test: Test texts.
        y_test: True labels.
        model_name: Name of the model.
        n_examples: Number of examples to show per error type.

    Returns:
        Dictionary with error analysis results.
    """
    y_pred = pipeline.predict(X_test)

    # Create a temporary DataFrame for analysis
    error_df = pd.DataFrame({
        "text": X_test.values,
        "true_label": y_test.values,
        "predicted_label": y_pred,
    })

    # False Positives: Real articles classified as Fake (true=1, pred=0)
    fp_mask = (error_df["true_label"] == 1) & (error_df["predicted_label"] == 0)
    false_positives = error_df[fp_mask]

    # False Negatives: Fake articles classified as Real (true=0, pred=1)
    fn_mask = (error_df["true_label"] == 0) & (error_df["predicted_label"] == 1)
    false_negatives = error_df[fn_mask]

    def extract_examples(subset: pd.DataFrame, n: int) -> list:
        examples = []
        for _, row in subset.head(n).iterrows():
            text_preview = row["text"][:300] + "..." if len(row["text"]) > 300 else row["text"]
            examples.append({
                "text_preview": text_preview,
                "true_label": LABEL_NAMES[row["true_label"]],
                "predicted_label": LABEL_NAMES[row["predicted_label"]],
            })
        return examples

    analysis = {
        "model_name": model_name,
        "total_errors": int(fp_mask.sum() + fn_mask.sum()),
        "false_positives_count": int(fp_mask.sum()),
        "false_negatives_count": int(fn_mask.sum()),
        "false_positive_examples": extract_examples(false_positives, n_examples),
        "false_negative_examples": extract_examples(false_negatives, n_examples),
    }

    print(f"\nError Analysis — {model_name}")
    print(f"  Total errors: {analysis['total_errors']}")
    print(f"  False Positives (Real→Fake): {analysis['false_positives_count']}")
    print(f"  False Negatives (Fake→Real): {analysis['false_negatives_count']}")

    return analysis


def save_all_metrics(all_metrics: Dict[str, dict], filename: str = "model_metrics.json") -> str:
    """
    Save all model metrics to a JSON file.

    Args:
        all_metrics: Dictionary of model evaluation results.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    results_dir = get_results_dir()
    path = os.path.join(results_dir, filename)

    # Make serializable
    serializable = {}
    for name, metrics in all_metrics.items():
        serializable[name] = {
            k: v for k, v in metrics.items()
            if k != "classification_report"  # Keep report as string
        }
        serializable[name]["classification_report"] = metrics.get("classification_report", "")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"All metrics saved to {path}")
    return path

