"""
Training Runner Script

Orchestrates the full training pipeline:
1. Load and validate data
2. Preprocess text
3. Remove duplicates
4. Train/test split
5. Train all models
6. Evaluate all models
7. Save best pipeline and metrics

Run from project root:
    python run_training.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_and_merge_data, validate_dataset, remove_duplicates
from src.preprocessing import clean_text, combine_title_text
from src.train import build_pipelines, prepare_train_test_split, train_all_models, save_pipeline, save_training_results
from src.evaluate import (
    evaluate_all_models,
    create_comparison_table,
    plot_confusion_matrix,
    plot_model_comparison,
    analyze_errors,
    save_all_metrics,
)
from src.utils import get_results_dir, save_results_json


def main():
    print("=" * 60)
    print("FAKE NEWS DETECTION — TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load Data ─────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    df = load_and_merge_data()
    validation = validate_dataset(df)
    print(f"  Total articles: {validation['total_rows']}")
    print(f"  Columns: {validation['columns']}")
    print(f"  Label distribution: {validation['label_distribution']}")
    print(f"  Missing values: {validation['missing_values']}")
    print(f"  Duplicates: {validation['duplicates_count']}")

    # ── Step 2: Remove Duplicates ─────────────────────────────────────
    print("\n[2/7] Removing duplicates...")
    df, n_removed = remove_duplicates(df, subset=["title", "text"])
    print(f"  Removed {n_removed} duplicate articles")
    print(f"  Remaining articles: {len(df)}")

    # ── Step 3: Preprocess Text ───────────────────────────────────────
    print("\n[3/7] Preprocessing text...")

    # Combine title + text for richer features
    df["combined_text"] = df.apply(
        lambda row: combine_title_text(
            row.get("title", ""),
            row.get("text", "")
        ),
        axis=1
    )

    # Clean the combined text
    df["clean_text"] = df["combined_text"].apply(
        lambda x: clean_text(x, remove_source_markers=True)
    )

    # Remove rows with empty cleaned text
    empty_mask = df["clean_text"].str.strip() == ""
    n_empty = empty_mask.sum()
    if n_empty > 0:
        print(f"  Removed {n_empty} articles with empty text after cleaning")
        df = df[~empty_mask].reset_index(drop=True)

    print(f"  Final dataset size: {len(df)}")

    # ── Step 4: Train/Test Split ──────────────────────────────────────
    print("\n[4/7] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df,
        text_column="clean_text",
        label_column="label",
        test_size=0.2,
        random_state=42
    )

    # ── Step 5: Train Models ──────────────────────────────────────────
    print("\n[5/7] Training models...")
    model_results = train_all_models(X_train, y_train)
    save_training_results(model_results)

    # ── Step 6: Evaluate Models ───────────────────────────────────────
    print("\n[6/7] Evaluating models...")
    all_metrics = evaluate_all_models(model_results, X_test, y_test)

    # Create comparison table
    comparison_df = create_comparison_table(all_metrics)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    # Save metrics
    save_all_metrics(all_metrics)

    # Save comparison table
    comparison_df.to_csv(
        os.path.join(get_results_dir(), "model_comparison.csv"),
        index=False
    )

    # Generate confusion matrices for all models
    for name, info in model_results.items():
        y_pred = info["pipeline"].predict(X_test)
        plot_confusion_matrix(y_test, y_pred, name)

    # Generate comparison chart
    plot_model_comparison(comparison_df)

    # ── Step 7: Save Best Model ───────────────────────────────────────
    print("\n[7/7] Saving best model...")

    # Select best model by F1-Score (not just accuracy)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_pipeline = model_results[best_model_name]["pipeline"]
    best_f1 = comparison_df.iloc[0]["F1-Score"]

    save_pipeline(best_pipeline, "best_model.joblib")
    print(f"  Best model: {best_model_name} (F1-Score: {best_f1:.4f})")

    # Also save each model individually
    for name, info in model_results.items():
        safe_name = name.replace(" ", "_").lower()
        save_pipeline(info["pipeline"], f"{safe_name}.joblib")

    # ── Error Analysis on Best Model ──────────────────────────────────
    print("\n" + "=" * 60)
    print(f"ERROR ANALYSIS — {best_model_name}")
    print("=" * 60)
    error_analysis = analyze_errors(best_pipeline, X_test, y_test, best_model_name, n_examples=5)
    save_results_json(error_analysis, "error_analysis.json")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Dataset size: {len(df)} articles")
    print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"  Best model: {best_model_name}")
    print(f"  Best F1-Score: {best_f1:.4f}")
    print(f"\nAll results saved to reports/results/")
    print(f"All figures saved to reports/figures/")
    print(f"All models saved to models/")
    print(f"\nTo launch the app:")
    print(f"  streamlit run app/app.py")


if __name__ == "__main__":
    main()

