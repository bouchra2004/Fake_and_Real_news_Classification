"""
Model Training Module

Builds scikit-learn Pipelines (TF-IDF + Classifier), trains multiple models,
and saves the best pipeline for production use.

Key design decisions:
- TF-IDF is INSIDE the Pipeline to prevent data leakage.
  The vectorizer learns vocabulary/IDF only from training data.
- All models use the same train/test split for fair comparison.
- Random seeds are fixed for reproducibility.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from src.utils import get_model_dir, get_results_dir, ensure_dir


# ── TF-IDF configuration ─────────────────────────────────────────────────
# Justification for each parameter:
# - lowercase=True: normalizes case so "Fake" and "fake" are the same token
# - ngram_range=(1,2): includes bigrams for phrases like "fake news", "white house"
# - max_features=50000: limits vocabulary size to reduce noise from rare words
# - min_df=3: ignores terms appearing in fewer than 3 documents (likely typos/noise)
# - max_df=0.95: ignores terms appearing in >95% of documents (too common to be discriminative)
# - sublinear_tf=True: applies log(1+tf) dampening so very long documents don't dominate
# - stop_words='english': removes common English words that don't carry meaning

TFIDF_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "max_features": 50000,
    "min_df": 3,
    "max_df": 0.95,
    "sublinear_tf": True,
    "stop_words": "english",
}


def build_pipelines() -> Dict[str, Pipeline]:
    """
    Create sklearn Pipelines for each model.

    Each pipeline includes TF-IDF vectorization + classifier,
    ensuring the vectorizer is fit only on training data.

    Returns:
        Dictionary mapping model names to Pipeline objects.
    """
    pipelines = {
        "DummyClassifier": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("classifier", DummyClassifier(
                strategy="most_frequent",
                random_state=42
            ))
        ]),
        "MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("classifier", MultinomialNB(alpha=1.0))
        ]),
        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver="lbfgs",
                C=1.0
            ))
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("classifier", LinearSVC(
                max_iter=2000,
                random_state=42,
                C=1.0
            ))
        ]),
    }
    return pipelines


def prepare_train_test_split(
    df: pd.DataFrame,
    text_column: str = "clean_text",
    label_column: str = "label",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train and test sets with stratification.

    Args:
        df: DataFrame with text and label columns.
        text_column: Name of the column containing preprocessed text.
        label_column: Name of the label column.
        test_size: Fraction of data for testing (default 0.2 = 20%).
        random_state: Random seed for reproducibility.

    Returns:
        (X_train, X_test, y_train, y_test) as pandas Series.
    """
    X = df[text_column]
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    print(f"Train label distribution:\n{y_train.value_counts().to_string()}")
    print(f"Test label distribution:\n{y_test.value_counts().to_string()}")

    return X_train, X_test, y_train, y_test


def train_all_models(
    X_train: pd.Series,
    y_train: pd.Series,
    pipelines: Optional[Dict[str, Pipeline]] = None
) -> Dict[str, dict]:
    """
    Train all model pipelines and record training times.

    Args:
        X_train: Training text data.
        y_train: Training labels.
        pipelines: Dictionary of pipelines. Built from build_pipelines() if None.

    Returns:
        Dictionary mapping model names to {'pipeline': fitted_pipeline, 'train_time': seconds}.
    """
    if pipelines is None:
        pipelines = build_pipelines()

    results = {}

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start
        print(f"  → {name} trained in {train_time:.2f}s")

        results[name] = {
            "pipeline": pipeline,
            "train_time": train_time
        }

    return results


def save_pipeline(pipeline: Pipeline, filename: str) -> str:
    """
    Save a trained pipeline to the models/ directory.

    Args:
        pipeline: Fitted sklearn Pipeline.
        filename: Filename (e.g., 'best_model.joblib').

    Returns:
        Path to the saved file.
    """
    model_dir = get_model_dir()
    path = os.path.join(model_dir, filename)
    joblib.dump(pipeline, path)
    print(f"Pipeline saved to {path}")
    return path


def save_training_results(
    model_results: Dict[str, dict],
    filename: str = "training_times.json"
) -> str:
    """
    Save training metadata (times) to a JSON file.

    Args:
        model_results: Output from train_all_models().
        filename: Output filename.

    Returns:
        Path to the saved file.
    """
    results_dir = get_results_dir()
    times = {
        name: {"train_time_seconds": info["train_time"]}
        for name, info in model_results.items()
    }
    path = os.path.join(results_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(times, f, indent=2)
    print(f"Training results saved to {path}")
    return path

