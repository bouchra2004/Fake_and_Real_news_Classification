"""
Data Loading Module

Handles loading the Fake.csv and True.csv files, labeling them,
merging into a single DataFrame, and performing initial validation.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def get_project_root() -> str:
    """Return the project root directory (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path(filename: str, subfolder: str = "raw") -> str:
    """
    Build the full path to a data file.

    Args:
        filename: Name of the file (e.g., 'Fake.csv').
        subfolder: Subfolder under data/ ('raw' or 'processed').

    Returns:
        Absolute path to the file.
    """
    return os.path.join(get_project_root(), "data", subfolder, filename)


def load_raw_data(
    fake_path: Optional[str] = None,
    true_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw Fake.csv and True.csv files.

    Args:
        fake_path: Path to Fake.csv. Defaults to data/raw/Fake.csv.
        true_path: Path to True.csv. Defaults to data/raw/True.csv.

    Returns:
        Tuple of (fake_df, true_df) DataFrames.

    Raises:
        FileNotFoundError: If either file is missing.
    """
    if fake_path is None:
        fake_path = get_data_path("Fake.csv")
    if true_path is None:
        true_path = get_data_path("True.csv")

    if not os.path.exists(fake_path):
        raise FileNotFoundError(
            f"Fake.csv not found at {fake_path}. "
            "Please download it from https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset "
            "and place it in data/raw/"
        )
    if not os.path.exists(true_path):
        raise FileNotFoundError(
            f"True.csv not found at {true_path}. "
            "Please download it from https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset "
            "and place it in data/raw/"
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    return fake_df, true_df


def create_labeled_dataset(
    fake_df: pd.DataFrame,
    true_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add labels and merge the two DataFrames.

    Labels:
        - Fake articles: label = 0
        - Real articles: label = 1

    Args:
        fake_df: DataFrame of fake news articles.
        true_df: DataFrame of real news articles.

    Returns:
        Merged DataFrame with a 'label' column.
    """
    fake_df = fake_df.copy()
    true_df = true_df.copy()

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)

    return df


def load_and_merge_data(
    fake_path: Optional[str] = None,
    true_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load, label, and merge both CSV files into a single DataFrame.

    This is the main entry point for data loading.

    Args:
        fake_path: Optional path to Fake.csv.
        true_path: Optional path to True.csv.

    Returns:
        Merged DataFrame with columns from the CSVs plus 'label'.
    """
    fake_df, true_df = load_raw_data(fake_path, true_path)
    df = create_labeled_dataset(fake_df, true_df)
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Perform basic validation checks on the merged dataset.

    Returns:
        Dictionary with validation results.
    """
    results = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "label_distribution": df["label"].value_counts().to_dict(),
        "has_text_column": "text" in df.columns,
        "has_title_column": "title" in df.columns,
        "has_label_column": "label" in df.columns,
        "duplicates_count": df.duplicated().sum(),
    }

    # Check for empty texts
    if "text" in df.columns:
        results["empty_texts"] = (df["text"].str.strip() == "").sum()
        results["null_texts"] = df["text"].isnull().sum()

    return results


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[list] = None,
    keep: str = "first"
) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows from the dataset.

    Args:
        df: Input DataFrame.
        subset: Columns to consider for identifying duplicates.
                Defaults to ['title', 'text'] if available.
        keep: Which duplicate to keep ('first', 'last', False).

    Returns:
        Tuple of (cleaned DataFrame, number of duplicates removed).
    """
    if subset is None:
        available_cols = [c for c in ["title", "text"] if c in df.columns]
        subset = available_cols if available_cols else None

    n_before = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    n_removed = n_before - len(df_clean)

    return df_clean, n_removed

