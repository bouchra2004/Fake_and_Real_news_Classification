"""
Tests for data validation and loading.

Note: Tests requiring the actual CSV files are skipped if the
dataset is not present. This allows the test suite to pass
during CI or before the user downloads the data.
"""

import os
import sys
import pytest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    load_raw_data,
    create_labeled_dataset,
    load_and_merge_data,
    validate_dataset,
    remove_duplicates,
    get_data_path,
)


FAKE_PATH = get_data_path("Fake.csv")
TRUE_PATH = get_data_path("True.csv")
DATA_EXISTS = os.path.exists(FAKE_PATH) and os.path.exists(TRUE_PATH)


class TestGetDataPath:
    def test_returns_string(self):
        path = get_data_path("test.csv")
        assert isinstance(path, str)
        assert "data" in path
        assert "raw" in path

    def test_processed_subfolder(self):
        path = get_data_path("test.csv", subfolder="processed")
        assert "processed" in path


class TestLoadRawData:
    def test_raises_if_file_missing(self):
        with pytest.raises(FileNotFoundError):
            load_raw_data(fake_path="nonexistent.csv", true_path="nonexistent2.csv")

    @pytest.mark.skipif(not DATA_EXISTS, reason="Dataset not downloaded")
    def test_loads_both_files(self):
        fake_df, true_df = load_raw_data()
        assert isinstance(fake_df, pd.DataFrame)
        assert isinstance(true_df, pd.DataFrame)
        assert len(fake_df) > 0
        assert len(true_df) > 0


class TestCreateLabeledDataset:
    def test_creates_labels(self):
        fake_df = pd.DataFrame({"text": ["fake article 1"], "title": ["title 1"]})
        true_df = pd.DataFrame({"text": ["real article 1"], "title": ["title 2"]})

        df = create_labeled_dataset(fake_df, true_df)

        assert "label" in df.columns
        assert df[df["label"] == 0].shape[0] == 1  # fake
        assert df[df["label"] == 1].shape[0] == 1  # real
        assert len(df) == 2

    def test_label_values(self):
        fake_df = pd.DataFrame({"text": ["a", "b"]})
        true_df = pd.DataFrame({"text": ["c", "d", "e"]})

        df = create_labeled_dataset(fake_df, true_df)
        assert set(df["label"].unique()) == {0, 1}
        assert (df["label"] == 0).sum() == 2
        assert (df["label"] == 1).sum() == 3


class TestValidateDataset:
    def test_validates_correctly(self):
        df = pd.DataFrame({
            "title": ["Title 1", "Title 2"],
            "text": ["Article body 1", "Article body 2"],
            "label": [0, 1],
        })
        results = validate_dataset(df)
        assert results["total_rows"] == 2
        assert results["has_text_column"] is True
        assert results["has_title_column"] is True
        assert results["has_label_column"] is True

    def test_handles_missing_columns(self):
        df = pd.DataFrame({"other_col": [1, 2], "label": [0, 1]})
        results = validate_dataset(df)
        assert results["has_text_column"] is False
        assert results["has_title_column"] is False


class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({
            "title": ["A", "A", "B"],
            "text": ["same", "same", "different"],
            "label": [0, 0, 1]
        })
        df_clean, n_removed = remove_duplicates(df)
        assert n_removed == 1
        assert len(df_clean) == 2

    def test_no_duplicates(self):
        df = pd.DataFrame({
            "title": ["A", "B", "C"],
            "text": ["one", "two", "three"],
            "label": [0, 1, 0]
        })
        df_clean, n_removed = remove_duplicates(df)
        assert n_removed == 0
        assert len(df_clean) == 3

