"""
Tests for the prediction module.

Note: Tests that require a trained model are skipped if the model
file does not exist. This allows tests to pass during development
before training has been run.
"""

import os
import sys
import pytest

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import load_pipeline, predict_text, LABEL_MAP, DEFAULT_MODEL_FILENAME
from src.utils import get_model_dir


MODEL_PATH = os.path.join(get_model_dir(), DEFAULT_MODEL_FILENAME)
MODEL_EXISTS = os.path.exists(MODEL_PATH)


class TestLoadPipeline:
    def test_raises_if_model_missing(self):
        with pytest.raises(FileNotFoundError):
            load_pipeline("nonexistent_model.joblib")

    @pytest.mark.skipif(not MODEL_EXISTS, reason="Model not yet trained")
    def test_loads_successfully(self):
        pipeline = load_pipeline()
        assert pipeline is not None
        assert hasattr(pipeline, "predict")


class TestPredictText:
    @pytest.mark.skipif(not MODEL_EXISTS, reason="Model not yet trained")
    def test_valid_article(self):
        text = (
            "The Senate passed the infrastructure bill today with bipartisan support. "
            "The legislation allocates funds for roads, bridges, and broadband internet "
            "across the United States."
        )
        result = predict_text(text)
        assert result["input_valid"] is True
        assert result["label"] in [0, 1]
        assert result["label_name"] in LABEL_MAP.values()
        assert result["word_count"] > 0

    def test_empty_text(self):
        result = predict_text("")
        assert result["input_valid"] is False
        assert result["label"] is None

    def test_whitespace_text(self):
        result = predict_text("     ")
        assert result["input_valid"] is False
        assert result["label"] is None

    def test_very_short_text(self):
        result = predict_text("Hi")
        assert result["input_valid"] is False

    def test_non_string_input(self):
        result = predict_text(12345)
        assert result["input_valid"] is False

    @pytest.mark.skipif(not MODEL_EXISTS, reason="Model not yet trained")
    def test_result_structure(self):
        text = "Breaking news: major earthquake hits the region causing widespread damage."
        result = predict_text(text)
        expected_keys = {
            "label", "label_name", "input_valid",
            "validation_message", "word_count", "cleaned_text_preview"
        }
        assert expected_keys.issubset(result.keys())

