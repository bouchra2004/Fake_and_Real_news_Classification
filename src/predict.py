"""
Prediction Module

Loads a saved pipeline and provides prediction on new text inputs.
Used by the Streamlit app and can be called from scripts or notebooks.
"""

import os
import joblib
from typing import Optional

from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text, validate_text_input
from src.utils import get_model_dir


LABEL_MAP = {0: "Fake News", 1: "Real News"}
DEFAULT_MODEL_FILENAME = "best_model.joblib"


def load_pipeline(filename: Optional[str] = None) -> Pipeline:
    """
    Load a saved sklearn Pipeline from the models/ directory.

    Args:
        filename: Model filename. Defaults to 'best_model.joblib'.

    Returns:
        Loaded sklearn Pipeline.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    if filename is None:
        filename = DEFAULT_MODEL_FILENAME

    model_dir = get_model_dir()
    path = os.path.join(model_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Please train the models first by running: python -m src.train"
        )

    pipeline = joblib.load(path)
    return pipeline


def predict_text(
    text: str,
    pipeline: Optional[Pipeline] = None,
    model_filename: Optional[str] = None,
    remove_source_markers: bool = True
) -> dict:
    """
    Predict whether a text is Fake or Real news.

    Steps:
    1. Validate the input text.
    2. Clean the text using the same preprocessing as training.
    3. Pass through the pipeline (TF-IDF + classifier).
    4. Return the prediction with metadata.

    Args:
        text: Raw input text to classify.
        pipeline: Pre-loaded Pipeline. If None, loads from file.
        model_filename: Filename to load if pipeline is None.
        remove_source_markers: Whether to remove Reuters-style markers.

    Returns:
        Dictionary with prediction results:
        - 'label': 0 or 1
        - 'label_name': 'Fake News' or 'Real News'
        - 'input_valid': bool
        - 'validation_message': str
        - 'word_count': int
        - 'cleaned_text_preview': first 200 chars of cleaned text
    """
    # Step 1: Validate input
    validation = validate_text_input(text)
    if not validation["is_valid"]:
        return {
            "label": None,
            "label_name": None,
            "input_valid": False,
            "validation_message": validation["message"],
            "word_count": validation["word_count"],
            "cleaned_text_preview": "",
        }

    # Step 2: Load pipeline if not provided
    if pipeline is None:
        pipeline = load_pipeline(model_filename)

    # Step 3: Clean text (same preprocessing as training)
    cleaned = clean_text(text, remove_source_markers=remove_source_markers)

    if not cleaned.strip():
        return {
            "label": None,
            "label_name": None,
            "input_valid": False,
            "validation_message": "Text is empty after cleaning. Please enter meaningful content.",
            "word_count": 0,
            "cleaned_text_preview": "",
        }

    # Step 4: Predict
    prediction = pipeline.predict([cleaned])[0]

    # Compute optional decision score or probability
    decision_score = None
    confidence = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba([cleaned])[0]
            confidence = float(probs[int(prediction)])
        except Exception:
            pass
    if hasattr(pipeline, "decision_function"):
        try:
            raw_score = pipeline.decision_function([cleaned])[0]
            decision_score = float(raw_score)
        except Exception:
            pass

    return {
        "label": int(prediction),
        "label_name": LABEL_MAP[int(prediction)],
        "input_valid": True,
        "validation_message": "Prediction successful.",
        "word_count": len(text.split()),
        "cleaned_text_preview": cleaned[:200],
        "decision_score": decision_score,
        "confidence": confidence,
    }


def predict_batch(
    texts: list,
    pipeline: Optional[Pipeline] = None,
    model_filename: Optional[str] = None
) -> list:
    """
    Predict labels for a batch of texts.

    Args:
        texts: List of raw text strings.
        pipeline: Pre-loaded Pipeline. If None, loads from file.
        model_filename: Filename to load if pipeline is None.

    Returns:
        List of prediction result dictionaries.
    """
    if pipeline is None:
        pipeline = load_pipeline(model_filename)

    return [predict_text(text, pipeline=pipeline) for text in texts]

