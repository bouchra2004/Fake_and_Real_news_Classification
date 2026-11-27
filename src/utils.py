"""
Utility Module

Shared helper functions used across the project:
path resolution, timing, result saving, and display helpers.
"""

import os
import json
import time
from typing import Any
from functools import wraps


def get_project_root() -> str:
    """Return the project root directory (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def get_model_dir() -> str:
    """Return path to models/ directory, creating it if needed."""
    return ensure_dir(os.path.join(get_project_root(), "models"))


def get_reports_dir() -> str:
    """Return path to reports/ directory, creating it if needed."""
    return ensure_dir(os.path.join(get_project_root(), "reports"))


def get_figures_dir() -> str:
    """Return path to reports/figures/ directory, creating it if needed."""
    return ensure_dir(os.path.join(get_project_root(), "reports", "figures"))


def get_results_dir() -> str:
    """Return path to reports/results/ directory, creating it if needed."""
    return ensure_dir(os.path.join(get_project_root(), "reports", "results"))


def timer(func):
    """Decorator to measure and print execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[Timer] {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


def save_results_json(results: dict, filename: str) -> str:
    """
    Save a dictionary of results to a JSON file in reports/results/.

    Args:
        results: Dictionary to save.
        filename: Filename (e.g., 'model_comparison.json').

    Returns:
        Path to the saved file.
    """
    path = os.path.join(get_results_dir(), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    return path


def load_results_json(filename: str) -> dict:
    """
    Load a JSON results file from reports/results/.

    Args:
        filename: Filename to load.

    Returns:
        Dictionary of results.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = os.path.join(get_results_dir(), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal value as a percentage string."""
    return f"{value * 100:.{decimals}f}%"

