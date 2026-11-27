"""
Text Preprocessing Module

Provides reusable, testable functions for cleaning and preparing
news article text for TF-IDF vectorization.

Design decisions:
- Lowercase conversion is handled by TfidfVectorizer (lowercase=True),
  so we do NOT apply it here to keep preprocessing and vectorization concerns separate.
- We preserve negation words (not, never, no, etc.) as they can carry meaning.
- We do NOT aggressively remove all numbers or punctuation — only clean
  patterns that add noise (URLs, HTML, excessive whitespace).
- Stop word removal is delegated to TfidfVectorizer for consistency.
"""

import re
from typing import Optional


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove URLs (http, https, www) from text."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_email_addresses(text: str) -> str:
    """Remove email addresses from text."""
    return re.sub(r"\S+@\S+\.\S+", " ", text)


def remove_special_characters(text: str) -> str:
    """
    Remove special characters while preserving letters, digits,
    basic punctuation (periods, commas, apostrophes, hyphens), and spaces.

    We keep apostrophes for contractions (don't, isn't) and hyphens
    for compound words (fact-checking, well-known).
    """
    return re.sub(r"[^a-zA-Z0-9\s.,;:!?'\-]", " ", text)


def normalize_whitespace(text: str) -> str:
    """Replace multiple whitespace characters with a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def remove_reuters_marker(text: str) -> str:
    """
    Remove the Reuters dateline pattern commonly found at the start
    of Real news articles (e.g., 'WASHINGTON (Reuters) - ').

    This marker can act as a classification shortcut — the model might
    learn to classify based on the presence of '(Reuters)' rather than
    the actual content. We document this choice and study its impact
    in the evaluation phase.
    """
    # Pattern: CITY_NAME (Reuters) - or CITY (Reuters) -
    return re.sub(r"^[A-Z\s]+ \(Reuters\)\s*-\s*", "", text)


def clean_text(text: str, remove_source_markers: bool = True) -> str:
    """
    Apply the full text cleaning pipeline.

    Steps (in order):
    1. Remove HTML tags
    2. Remove URLs
    3. Remove email addresses
    4. Optionally remove Reuters dateline markers
    5. Remove special characters (preserving useful punctuation)
    6. Normalize whitespace

    Args:
        text: Raw text string.
        remove_source_markers: Whether to remove Reuters-style markers.
                               Default True to reduce source-based bias.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()
    if not text:
        return ""

    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_email_addresses(text)

    if remove_source_markers:
        text = remove_reuters_marker(text)

    text = remove_special_characters(text)
    text = normalize_whitespace(text)

    return text


def combine_title_text(
    title: Optional[str],
    text: Optional[str],
    separator: str = " "
) -> str:
    """
    Combine title and text into a single string.

    Using both title and text gives the model more signal,
    since titles often contain key information about the article.

    Args:
        title: Article title.
        text: Article body text.
        separator: String to place between title and text.

    Returns:
        Combined string.
    """
    parts = []
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())
    if isinstance(text, str) and text.strip():
        parts.append(text.strip())
    return separator.join(parts)


def validate_text_input(text: str, min_length: int = 10) -> dict:
    """
    Validate a text input for prediction.

    Args:
        text: Input text to validate.
        min_length: Minimum character length after stripping.

    Returns:
        Dictionary with 'is_valid', 'message', and 'word_count'.
    """
    if not isinstance(text, str):
        return {
            "is_valid": False,
            "message": "Input must be a text string.",
            "word_count": 0
        }

    stripped = text.strip()

    if not stripped:
        return {
            "is_valid": False,
            "message": "Text is empty. Please enter a news article.",
            "word_count": 0
        }

    if len(stripped) < min_length:
        return {
            "is_valid": False,
            "message": f"Text is too short ({len(stripped)} characters). "
                       f"Please enter at least {min_length} characters.",
            "word_count": len(stripped.split())
        }

    word_count = len(stripped.split())
    return {
        "is_valid": True,
        "message": "Valid input.",
        "word_count": word_count
    }

