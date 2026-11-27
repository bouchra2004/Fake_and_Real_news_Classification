"""
Tests for the preprocessing module.
"""

import pytest
from src.preprocessing import (
    clean_text,
    remove_html_tags,
    remove_urls,
    remove_email_addresses,
    remove_special_characters,
    normalize_whitespace,
    remove_reuters_marker,
    combine_title_text,
    validate_text_input,
)


class TestRemoveHtmlTags:
    def test_removes_basic_tags(self):
        assert "hello world" in remove_html_tags("<p>hello world</p>")

    def test_removes_nested_tags(self):
        result = remove_html_tags("<div><b>bold</b> text</div>")
        assert "<" not in result
        assert "bold" in result

    def test_no_tags(self):
        assert remove_html_tags("plain text") == "plain text"


class TestRemoveUrls:
    def test_removes_http(self):
        assert "check this " in remove_urls("check this https://example.com")

    def test_removes_www(self):
        result = remove_urls("visit www.example.com today")
        assert "www" not in result

    def test_no_urls(self):
        assert remove_urls("no urls here") == "no urls here"


class TestRemoveReutersMarker:
    def test_removes_reuters_dateline(self):
        text = "WASHINGTON (Reuters) - The president said today..."
        result = remove_reuters_marker(text)
        assert "(Reuters)" not in result
        assert "The president said today" in result

    def test_keeps_text_without_marker(self):
        text = "Some normal article without dateline."
        assert remove_reuters_marker(text) == text


class TestCleanText:
    def test_full_cleaning(self):
        text = '<p>WASHINGTON (Reuters) - Visit https://example.com for info@test.com</p>'
        result = clean_text(text, remove_source_markers=True)
        assert "<p>" not in result
        assert "https://" not in result
        assert "(Reuters)" not in result
        assert "@" not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_whitespace_only(self):
        assert clean_text("   ") == ""

    def test_preserves_meaningful_content(self):
        text = "The president didn't approve the bill. That's not true."
        result = clean_text(text)
        assert "didn't" in result or "didn" in result  # apostrophe handling
        assert "president" in result

    def test_without_source_markers(self):
        text = "NEW YORK (Reuters) - Some news here."
        result = clean_text(text, remove_source_markers=False)
        assert "Reuters" in result


class TestCombineTitleText:
    def test_both_present(self):
        result = combine_title_text("Title Here", "Body text here")
        assert "Title Here" in result
        assert "Body text here" in result

    def test_title_only(self):
        result = combine_title_text("Title Here", None)
        assert result == "Title Here"

    def test_text_only(self):
        result = combine_title_text(None, "Body text")
        assert result == "Body text"

    def test_both_empty(self):
        result = combine_title_text("", "")
        assert result == ""

    def test_both_none(self):
        result = combine_title_text(None, None)
        assert result == ""


class TestValidateTextInput:
    def test_valid_text(self):
        result = validate_text_input("This is a valid news article about politics.")
        assert result["is_valid"] is True
        assert result["word_count"] > 0

    def test_empty_string(self):
        result = validate_text_input("")
        assert result["is_valid"] is False

    def test_whitespace_only(self):
        result = validate_text_input("     ")
        assert result["is_valid"] is False

    def test_too_short(self):
        result = validate_text_input("Hi", min_length=10)
        assert result["is_valid"] is False

    def test_non_string(self):
        result = validate_text_input(12345)
        assert result["is_valid"] is False

    def test_exactly_min_length(self):
        result = validate_text_input("1234567890", min_length=10)
        assert result["is_valid"] is True

