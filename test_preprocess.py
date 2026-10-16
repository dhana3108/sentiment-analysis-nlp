"""
Unit Tests for Text Preprocessing Module
=======================================
Validates that text cleaning, negation preservation, HTML stripping,
contraction expansion, and edge cases function correctly.
"""

import pytest
import numpy as np
from preprocess import (
    clean_text,
    expand_contractions,
    clean_html_and_urls,
    NEGATION_WORDS,
)


def test_lowercase_conversion():
    """Verify that uppercase and mixed-case text is converted to lowercase."""
    raw = "EXCELLENT Product! Super FAST Delivery."
    result = clean_text(raw)
    assert result == result.lower()
    assert "excellent" in result
    assert "product" in result


def test_missing_and_invalid_inputs():
    """Verify that None, empty strings, numbers, and NaN are handled safely without exceptions."""
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   \n\t  ") == ""
    assert clean_text(12345) == ""  # numbers get stripped of digits leaving empty
    assert clean_text(np.nan) == ""


def test_html_and_url_removal():
    """Verify that HTML tags and web URLs are removed."""
    raw = "Great product! <br /> Check details at https://amazon.com/item and www.reviews.org"
    cleaned = clean_html_and_urls(raw)
    assert "<br />" not in cleaned
    assert "https://" not in cleaned
    assert "amazon.com" not in cleaned
    assert "reviews.org" not in cleaned


def test_contraction_expansion():
    """Verify that common English contractions are expanded."""
    assert "do not" in expand_contractions("don't")
    assert "cannot" in expand_contractions("can't")
    assert "would not" in expand_contractions("wouldn't")
    assert "is not" in expand_contractions("isn't")
    assert "have not" in expand_contractions("haven't")


def test_negation_preservation():
    """
    CRITICAL TEST: Ensure negation tokens ('not', 'never', 'no', 'neither')
    are NOT removed as standard stopwords, preserving sentiment polarity.
    """
    raw = "This phone is not good, never charges, and has no battery life."
    cleaned = clean_text(raw, remove_stopwords=True)

    # All these critical sentiment modifiers must be present
    assert "not" in cleaned, "Negation word 'not' was erroneously stripped!"
    assert "never" in cleaned, "Negation word 'never' was erroneously stripped!"
    assert "no" in cleaned, "Negation word 'no' was erroneously stripped!"


def test_whitespace_normalization():
    """Verify multiple spaces, tabs, and newlines are collapsed into a single space."""
    raw = "Too   many    spaces   and\n\nnewlines\t\thovering."
    cleaned = clean_text(raw)
    assert "   " not in cleaned
    assert "\n" not in cleaned
    assert "\t" not in cleaned
    assert cleaned == "many spaces newlines hovering"


def test_punctuation_stripping():
    """Verify special symbols and punctuation are removed, leaving clean alphabetic tokens."""
    raw = "Amazing!!! Best purchase ever :) 100% recommended $$$."
    cleaned = clean_text(raw)
    assert "!" not in cleaned
    assert "%" not in cleaned
    assert "$" not in cleaned
    assert ":" not in cleaned
    assert "amazing" in cleaned
    assert "recommended" in cleaned
