"""
Text Preprocessing Module for Sentiment Analysis
=================================================
This module provides reusable, battle-tested text cleaning and normalization
functions specifically tailored for sentiment analysis on product reviews.

Why Preprocessing Matters in Sentiment Analysis:
------------------------------------------------
1. Case Normalization: "Good", "GOOD", and "good" convey identical sentiment.
   Converting all text to lowercase prevents feature fragmentation in TF-IDF.
2. Contraction Expansion: Expanding "don't" -> "do not", "isn't" -> "is not"
   makes negation words explicit and tokenizable.
3. Negation Preservation: Traditional stopword lists mistakenly discard words
   like 'not', 'no', 'never', 'hardly'. Discarding them turns "not good" into "good",
   inverting the sentiment polarity. We explicitly protect negation tokens.
4. Noise Reduction: Strips HTML tags (<br />), URLs, excess punctuation, and whitespace
   that add no sentiment signal.
"""

import re
import html
import unicodedata
from typing import Any, List, Set


# Common English contraction dictionary
CONTRACTION_MAP = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

# Negation words that MUST be preserved for accurate sentiment classification
NEGATION_WORDS: Set[str] = {
    "no",
    "not",
    "nor",
    "never",
    "neither",
    "nobody",
    "none",
    "nothing",
    "nowhere",
    "hardly",
    "scarcely",
    "barely",
    "cannot",
    "without",
    "against",
}

# Safe baseline stopword list (excluding all negation words)
# Using standard NLTK stopwords as base, minus negation words
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        BASE_STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        BASE_STOPWORDS = set(stopwords.words("english"))
except Exception:
    # Fallback minimal stopword set if NLTK is not available
    BASE_STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
        "at", "by", "for", "with", "about", "into", "through", "during", "before",
        "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
        "off", "over", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "only", "own", "same", "so", "than", "too",
        "very", "s", "t", "can", "will", "just", "don", "should", "now"
    }

# Remove any negation words from the stopword set
SENTIMENT_STOPWORDS: Set[str] = BASE_STOPWORDS - NEGATION_WORDS


def expand_contractions(text: str) -> str:
    """
    Expands common English contractions (e.g. "don't" -> "do not", "can't" -> "cannot").
    Prevents negation words from losing their meaning when punctuation is stripped.
    """
    # Compile regex pattern for contractions with word boundaries
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r")\b",
        re.IGNORECASE,
    )

    def replace(match):
        contracted = match.group(0).lower()
        return CONTRACTION_MAP.get(contracted, contracted)

    return pattern.sub(replace, text)


def clean_html_and_urls(text: str) -> str:
    """
    Removes HTML tags (e.g., <br />, <p>), unescapes HTML entities,
    and removes web URLs and email addresses.
    """
    # Unescape HTML entities like &amp;, &quot;, &#39;
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    return text


def clean_text(text: Any, remove_stopwords: bool = True) -> str:
    """
    Main text preprocessing pipeline for sentiment reviews.

    Steps performed:
    1. Type safety check: handles None, NaN, and non-string inputs gracefully.
    2. HTML tag and URL removal.
    3. Contraction expansion (e.g. "didn't" -> "did not").
    4. Lowercase conversion.
    5. Unicode normalization (converts accented characters to ASCII equivalents).
    6. Punctuation and digit handling: retains alphabetical characters and negations.
    7. Negation-preserving stopword filtering (optional, default=True).
    8. Whitespace normalization: collapses multiple spaces, tabs, and newlines.

    Parameters:
    -----------
    text : Any
        Input review text (string or convertible object).
    remove_stopwords : bool, optional
        Whether to filter non-sentiment stopwords (default: True).

    Returns:
    --------
    str : Cleaned and normalized review text.
    """
    # 1. Type safety check
    if text is None:
        return ""
    # Check for float NaN without requiring numpy import
    if isinstance(text, float) and (text != text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Check for textual representations of missing values
    if text.strip().lower() in ("nan", "none", "null", ""):
        return ""

    # 2. Remove HTML tags, entities, and URLs
    text = clean_html_and_urls(text)

    # 3. Expand contractions before punctuation removal
    text = expand_contractions(text)

    # 4. Lowercase conversion
    text = text.lower()

    # 5. Unicode normalization (e.g., accents -> standard ASCII)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8", "ignore")

    # 6. Remove non-alphabetical characters (keeps letters and spaces)
    # Replaces numbers, symbols, and punctuation with spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # 7. Tokenize and optionally filter stopwords (preserving negations)
    tokens: List[str] = text.split()

    if remove_stopwords:
        tokens = [token for token in tokens if token not in SENTIMENT_STOPWORDS and len(token) > 1]
    else:
        tokens = [token for token in tokens if len(token) > 1]

    # 8. Reconstruct string with normalized single spacing
    cleaned = " ".join(tokens)
    return cleaned


def preprocess_series(series, remove_stopwords: bool = True):
    """
    Applies clean_text across a pandas Series of review strings.
    """
    return series.apply(lambda x: clean_text(x, remove_stopwords=remove_stopwords))


if __name__ == "__main__":
    # Demonstration of preprocessing behavior
    sample_reviews = [
        "This product is NOT good at all! Terrible quality. <br /> Don't buy it!",
        "Absolutely LOVE this! Works perfectly and arrived in 2 days. 10/10 stars! :)",
        "It was neither fast nor durable. I wouldn't recommend it to anyone: http://example.com",
    ]

    print("=" * 60)
    print("TEXT PREPROCESSING DEMONSTRATION")
    print("=" * 60)
    for raw in sample_reviews:
        cleaned = clean_text(raw)
        print(f"RAW:     {raw}")
        print(f"CLEANED: {cleaned}")
        print("-" * 60)
