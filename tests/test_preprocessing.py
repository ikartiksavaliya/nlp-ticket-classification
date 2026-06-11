"""
Unit tests for the preprocessing module.
Tests normalizations, tokenizations, and masking functions in src/preprocessing.py.
"""

import pytest
from src.utils import check_nltk_assets, check_spacy_model

# Run the asset downloads before running any tests
check_nltk_assets()
check_spacy_model()

from src.preprocessing import (
    normalize_unicode,
    handle_contractions,
    handle_emojis,
    mask_entities,
    remove_punctuation_and_numbers,
    tokenize_sentences,
    tokenize_words,
    remove_stopwords,
    apply_stemming,
    apply_lemmatization,
    detect_language,
    preprocess_text
)

def test_normalize_unicode():
    """Verify that accented Unicode characters are decomposed and normalized."""
    assert normalize_unicode("Café") == "Cafe"
    assert normalize_unicode("München") == "Munchen"
    assert normalize_unicode("facade") == "facade"

def test_handle_contractions():
    """Verify that contractions are successfully expanded (e.g., 'shouldn't' -> 'should not')."""
    assert handle_contractions("I'm fine.") == "I am fine."
    assert handle_contractions("shouldn't") == "should not"
    assert handle_contractions("they'd") == "they would"

def test_handle_emojis():
    """Verify that emojis are extracted or converted into textual representation."""
    text, emojis = handle_emojis("Hello 😊!")
    assert "smiling_face_with_smiling_eyes" in text
    assert "😊" in emojis

def test_mask_entities():
    """Verify that URLs, emails, and mentions are masked with specific tokens (<URL>, <EMAIL>, <MENTION>)."""
    text = "Contact me at test@example.com or visit http://google.com. Follow @developer."
    masked = mask_entities(text)
    assert "<EMAIL>" in masked
    assert "<URL>" in masked
    assert "<MENTION>" in masked
    assert "test@example.com" not in masked
    assert "http://google.com" not in masked
    assert "@developer" not in masked

def test_remove_punctuation_and_numbers():
    """Verify that all numeric characters and punctuations are stripped from texts, preserving masks."""
    text = "Hello, World! 123. Contact <EMAIL> at <URL> from @user."
    text = mask_entities(text)
    cleaned = remove_punctuation_and_numbers(text)
    assert "123" not in cleaned
    assert "," not in cleaned
    assert "!" not in cleaned
    assert "<EMAIL>" in cleaned
    assert "<URL>" in cleaned
    assert "<MENTION>" in cleaned

def test_tokenize_sentences():
    """Verify sentence tokenization partitions paragraphs correctly."""
    text = "First sentence. Second sentence! Is this the third?"
    sentences = tokenize_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence."
    assert sentences[1] == "Second sentence!"
    assert sentences[2] == "Is this the third?"

def test_tokenize_words():
    """Verify word tokenization splits sentences into correct lists of sub-tokens."""
    text = "Hello world, testing."
    tokens = tokenize_words(text)
    assert len(tokens) == 5
    assert tokens[0] == "Hello"
    assert tokens[2] == ","

def test_remove_stopwords():
    """Verify common stopwords are removed from token lists."""
    tokens = ["This", "is", "a", "sample", "ticket", "<URL>"]
    filtered = remove_stopwords(tokens)
    assert "is" not in filtered
    assert "a" not in filtered
    assert "ticket" in filtered
    assert "<URL>" in filtered

def test_stemming_vs_lemmatization():
    """Verify stemming reduces tokens to roots and lemmatization maps tags to lemmas correctly."""
    tokens = ["running", "cats", "studies", "<URL>"]
    stemmed = apply_stemming(tokens)
    lemmatized = apply_lemmatization(tokens)
    
    assert "run" in stemmed or "runn" in stemmed
    assert "cat" in stemmed
    assert "studi" in stemmed
    assert "<URL>" in stemmed
    
    assert "run" in lemmatized
    assert "cat" in lemmatized
    assert "study" in lemmatized
    assert "<URL>" in lemmatized

def test_detect_language():
    """Verify language detection works on a sample set of non-English and English tickets."""
    assert detect_language("This is a simple English sentence.") == "en"
    assert detect_language("Bonjour, c'est une phrase en français.") == "fr"

def test_preprocess_text():
    """Verify end-to-end orchestration of preprocess_text."""
    raw = "The payment wasn't processed! Check https://bank.com or email support@bank.com 😊."
    cleaned = preprocess_text(raw, method="lemmatize")
    assert "payment" in cleaned
    assert "process" in cleaned or "processed" in cleaned
    assert "<URL>" in cleaned
    assert "<EMAIL>" in cleaned
    assert "smiling_face_with_smiling_eyes" in cleaned
