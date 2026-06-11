"""
Unit tests for the feature engineering module.
Tests fitting and transforming vectorizers, hashing tricks, and sparsity checks in src/feature_engineering.py.
"""

import pytest
import numpy as np
from src.feature_engineering import (
    fit_tfidf_vectorizer,
    fit_bow_vectorizer,
    transform_texts,
    apply_hashing_trick,
    extract_vocabulary,
    analyze_dimensionality
)

def test_fit_tfidf_vectorizer():
    """Verify that TF-IDF vectorizer fits vocabulary and registers properties properly."""
    train_texts = ["login issue account portal", "payment deducted bank failed", "bug latest update report"]
    vectorizer = fit_tfidf_vectorizer(train_texts, min_df=1, max_df=1.0)
    assert vectorizer is not None
    vocab = extract_vocabulary(vectorizer)
    assert len(vocab) > 0
    assert "login" in vocab

def test_transform_texts_no_leakage():
    """
    Verify transforming texts with a pre-fitted vectorizer does not alter vocabulary.
    Verifies that terms in the test text that were not in training are ignored (handled as OOV).
    """
    train_texts = ["login issue account portal"]
    vectorizer = fit_tfidf_vectorizer(train_texts, min_df=1, max_df=1.0)
    vocab_before = extract_vocabulary(vectorizer).copy()
    
    test_texts = ["login issue validation token unseenword"]
    transformed = transform_texts(vectorizer, test_texts)
    
    vocab_after = extract_vocabulary(vectorizer)
    assert vocab_before == vocab_after
    assert "unseenword" not in vocab_after
    assert transformed.shape[1] == len(vocab_before)

def test_apply_hashing_trick():
    """Verify HashingVectorizer maps text to fixed dimensions without fitting."""
    texts = ["login issue account portal"]
    n_features = 100
    transformed = apply_hashing_trick(texts, n_features=n_features)
    assert transformed.shape == (1, n_features)

def test_analyze_dimensionality():
    """Verify that dimensionality analysis outputs correct density, shapes, and sparsity indices."""
    train_texts = ["login issue account portal", "payment deducted bank failed", "bug latest update report"]
    vectorizer = fit_tfidf_vectorizer(train_texts, min_df=1, max_df=1.0)
    matrix = transform_texts(vectorizer, train_texts)
    
    analysis = analyze_dimensionality(matrix, vectorizer)
    assert "shape" in analysis
    assert "density" in analysis
    assert "sparsity" in analysis
    assert "vocab_size" in analysis
    assert "non_zero_elements" in analysis
    assert "sparse_memory_bytes" in analysis
    assert "dense_memory_bytes" in analysis
    assert "memory_saving_ratio" in analysis
    
    assert analysis["shape"] == matrix.shape
    assert 0.0 <= analysis["density"] <= 1.0
    assert 0.0 <= analysis["sparsity"] <= 1.0
    assert analysis["density"] + analysis["sparsity"] == pytest.approx(1.0)
    assert analysis["vocab_size"] == len(vectorizer.vocabulary_)
