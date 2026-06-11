"""
Feature Engineering module for Customer Support Ticket Classification.
Contains methods for transforming preprocessed text into numeric vectors:
- Bag of Words (BoW)
- Binary Vectorization
- TF-IDF Vectorization
- Hashing Trick
- Dimensionality Analysis & Out-Of-Vocabulary (OOV) Handling
"""

from typing import Dict, List, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from src.config import MAX_FEATURES, NGRAM_RANGE, MIN_DF, MAX_DF

def fit_tfidf_vectorizer(
    train_texts: List[str],
    max_features: int = MAX_FEATURES,
    ngram_range: Tuple[int, int] = NGRAM_RANGE,
    min_df: Union[int, float] = MIN_DF,
    max_df: Union[int, float] = MAX_DF
) -> TfidfVectorizer:
    """
    Fits a TF-IDF vectorizer on the training text corpus.
    Configured with parameters from src/config.py by default.
    
    Args:
        train_texts: List of preprocessed training text strings.
        max_features: Maximum number of features to keep.
        ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted.
        min_df: Minimum document frequency for terms to be kept.
        max_df: Maximum document frequency for terms to be kept.
        
    Returns:
        Fitted TfidfVectorizer instance.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )
    vectorizer.fit(train_texts)
    return vectorizer

def fit_bow_vectorizer(
    train_texts: List[str],
    binary: bool = False,
    max_features: int = MAX_FEATURES,
    ngram_range: Tuple[int, int] = NGRAM_RANGE,
    min_df: Union[int, float] = MIN_DF,
    max_df: Union[int, float] = MAX_DF
) -> CountVectorizer:
    """
    Fits a Bag-of-Words (BoW) or Binary vectorizer on the training text corpus.
    
    Args:
        train_texts: List of preprocessed training text strings.
        binary: If True, creates binary vectors.
        max_features: Maximum number of features to keep.
        ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted.
        min_df: Minimum document frequency for terms to be kept.
        max_df: Maximum document frequency for terms to be kept.
        
    Returns:
        Fitted CountVectorizer instance.
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        binary=binary
    )
    vectorizer.fit(train_texts)
    return vectorizer

def transform_texts(vectorizer: Any, texts: List[str]) -> csr_matrix:
    """
    Transforms new text documents using a pre-fitted vectorizer.
    This prevents data leakage by ensuring no validation/test terms are fit.
    Handles Out-of-Vocabulary (OOV) terms by ignoring them.
    
    Args:
        vectorizer: A fitted Vectorizer instance.
        texts: List of preprocessed text strings.
        
    Returns:
        Sparse matrix representation of the texts.
    """
    return vectorizer.transform(texts)

def apply_hashing_trick(texts: List[str], n_features: int = 10000) -> csr_matrix:
    """
    Applies the Hashing Trick using scikit-learn's HashingVectorizer.
    This is useful for online learning or to bypass explicit vocabulary storage.
    
    Args:
        texts: List of preprocessed text strings.
        n_features: Dimension of the output sparse matrix.
        
    Returns:
        Sparse matrix representation of the texts.
    """
    vectorizer = HashingVectorizer(
        n_features=n_features,
        ngram_range=NGRAM_RANGE,
        alternate_sign=True
    )
    return vectorizer.transform(texts)

def extract_vocabulary(vectorizer: Any) -> Dict[str, int]:
    """
    Extracts the vocabulary dictionary (word to index map) from the vectorizer.
    
    Args:
        vectorizer: Fitted vectorizer instance.
        
    Returns:
        Dictionary mapping features to indices.
    """
    if hasattr(vectorizer, "vocabulary_") and vectorizer.vocabulary_ is not None:
        return vectorizer.vocabulary_
    return {}

def analyze_dimensionality(sparse_matrix: csr_matrix, vectorizer: Any) -> Dict[str, Any]:
    """
    Performs dimensional analysis on the generated feature space:
    - Sparse vs Dense density metrics (percent of non-zero elements).
    - Vocab size and shape.
    - Word counts and term-frequency statistics.
    
    Args:
        sparse_matrix: Compressed Sparse Row matrix from vectorizer.
        vectorizer: Fitted vectorizer instance.
        
    Returns:
        A dictionary containing sparsity ratio, vocabulary length, and shape details.
    """
    n_rows, n_cols = sparse_matrix.shape
    total_elements = n_rows * n_cols
    non_zero_elements = sparse_matrix.nnz
    density = non_zero_elements / total_elements if total_elements > 0 else 0.0
    sparsity = 1.0 - density
    
    vocab_size = len(vectorizer.vocabulary_) if hasattr(vectorizer, "vocabulary_") and vectorizer.vocabulary_ is not None else 0
    
    # Calculate memory footprint of CSR sparse matrix
    sparse_memory_bytes = int(sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes)
    
    # Calculate dense counterpart footprint
    element_size = np.dtype(sparse_matrix.dtype).itemsize
    dense_memory_bytes = int(total_elements * element_size)
    
    return {
        "shape": sparse_matrix.shape,
        "density": density,
        "sparsity": sparsity,
        "vocab_size": vocab_size,
        "non_zero_elements": non_zero_elements,
        "sparse_memory_bytes": sparse_memory_bytes,
        "dense_memory_bytes": dense_memory_bytes,
        "memory_saving_ratio": dense_memory_bytes / sparse_memory_bytes if sparse_memory_bytes > 0 else 1.0
    }
