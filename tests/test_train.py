"""
Unit tests for the training pipeline in src/train.py.
"""

import os
import pytest
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import joblib

from src.train import create_model, train_baseline_model, save_model_artifacts, load_and_split_data
from src.config import MODEL_DIR, MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH

def test_create_model():
    """Test that model instantiation works for both GPU and CPU fallbacks."""
    # Test Naive Bayes
    nb_model = create_model("naive_bayes", {"alpha": 0.5}, force_cpu=True)
    assert nb_model is not None
    assert nb_model.alpha == 0.5
    
    # Test Logistic Regression
    lr_model = create_model("logistic_regression", {"C": 2.0}, force_cpu=True)
    assert lr_model is not None
    assert lr_model.C == 2.0
    
    # Test SVM
    svm_model = create_model("svm", {"C": 0.5, "kernel": "linear"}, force_cpu=True)
    assert svm_model is not None
    assert svm_model.C == 0.5
    assert svm_model.kernel == "linear"
    
    # Test XGBoost
    xgb_model = create_model("xgboost", {"n_estimators": 50}, force_cpu=True)
    assert xgb_model is not None
    assert xgb_model.n_estimators == 50

def test_train_baseline_model():
    """Verify that training the baseline Naive Bayes model fits and returns an estimator."""
    # Setup dummy training data
    np.random.seed(42)
    # 10 samples, 5 features
    x_train = csr_matrix(np.random.rand(10, 5).astype(np.float32))
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    
    model = train_baseline_model(x_train, y_train, "naive_bayes")
    assert model is not None
    
    # Check that predict works on the fitted model
    preds = model.predict(x_train)
    if hasattr(preds, "get"):
        preds = preds.get()
    assert len(preds) == 10

def test_save_model_artifacts(tmp_path):
    """Test serialization of model artifacts (model, vectorizer, label encoder)."""
    # Create dummy objects
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    
    model = MultinomialNB()
    # Fit dummy data
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])
    model.fit(X, y)
    
    vectorizer = TfidfVectorizer()
    vectorizer.fit(["dummy text here"])
    
    label_encoder = LabelEncoder()
    label_encoder.fit(["category_a", "category_b"])
    
    # Override paths using monkeypatch or mock, or just call with temp files.
    # To keep it simple, we patch config attributes
    import src.config as config
    original_model_path = config.MODEL_PATH
    original_vec_path = config.VECTORIZER_PATH
    original_le_path = config.LABEL_ENCODER_PATH
    
    temp_model_path = tmp_path / "model.joblib"
    temp_vec_path = tmp_path / "vectorizer.joblib"
    temp_le_path = tmp_path / "label_encoder.joblib"
    
    config.MODEL_PATH = str(temp_model_path)
    config.VECTORIZER_PATH = str(temp_vec_path)
    config.LABEL_ENCODER_PATH = str(temp_le_path)
    
    try:
        save_model_artifacts(model, vectorizer, label_encoder)
        
        # Verify files exist
        assert os.path.exists(temp_model_path)
        assert os.path.exists(temp_vec_path)
        assert os.path.exists(temp_le_path)
        
        # Load and verify
        loaded_model = joblib.load(temp_model_path)
        loaded_vec = joblib.load(temp_vec_path)
        loaded_le = joblib.load(temp_le_path)
        
        assert loaded_model is not None
        assert loaded_vec is not None
        assert loaded_le is not None
        assert list(loaded_le.classes_) == ["category_a", "category_b"]
    finally:
        # Restore original paths
        config.MODEL_PATH = original_model_path
        config.VECTORIZER_PATH = original_vec_path
        config.LABEL_ENCODER_PATH = original_le_path
